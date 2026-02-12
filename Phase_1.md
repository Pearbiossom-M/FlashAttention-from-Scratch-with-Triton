# 从零实现 FlashAttention（Phase 1）：Naive Attention —— 暴力实现的死胡同

目标：实现一个逻辑正确，但物理上会被 SRAM 容量限制死的 Naive Attention。让大家亲眼看到 SRAM 是如何被随序列长度增大的中间矩阵撑爆的，从而理解为什么 K 和 V 矩阵必须分块？为什么必须引入 online softmax？

**读者提示（重要）**

**在此处，我们会在代码中暂时借用 General Matrix Multiply（GEMM） 的 M/N/K 视角来描述 Attention。**

这样做是因为在算法原型的探索阶段，我们将暂时忽略 Batch 和 Head 维度，专注于最核心的矩阵计算逻辑。此时，Attention 的计算本质上就是一个**分块矩阵乘 + online softmax**。使用大家熟悉的 GEMM 视角，可以降低认知负担，帮助我们专注于理解 Block（分块）、Tile（切片）的运作机制。

当我们在后续章节（**Phase 3**）完成核心逻辑的构建，准备将其扩展为支持多维 tensor 的工业级算子时，我们会再将命名回归到大家熟悉的 `Seq_len` / `Head_dim` 语义。加油！

------

## 1.1 Pytorch 的参考实现

这是我们的 “标准答案”，用于后续实现的正确性校验。

```python
import torch.nn.functional as F

def bench_attention(q, k, v): # shape: [seq_len, head_dim]
    scale = 1 / (q.size(-1) ** 0.5)
    s = q @ k.transpose(-2, -1) * scale
    p = F.softmax(s, dim=-1)
    o = p @ v
    return o
```

**值得说明的是：**

FlashAttention 虽然在**逻辑上**处理的是 **4D 张量** `[B, H, S, D]`，但在 kernel 设计上，将 `(B, H)` **合并映射**到 `tl.program_id(1)`，将 `(S, D)` 通过 block tiling 映射到 `tl.program_id(0)`，**使 kernel 内部只处理 2D 张量**，从而将整个 attention 计算转化为高度优化的 **block GEMM** 任务调度问题。这种让 `(B, H)` 维度作为 “**粗粒度并行**”，让 `(S, D)` 作为 “**细粒度并行**” 的映射方式，是 FlashAttention 获得极高性能的核心原因之一。需要注意，通常不建议将 `(B, H)` 映射到 `tl.program_id(0)`，将 `(S, D)` 映射到 `tl.program_id(1)`。

>因为 `(S, D)` 方向的 block 计算对内存连续性与数据复用极其敏感，必须映射到 `tl.program_id(0)` 以最大化数据局部性与跨 block 的缓存复用，而 `(B, H)` 具有天然的独立性与粗粒度并行特征，放在 `tl.program_id(1)` 更符合 GPU 的调度特性。

------

## 1.2 第一次尝试 —— 物理之墙

我们在 Phase 0 中已经知道，要提高标准 attention 的计算效率，就必须减少 HBM 访存次数，最好的做法就是 kernel fusion，将中间结果保存在寄存器和缓存中。可我们也知道，attention 的中间结果大小随序列长度指数增加，而寄存器和缓存容量毕竟有限，该怎么缓解存储压力呢？

一种方法是：**分块**，既然完整加载全部矩阵会装不下，那一次处理一小块不就行了吗？于是得出**方案 A**：

> 假设：输入矩阵 Q，K 和 V，形状分别为：`[M, K_dim]`，`[N, K_dim]` 和 `[N, K_dim]`

* 只分块 Q，每块大小为 [BLOCK_M, K_dim]
* 由于标准 softmax 需要一整行的完整数据，所以矩阵 K 必须完整加载
* 每个 GPU program 负责计算一块 Q 的结果

### 1.2.1 模拟 GPU 分块实现

这一步先用 Python 模拟 GPU 分块实现，以便于大家理解，下一节只需使用 triton 语法翻译即可。

```python
import torch.nn.functional as F
# 假装这是 GPU 的一个 Program 正在处理第 pid 个块
def attention_kernel_sim_q_blocked(q, k, v, pid, BLOCK_M):
    """
    Q 分块 + Q 和 V 不分块的 naive attention 模拟实现
    Q: [M, K_dim]
    K: [N, K_dim]
    V: [N, K_dim]
    """
    # 1. 当前 program 负责的 Q block
    start = pid * BLOCK_M
    end = start + BLOCK_M
    q_block = q[start:end, :] # shape: [BLOCK_M, K_dim]
    
    # 2. 加载所有的 K 和 V
    # 假设 SRAM 无限大，我们把整个 K, V 都读进来
    k_all = k[:, :] # shape: [N, K_dim]
    v_all = v[:, :] # shape: [N, K_dim]
    
    # 3. 计算 s
    # shape: [BLOCK_M, K_dim] @ [K_dim, N] -> [BLOCK_M, N]
    scale = 1 / (q.size(-1) ** 0.5)
    s_block = q_block @ k_all.T * scale
    
    # 4. 计算 p - 标准 Softmax，需要整行数据
    # shape: [BLOCK_M, N]
    p_block = F.softmax(s_block, dim=-1)
    
    # 5. 计算 o
    # shape: [BLOCK_M, N] @ [N, K_dim] -> [BLOCK_M, K_dim]
    o_block = p_block @ v_all
    
    return o_block
```

可以发现，为了计算一小块 Q，需要存储完整的 K 和 V，以及大中间矩阵 `s_block` 和 `p_block`，理论上当序列长度增加，SRAM 必然会被撑爆。

### 1.2.2 方案 A 的 triton 实现

```python
import triton
import triton.language as tl

@triton.jit
def naive_attention_q_blocked_kernel(
    # -------------------- 指针 --------------------
    q, k, v, o,            # 输入输出矩阵指针

    # -------------------- stride --------------------
    # stride 的单位是 element, 不是 byte
    stride_qm, stride_qk,  # Q 在两个维度上的 stride
    stride_km, stride_kk,  # K 在两个维度上的 stride
    stride_vm, stride_vk,  # V 在两个维度上的 stride
    stride_om, stride_ok,  # O 在两个维度上的 stride

    # -------------------- 维度信息 --------------------
    M, 
    N: tl.constexpr, 
    K_dim: tl.constexpr,   # Q:[M,K_dim], K:[N,K_dim], V:[N,K_dim], O:[M,K_dim]

    # -------------------- 配置参数 ---------------------
    BLOCK_M: tl.constexpr, # 每个 program 负责 BLOCK_M 行 Q 和 O
):
    """
    Q 分块 + K 和 V 不分块 的 naive attention Triton 实现
    """
    pid = tl.program_id(0)

    # 1. 计算当前 program 负责的 Q block 的 offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, K_dim)

    mask_m = offs_m < M

    # 2. 加载 Q block, shape: [BLOCK_M, K_dim]
    q_ptrs = q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 3. 加载完整 K, shape: [N, K_dim]
    offs_n = tl.arange(0, N)
    k_ptrs = k + offs_n[:, None] * stride_km + offs_k[None, :] * stride_kk
    kk = tl.load(k_ptrs)

    # 4. 计算 s = q @ K^T, shape: [BLOCK_M, N]
    scale = 1 / (K_dim ** 0.5)
    s = tl.dot(q_block, tl.trans(kk)) * scale

    # 5. 计算 softmax, shape: [BLOCK_M, N]
    m_i = tl.max(s, 1) # 找到每一行的最大值
    p = tl.exp(s - m_i[:, None])
    l_i = tl.sum(p, 1) # 每一行求和
    p = p / l_i[:, None]

    # 6. 完整加载 V 并转化为fp32
    v_ptrs = v + offs_n[:, None] * stride_vm + offs_k[None, :] * stride_vk
    vv = tl.load(v_ptrs).to(tl.float32)

    # 7. 计算 output
    output = tl.dot(p, vv)

    # 8. 写回 o
    o_ptrs = o + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, output, mask=mask_m[:, None])
```

接下来，我们就通过实验来验证大序列长度是否真的会撑爆 SRAM。

* 设置：K_dim=128，BLOCK_M=64，使用 RTX 5060ti 进行实验

* 当 M = N = 128，正确执行

* 当 M = N = 256，我们将看到：

  ```python
  triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. # 爆了
  ```

说明尽管工程上可以实现，但完整加载 K、V 并保存大中间矩阵 s 和 p 的方法，还是会带来极高的撑爆 SRAM 的风险。需要注意，在实际工业场景中，序列长度往往是几k~几百k，对 SRAM 的消耗远高于此，所以该方案在工业场景中几乎没有应用可能。

------

## 1.3 第二次尝试 —— 逻辑之墙

只对 Q 矩阵分块，SRAM 的占用依然很高，大家很容易想到：能不能进一步减少 SRAM 消耗呢？比如把 V 也进行分块。

于是，我们想出了理论上更优的**方案 B**：

* 对 Q、V 分块，其余同上

### 1.3.1 模拟 GPU 分块实现

还是先用 Python 模拟 GPU 分块实现，便于大家理解。

```python
import torch.nn.functional as F

def attention_kernel_sim_qv_blocked(q, k, v, pid, BLOCK_M, BLOCK_N):
    """
    Q 分块 + V 分块 + K 不分块 的 naive attention 模拟实现
    Q: [M, K_dim]
    K: [N, K_dim]
    V: [N, K_dim]
    """
    # 1. 当前 program 负责的 Q block
    start = pid * BLOCK_M
    end = start + BLOCK_M
    q_block = q[start:end, :]  # [BLOCK_M, K_dim]
    
    # 2. 加载完整 K
    k_all = k[:, :] # [N, K_dim]

    # 3. 计算完整 s 和 p, 仍然会产生巨大的 [BLOCK_M, N] 中间矩阵
    scale = 1 / (q.size(-1) ** 0.5)
    s_block = q_block @ k_all.T * scale    # [BLOCK_M, N]
    p_block = F.softmax(s_block, dim=-1)   # [BLOCK_M, N]

    # 4. 对 V 分块并累加输出
    N = v.shape[0]
    K_dim = v.shape[1]
	
    # 初始化 o_block
    o_block = torch.zeros((BLOCK_M, K_dim), device=q.device, dtype=q.dtype)
    
	# 计算 o 并累加
    for v_start in range(0, N, BLOCK_N):
        v_end = min(v_start + BLOCK_N, N)
        v_block = v[v_start:v_end, :]     # [BLOCK_N, K_dim]
        p_sub = p_block[:, v_start:v_end] # [BLOCK_M, BLOCK_N]
        o_block += p_sub @ v_block        # [BLOCK_M, BLOCK_N] @ [BLOCK_N, K_dim] -> [BLOCK_M, K_dim]

    return o_block
```

虽然我们避免了完整加载 V，但由于 s_block 和 p_block 仍然必须完整存在，所以理论上 SRAM 依然很容易被中间矩阵撑爆。

### 1.3.2 方案 B 的 triton 实现

```python
import triton
import triton.language as tl

@triton.jit
def naive_attention_qv_blocked_kernel(
    # -------------------- 指针 --------------------
    q, k, v, o,            # 输入输出矩阵指针

    # -------------------- stride --------------------
    stride_qm, stride_qk,  # Q 在两个维度上的 stride
    stride_km, stride_kk,  # K 在两个维度上的 stride
    stride_vm, stride_vk,  # V 在两个维度上的 stride
    stride_om, stride_ok,  # O 在两个维度上的 stride

    # -------------------- 维度信息 --------------------
    M, 
    N: tl.constexpr, 
    K_dim: tl.constexpr,   # Q:[M,K_dim], K:[N,K_dim], V:[N,K_dim], O:[M,K_dim]

    # -------------------- 配置参数 ---------------------
    BLOCK_M: tl.constexpr, # 每个 program 负责 BLOCK_M 行 Q 和 O
    BLOCK_N: tl.constexpr  # 每次迭代负责 BLOCK_N 行 V
):
    """
    Q 分块 + V 分块 + K 不分块 的 naive attention Triton 实现
    """
    pid = tl.program_id(0)

    # 1. 计算当前 program 负责的 Q block 的 offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, K_dim)

    mask_m = offs_m < M

    # 2. 加载 Q block, shape: [BLOCK_M, K_dim]
    q_ptrs = q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 3. 加载完整 K, shape: [N, K_dim]
    offs_n = tl.arange(0, N)
    k_ptrs = k + offs_n[:, None] * stride_km + offs_k[None, :] * stride_kk
    kk = tl.load(k_ptrs)

    # 4. 计算 s = q @ K^T, shape: [BLOCK_M, N]
    scale = 1 / (K_dim ** 0.5)
    s = tl.dot(q_block, tl.trans(kk)) * scale

    # 5. 计算 softmax, shape: [BLOCK_M, N]
    m_i = tl.max(s, 1) # 找到每一行的最大值
    p = tl.exp(s - m_i[:, None])
    l_i = tl.sum(p, 1) # 每一行求和
    p = p / l_i[:, None]

    # 6. 分块加载 V 并累加输出
    output = tl.zeros((BLOCK_M, K_dim), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n_blk = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n_blk < N

        # 读取 V block 并转化为 float32, shape: [BLOCK_N, K_dim]
        v_ptrs = v + offs_n_blk[:, None] * stride_vm + offs_k[None, :] * stride_vk
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # 取对应 p block, shape: [BLOCK_M, BLOCK_N]
        p_sub = tl.where(mask_n[None, :], p[:, offs_n_blk], 0.0)

        # 累加, shape: [BLOCK_M, K_dim]
        output += tl.dot(p_sub, v_block)

    # 7. 写回 o
    o_ptrs = o + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, output, mask=mask_m[:, None])
```

大家有没有尝试运行这个 kernel呢？发现什么了吗？没错，它根本运行不了！

错在哪里呢？问题出在分块加载 V 并累加输出的循环中的 `p_sub = tl.where(mask_n[None, :], p[:, offs_n_blk], 0.0)`，在 Triton 中：只能对“指针 + 偏移”做动态访问，而不能对**“中间 tensor”**做**动态切片**，也就是说这里取 `p_sub` 的操作 `p[:, offs_n_blk]` 非法。

这个看似更优的只完整加载 K，分块加载 Q、V 的方案，因为触及了工程能力边界——无法使用 triton 实现而失败。

------

## 1.4 小结

到目前为止，我们已经穷尽了所有在 **不改变 softmax 计算方式** 的前提下，缓解标准 attention 算法内存瓶颈的方法。

仿佛被逼进了死角：

* **想全读进来？** 物理硬件（SRAM）说：**太大了，滚。** (方案 A 失败)
* **想分批处理？** 编程模型（Compiler）说：**不支持动态切片，滚。** (方案 B 失败)

真的到极限了吗？不，还没有！相信大家已经发现这所有问题的症结 —— **标准 softmax** 公式。只要我们还抱着这个公式不放，就必须持有完整的 K 矩阵和 $N^2$ 级中间矩阵，必然导致 SRAM 爆炸。

要破局，我们就需要一种算法，它能在只看到局部数据的情况下，计算局部 softmax 结果，并在看到新数据时修正之前的结果 —— 这样就可以流式计算，不用持有全部数据了。那有这样的算法吗？有的兄弟，有的。而且你已经见过它了，它就是 Phase 0 中介绍过的 **online softmax**。

到这里，大家应该已经认识到了：

* **online softmax** 是 attention 高性能实现的**唯一选择**
* FlashAttention 不是“更快的 GEMM”，而是完全的**计算路径重构**

下一章，我们将正式进入 FlashAttention。