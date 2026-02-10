# Phase 2：FlashAttention 的算法核心 —— 物理与逻辑的双重突围

目标：

* 完成一个 FlashAttention 前向传播的**最小实现**（Single Batch，Single Head，无 Causal Mask）

* 让大家第一次真正写出来 FlashAttention 的灵魂 —— **Inner Loop：online softmax + 流式累加 output**
* 通过算法变革（online softmax），同时破解 SRAM 容量爆炸（**物理之墙**）和 Triton 动态切片限制（**逻辑之墙**）

注意事项：

* **Kernel 内部**：只计算一个 `Q_block`（`[BLOCK_M, K_dim]`），然后沿着序列维度 `N` 扫描所有 `K/V block`
* 虽然这只是 FlashAttention 前向传播的最小实现，但它包含了 FlashAttention **最关键、最难理解**的部分：**“看到新 block 时，如何修正旧 softmax 的归一化基准，并同步修正累加的输出向量。”**

------

## 2.1 FlashAttention 的核心思路 —— “如何破墙”

在 Phase 1 中，我们已经看到两堵墙：

- **物理之墙**：
   完整加载 K/V 或持有 `[BLOCK_M, N]` 的中间矩阵，SRAM 一定会爆
- **逻辑之墙**：
   在 Triton 中，无法对中间 tensor（如 `p`）做动态切片，导致完整持有中间矩阵 `p` 时，即使想通过对 V 分块来降低 SRAM 占用，编译器也不允许。

FlashAttention 的做法是使用 **online softmax**：

* **不存完整的中间矩阵，现算现用，用完即弃**
* 既然不存完整的中间矩阵，自然就不存在“切片”这个操作，**逻辑之墙**也就自然消失了
* 由于 SRAM 不存完整矩阵，只存当前处理的那一小块 K/V 和中间矩阵，内存占用也就不再随 N 增长，**物理之墙**随之消失。

------

## 2.2 FlashAttention 前向传播的最小实现

>这里的“最小实现”指的是：**在不引入 batch / head / causal mask 等工程维度的前提下，完整覆盖 FlashAttention 前向传播的计算语义。**

说明：

* 输入矩阵 Q、K 和 V 的形状分别是 [M, K_dim]，[N, K_dim] 和 [N, K_dim]
* 这里只取 Q 的一个 block 参与计算，形状为：[BLOCK_M, K_dim]

**关键：**K/V 仍然完整参与计算，但**不一次性加载，而是沿 N 分块扫描**

### 2.2.1 Inner Loop 算法流程详解

#### 步骤一：初始化统计量

online softmax 的统计量是 **Per-row** 的，因此各个统计量并不是标量。

```python
# 假设 Q_block 的形状为：[BLOCK_M, K_dim]
m = -inf # shape: [BLOCK_M,]
l = 0    # shape: [BLOCK_M,]
o = 0    # shape: [BLOCK_M, K_dim]
```

* **m**：当前已处理数据的**最大值**，作为数值稳定性的锚点，所有 exp 计算都相对于这个基准
* **l**：归一化分母，即 $\sum_j e^{s_{ij} - m_i} $，用于**最后的归一化**
* **o**：**未归一化**的输出累加器，即 $\sum_j e^{s_{ij} - m_i} \cdot V_j $，维护当前的"加权和"，最后除以 l 得到归一化的输出

#### 步骤二：逐块处理 K/V

```python
for start_n in range(0, N, BLOCK_N):
    # 加载当前 K block 和 V block
    K_block = K[start_n : start_n+BLOCK_N, :]  # [BLOCK_N, K_dim]
    V_block = V[start_n : start_n+BLOCK_N, :]  # [BLOCK_N, K_dim]
    
    # 计算局部 score
    scale = 1 / (q.size(-1) ** 0.5)
    s_block = Q_block @ K_block.T * scale  # [BLOCK_M, BLOCK_N]
    
    # 更新统计量（核心！）
    m_new = max(m, row_max(s_block))
    
    # 修正旧的累加器
    correction = exp(m - m_new)
    l = l * correction + row_sum(exp(s_block - m_new))
    o = o * correction + (exp(s_block - m_new) @ V_block)
    
    # 更新 m
    m = m_new
```

注意：

* `correction = exp(m - m_new)`，不要把 `m` 和 `m_new` 的位置写反了哦！

* 如果忘记了这个公式，可以参考 Phase 0 的”**0.2.3 Online Softmax 的数学推导**“

* 看不明白也没关系，可以这样理解：

  想一想 softmax 的公式（ `exp(score_block - m)` ），原本的累加器（l 和 o）是基于更小的 `m` 计算的，相比于真实值，是不是就偏大了？那要修正，是不是就需要乘上一个小于 1 的修正因子？那 `exp(m - m_new)` 和 `exp(m_new - m)` 哪一个小于 1 呢？当然是 `exp(m - m_new)` 了！所以，就有了：`correction = exp(m - m_new)`

#### 步骤三：最终归一化

```python
o_final = o / l[:, None] # o 的每一行都除以对应的 l
```

注意：

* online softmax 并不生成中间矩阵 `p`，归一化被移动到了最后

### 2.2.2 Python 模拟实现

现在我们把上述步骤串起来，依然先用 python 模拟实现，便于大家理解。

```python
import torch
import torch.nn.functional as F

def flash_attention_forward_sim(Q_block, K, V, BLOCK_N):
    """
    FlashAttention forward pass 模拟实现
    Q_block: [BLOCK_M, K_dim]
    K, V: [N, K_dim]
    """
    BLOCK_M, K_dim = Q_block.shape
    N = K.size(0)
    scale = 1 / (K_dim ** 0.5)
    device = Q_block.device

    # 初始化统计量
    m = torch.full((BLOCK_M,), float('-inf'), dtype=torch.float32, device=device)
    l = torch.zeros((BLOCK_M,), dtype=torch.float32, device=device)
    o = torch.zeros((BLOCK_M, K_dim), dtype=torch.float32, device=device)

    # 逐块处理 K/V
    for start_n in range(0, N, BLOCK_N):
        # 加载 K block 和 V block
        K_block = K[start_n: start_n+BLOCK_N, :] # [BLOCK_N, K_dim]
        V_block = V[start_n: start_n+BLOCK_N, :] # [BLOCK_N, K_dim]

        # 计算 score 
        s_block = Q_block.to(torch.float32) @ K_block.T.to(torch.float32) * scale # [BLOCK_M, BLOCK_N]

        # 更新统计量
        m_new = torch.maximum(m, s_block.max(dim=1)[0]) # [BLOCK_M,]

        correction = torch.exp(m - m_new) # [BLOCK_M,]
        numerator = torch.exp(s_block - m_new[:, None]) # [BLOCK_M, BLOCK_N]
        l = l * correction + torch.sum(numerator, dim=1) # [BLOCK_M,]
        o = o * correction[:, None] + numerator @ V_block.to(torch.float32) # [BLOCK_M, K_dim]
        
        m = m_new

    # 最终归一化
    o_final = o / l[:, None] # [BLOCK_M, K_dim]
    return o_final
```

### 2.2.3 triton 实现

将上述 python 实现翻译为 triton kernel，看看还会不会出现那堵**”逻辑之墙“**？

```python
import triton
import triton.language as tl

@triton.jit
def flash_attention_forward_kernel(
    # -------------------- 指针 --------------------
    Q_ptr, K_ptr, V_ptr, O_ptr, # 输入输出矩阵指针

    # -------------------- stride --------------------
    stride_qm, stride_qk,  # Q 在两个维度上的 stride
    stride_km, stride_kk,  # K 在两个维度上的 stride
    stride_vm, stride_vk,  # V 在两个维度上的 stride
    stride_om, stride_ok,  # O 在两个维度上的 stride

    # -------------------- 缩放因子 --------------------
    scale, # 1 / sqrt(K_dim)
    
    # -------------------- 维度信息 --------------------
    # Q:[M,K_dim], K:[N,K_dim], V:[N,K_dim], O:[M,K_dim]
    M,       # 序列长度 (Q 的行数)
    N: tl.constexpr,       # 序列长度 (K/V 的行数)
    K_dim: tl.constexpr,   # head_dim

    # -------------------- 配置参数 ---------------------
    BLOCK_M: tl.constexpr, # Q_block 的行数
    BLOCK_N: tl.constexpr, # 流式扫描 K/V 的列块大小 (沿 N 维)
):
    """
    FlashAttention forward pass (最小实现)
    """
    pid_0 = tl.program_id(0)
    m_offs = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M,]
    k_offs = tl.arange(0, K_dim)                      # [K_dim,]

    # mask：处理最后一个 Q_block, 因为可能越界
    mask_m = m_offs < M
    
    # load Q block
    q_ptrs = Q_ptr + m_offs[:, None] * stride_qm + k_offs[None, :] * stride_qk
    q_block = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # 初始化统计量
    m = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    o = tl.zeros([BLOCK_M, K_dim], tl.float32)

    LOG2_E = 1.44269504 # log2(e), 用于tl.exp 到 tl.exp2 的转化
    
    # 逐块处理 K/V block
    for start_n in range(0, N, BLOCK_N):
        # load K/V block
        n_offs = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + n_offs[:, None] * stride_km + k_offs[None, :] * stride_kk
        v_ptrs = V_ptr + n_offs[:, None] * stride_vm + k_offs[None, :] * stride_vk
        mask_n = n_offs < N
        k_block = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0) # [BLOCK_N, K_dim]
        v_block = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0) # [BLOCK_N, K_dim]

        # 计算 score, [BLOCK_M, K_dim] @ [K_dim, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q_block, tl.trans(k_block)) * scale  # [BLOCK_M, BLOCK_N]
        s = tl.where(mask_n[None, :], s, float('-inf')) # [BLOCK_M, BLOCK_N]

        # 更新统计量
        m_new = tl.maximum(m, tl.max(s, axis=1)) # [BLOCK_M,]

        # 使用 tl.exp2 比 tl.exp 更快
        # 在很多 GPU 后端里，exp2 往往比 exp 更容易映射到高效的实现路径
		# 因此这里用 exp2(x * log2(e)) 来替代 exp(x)
        # 这属于常见的工程优化，细节依赖具体架构与编译器实现
        correction = tl.exp2((m - m_new) * LOG2_E) # [BLOCK_M,]
        numerator = tl.exp2((s - m_new[:, None]) * LOG2_E) # [BLOCK_M, BLOCK_N]

        l = l * correction + tl.sum(numerator, axis=1) # [BLOCK_M,]
        o = o * correction[:, None] + tl.dot(numerator.to(tl.float16), v_block) # [BLOCK_M, K_dim]

        m = m_new
    
    # 最终归一化
    o_final = o / l[:, None]

    # write back to O_ptr
    o_ptrs = O_ptr + m_offs[:, None] * stride_om + k_offs[None, :] * stride_ok
    tl.store(o_ptrs, o_final, mask=mask_m[:, None])
```

大家注意看，代码中再也没有出现对中间矩阵的切片操作。因为中间矩阵本身就是在这个 `for` 循环里根据当前分块计算出来的**局部变量**。我们不是在切分蛋糕，而是一次只烤出一小块蛋糕直接吃掉。

**flash_attention_forward_kernel** 的成功实现，说明 online softmax 逐块处理 K/V，不再一次性计算 `p` 矩阵的思路，的确从根本上避免了动态切片的需求，**“逻辑之墙”**自动瓦解。

至于另一堵墙，大家还记得 **Phase 1 的 1.2 节**中，我们对**方案 A** 做的那个实验吗？当时，我们尝试通过实验来验证大序列长度是否真的会撑爆 SRAM，并设置：K_dim=128，BLOCK_M=64，使用 RTX 5060ti 进行实验，得到的结果是，当 N = 256 时，程序就因 SRAM 被撑爆而崩溃。那么现在我们可以再做一次实验：

设置：BLOCK_M=64，BLOCK_N=64，K_dim=128。依旧使用 RTX 5060ti 进行实验，kernel 的**调用函数**、**正确性校验基准**以及**校验函数**如下：

```python
import torch
import torch.nn.functional as F

# 正确性校验基准
def bench_attention(q, k, v):
    scale = 1 / (q.size(-1) ** 0.5)
    s = q @ k.transpose(-2, -1) * scale
    p = F.softmax(s, dim=-1)
    o = p @ v
    return o

# kernel 的调用函数
def launch_kernel(M, N, K_dim, BLOCK_M, BLOCK_N, device):
    
    Q = torch.randn((M, K_dim), dtype=torch.float16, device=device)
    K = torch.randn((N, K_dim), dtype=torch.float16, device=device)
    V = torch.randn((N, K_dim), dtype=torch.float16, device=device)
    O = torch.empty((M, K_dim), dtype=torch.float16, device=device)

    grid = (triton.cdiv(M, BLOCK_M),)
    flash_attention_forward_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        scale=1 / (K_dim ** 0.5),
        M=M, N=N, K_dim=K_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,        
        num_warps=4,
        num_stages=3,
    )

    bench = bench_attention(
        Q.to(torch.float32), 
        K.to(torch.float32), 
        V.to(torch.float32)
    ).to(torch.float16)

    return O, bench

# 正确性校验函数
def verify_results(bench, triton_output, name="Attention"):
    # 将结果转为 fp32 进行指标计算，避免计算指标时引入二次误差
    b = bench.to(torch.float32)
    t = triton_output.to(torch.float32)
    diff_abs = torch.abs(b - t)

    # 1. 计算绝对误差
    max_abs_err = torch.max(diff_abs).item()
    mean_abs_err = torch.mean(diff_abs).item()
    
    # 2. 计算相对误差 (加上 epsilon 避免除零)
    rel_err = diff_abs / (torch.abs(b) + 1e-5)
    max_rel_err = torch.max(rel_err).item()
    
    # 3. 余弦相似度
    cosine_sim = torch.nn.functional.cosine_similarity(
        b.flatten(), t.flatten(), dim=0
    ).item()
    
    print(f"[{name} Verification]")
    print(f"Max Abs Error: {max_abs_err:.2e}")
    print(f"Mean Abs Error: {mean_abs_err:.2e}")
    print(f"Max Rel Error: {max_rel_err:.2e}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")
    
    # 4. 判定标准, 对于 fp16: 
    is_allclose = torch.allclose(b, t, rtol=1e-2, atol=1e-3)
    
    if is_allclose and cosine_sim > 0.999:
        print("✅ Test Passed!")
    else:
        print("❌ Test Failed!")
```

测试不同的序列长度，结果如下：

* 当 N = 128，正确执行
* 当 N = 256，正确执行
* 当 N = 500，正确执行
* ……
* 当 N = 4096，仍然正确执行，且不会 OOM

由此可见，我们真的通过 **online softmax + 流式累加 output** 破开了之前困扰我们的**”物理之墙“** 。

恭喜你，已经完成了 FlashAttention 最艰难的部分！👏👏

------

## 2.3 小结

本章我们通过 **online softmax + 流式累加 output** 成功破开了attention 的物理与逻辑之墙。同时也带大家完整学习了 FlashAttention 前向传播的**最小实现**，希望大家可以从上述 python 和 triton 实现中真正掌握 online softmax 的算法流程。

那么现在，大家可以尝试问自己两个问题，：

* FlashAttention 相比于标准 attention 改变了哪一步的计算顺序？
* 为什么 Phase 1 的方案 B（Q/V 分块但不分块 K）会在 Triton 工程层面卡死，而 Phase 2 却能顺畅实现？

如果你已经可以轻松回答，请先给自己束个大拇指👍，再说一句：老己，你真棒！然后，就可以进入下一章啦。在下一章（Phase 3）中，我们将逐步让这个 FlashAttention kernel 走向工业级，为其增加**通用性扩展和工程级优化**。加油！
