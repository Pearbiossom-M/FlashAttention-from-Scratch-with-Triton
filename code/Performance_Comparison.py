import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.amp import autocast
from typing import Literal, Tuple

# ==================== Benchmark 核心函数 ====================

def benchmark_attention(
    provider: Literal['naive', 'triton', 'pytorch'],
    mode: Literal['fwd', 'bwd', 'fwd_bwd'],
    B: int,
    H: int,
    S_q: int,
    S_k: int,
    D: int,
    is_causal: bool,
    device: torch.device,
    warmup: int = 10,
    repeat: int = 30,
) -> Tuple[float, float]:
    """
    Benchmark FlashAttention
    
    Args:
        provider: 'naive' (纯 Python), 'triton' (我们的实现), 'pytorch' (官方)
        mode: 'fwd' (仅前向), 'bwd' (仅反向), 'fwd_bwd' (前向+反向)
        B, H, S_q, S_k, D: batch, heads, seq_len of Q, seq_len of K, head_dim
        is_causal: 是否启用 causal mask
        device: 设备
        warmup: warmup 次数
        rep: 重复次数
           
    Returns:
        (avg_time_ms, tflops)
    """
    # 准备数据
    Q = torch.randn(B, H, S_q, D, device=device, dtype=torch.float16, requires_grad=True)
    K = torch.randn(B, H, S_k, D, device=device, dtype=torch.float16, requires_grad=True)
    V = torch.randn(B, H, S_k, D, device=device, dtype=torch.float16, requires_grad=True)
    dO = torch.randn(B, H, S_q, D, device=device, dtype=torch.float16)
    
    # 选择实现
    if provider == "naive":
        def fn():
            return naive_attention(Q, K, V, is_causal)
    elif provider == "triton":
        from My_FlashAttention_optimized import flash_attention
        def fn():
            return flash_attention(Q, K, V, is_causal)
    else: # pytorch
        def fn():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                with autocast(device_type="cuda", dtype=torch.float16):
                    return F.scaled_dot_product_attention(
                        Q, K, V, is_causal=is_causal
                    )

    # 让梯度状态干净
    Q.grad = None
    K.grad = None
    V.grad = None

    # 根据 mode 选择测试内容
    # bwd 的测量使用 fwd_bwd - fwd
    if mode == 'fwd':
        def run_fn():
            O = fn()
            return O
    elif mode == 'fwd_bwd':
        def run_fn():
            O = fn()
            O.backward(dO)
            Q.grad = None
            K.grad = None
            V.grad = None
            return O
    else: # bwd
        def run_fn_fwd():
            O = fn()
            return O
        def run_fn_all():
            O = fn()
            O.backward(dO)
            Q.grad = None
            K.grad = None
            V.grad = None
            return O
   
    # Benchmark
    # Timing (CUDA Events)
    if mode == 'bwd':
        avg_time_ms = timing(run_fn_all, warmup, repeat) - timing(run_fn_fwd, warmup, repeat)
    else:
        avg_time_ms = timing(run_fn, warmup, repeat)

    # 计算 TFLOPS
    # 参考：https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py
    # Forward: 4 * B * H * S_q * S_k * D (2 matmuls: Q@K^T, P@V, each costs 2*S_q*S_k*D FLOPs)
    # Backward: 大约是 Forward 的 2.5 倍
    flops = 4 * B * H * S_q * S_k * D // (2 if is_causal else 1)
    if mode == 'fwd':
        tflops = flops / (avg_time_ms * 1e-3) / 1e12
    elif mode == 'bwd':
        tflops = 2.5 * flops / (avg_time_ms * 1e-3) / 1e12  # 2.5x forward
    else: # fwd_bwd
        tflops = 3.5 * flops / (avg_time_ms * 1e-3) / 1e12  # 1x forward + 2.5x forward

    return avg_time_ms, tflops

def timing(run_fn, warmup, repeat):
    # Warmup
    for _ in range(warmup):
        run_fn()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter.record()
    for _ in range(repeat):
        run_fn()
    ender.record()
    torch.cuda.synchronize()
    elapsed = starter.elapsed_time(ender)
    
    avg_time_ms = elapsed / repeat
    return avg_time_ms

def naive_attention(Q, K, V, is_causal):
    """
    纯 Python 实现的 Attention
    """
    scale = 1 / (Q.shape[-1] ** 0.5)
    S = Q @ K.transpose(-2, -1) * scale
    
    if is_causal:
        seq_len = S.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=S.device), diagonal=1).bool()
        S = S.masked_fill(mask, float('-inf'))
    
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O

if __name__ == '__main__':
    DEVICE = torch.device(torch.cuda.current_device())
    #for provider in ['naive']:
    for provider in ['pytorch', 'triton']:
        result = []
        print(f"{'='*20} {provider} {'='*20}")
        for S in [512, 1024, 2048, 4096, 8192, 16384]:
            avg_time_ms, tflops = benchmark_attention(
                provider = provider,
                mode = 'fwd_bwd',
                B = 4,
                H = 8,
                S_q = S,
                S_k = S,
                D = 128,
                is_causal = True,
                device = DEVICE,
            )
            result.append(tflops)
            #print(f"avg_time_ms: {avg_time_ms:.8f}ms")
            #print(f"tflops: {tflops}")
        print(result)