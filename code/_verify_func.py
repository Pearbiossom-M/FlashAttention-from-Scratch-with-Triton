import torch

def verify_results(bench, triton_output, name="Attention", rtol=1e-2, atol=1e-3):
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

    # 对齐 allclose 的“归一化误差”
    denom = atol + rtol * t.abs()
    norm = diff_abs / denom
    max_norm = norm.max().item() # 小于1就能通过
    
    # 3. 余弦相似度
    cosine_sim = torch.nn.functional.cosine_similarity(
        b.flatten(), t.flatten(), dim=0
    ).item()
    
    print(f"[{name} Verification]")
    print(f"Max Abs Error: {max_abs_err:.2e}")
    print(f"Mean Abs Error: {mean_abs_err:.2e}")
    print(f"Max Rel Error: {max_rel_err:.2e}")
    print(f"Max Normalized Error (allclose-style): {max_norm:.2e}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")
    
    # 4. 判定标准, 对于 fp16: 
    is_allclose = torch.allclose(b, t, rtol=rtol, atol=atol)
    
    if is_allclose and cosine_sim > 0.999:
        print("✅ Test Passed!")
    else:
        print("❌ Test Failed!")