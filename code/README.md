# 代码说明

* `_flash_attention_kernel_optimized.py`

  存放全部 kernel，包括：

  * `flash_attention_forward_kernel`
  * `flash_attention_dQ_kernel`
  * `flash_attention_dKV_kernel`
  * 及其对应的 autotune 配置和 pre hook 函数设置

* `_verify_func.py`

  存放正确性校验函数：`verify_results`

* `My_FlashAttention_optimized.py`

  包含：

  * 对 kernel 的调用封装：`flash_attention_forward` 和 `flash_attention_backward`
  * `torch.autograd.Function` 集成
  * 生成外部调用 API：`flash_attention`
  * 以及对 `O`、`dQ`、`dK`、`dV` 的正确性校验

* `Performance_Comparison.py`

  性能测试函数，用于测量不同实现之间的性能差异