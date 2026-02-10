# FlashAttention from Scratch (with Triton)

> 从算法动机出发，完整实现并优化 FlashAttention —— 直到你真的**理解它为什么快**

---

## 📌 项目简介

本仓库是一套**循序渐进、从零开始**的 FlashAttention 教程与实现，基于 **Triton** 编写，目标不是“复刻官方代码”，而是：

> **让你真正理解 FlashAttention 的算法结构、反向传播、以及性能优化背后的工程逻辑。**

你将看到 FlashAttention 如何一步步从朴素 Attention 演化而来，为什么 online softmax 是核心，为什么 backward 天生更难，以及在 Triton 能力边界内，哪些优化是**真的有效**，哪些只是“看起来很美”。

---

## 🎯 适合谁阅读？

如果你满足以下任意一条，这个仓库很适合你：

- 想**彻底搞懂 FlashAttention**，而不是只会调用 API
- 有 CUDA / GPU 编程基础，但 **Triton 经验不多**
- 能读 PyTorch / Triton 代码，但看官方 FlashAttention 实现时常常：
  > “我知道它在干嘛，但不知道**为什么要这么写**”
- 想知道：
  - FlashAttention 的 backward 为什么这么复杂？
  - 官方实现里哪些优化 **Triton 很难复现**？
  - 自己写的 kernel 到底卡在 memory 还是 compute？

---

## 🚫 不适合谁？

- 只想要一个 **drop-in 替代品**（请直接用 PyTorch SDPA / flash-attn）
- 完全不关心算法与性能细节
- 对长文档和推导极度不耐烦 😄

---

## 🧠 教程结构（Phase-by-Phase）

本教程按 **Phase** 组织，每一阶段只引入**必要的新复杂度**：

### Phase 0 — Attention 的计算模型与瓶颈
- Attention 的计算结构
- 为什么 naive attention 会 OOM / memory-bound
- FlashAttention 要解决的到底是什么问题？

### Phase 1 — 从朴素实现到分块 Attention
- QK^T / softmax / PV 的分块化
- 为什么**只分 Q 或只分 K 都不够**
- 为 online softmax 做铺垫

### Phase 2 — Online Softmax：FlashAttention 的灵魂
- 不存 P，如何正确计算 softmax？
- 数值稳定性的关键推导
- Flash K/V 的第一版实现

### Phase 3 — 完整 Forward：真正的 FlashAttention
- 流式扫描 K/V
- causal mask 的正确引入方式
- LogSumExp（LSE）的设计动机

### Phase 4 — Backward：Recomputation 的智慧
- dQ / dK / dV 的完整数学推导
- 为什么 backward 必须 recompute P？
- 两个 backward kernel 的分工逻辑
- 与 PyTorch SDPA 的数值对齐验证

### Phase 5 — 性能调优：从“能跑”到“飞起来”
- 性能基线与 TFLOPS 评估
- autotune 的正确打开方式
- TensorDescriptor / TMA 的适用边界
- 计算流程重排（如 delta 复用）
- 哪些优化**看起来合理但实际上没用**

### Phase 6 — 边界、拓展与收官
- Dropout：为什么它是工程问题而非算法问题
- 变长序列（cu_seqlens）：kernel 不难，接口难
- 为什么不逐行复刻官方实现
- Triton 的能力边界与下一步方向

---

## 🧪 正确性与性能

- **正确性**
  - Forward / Backward 均与 PyTorch `scaled_dot_product_attention` 对齐
  - 使用工业界常用的数值对齐方式（而非脆弱的 `gradcheck`）

- **性能**
  - 在 Forward 场景下，性能接近（部分配置可追平）PyTorch SDPA
  - Backward / fwd+bwd 场景中稳定达到其 **70%～80%**
  - 所有 benchmark 均基于 GPU event 计时，排除冷启动噪声

> ⚠️ 本项目目标不是击败官方实现，而是在 **Triton 可表达范围内** 达到接近上限的性能，同时保持可读性。

---

## 🧩 为什么选择 Triton？

- Triton 非常适合表达：
  - block-level 并行
  - online softmax
  - 算法级优化（memory ↔ compute tradeoff）
- 但 Triton **并不能完全复现**：
  - persistent thread blocks
  - 深度 warp specialization
  - 一些调度级黑魔法

本教程会**明确指出这些边界**，而不是假装它们不存在。

---

## 🛠️ 环境要求（建议）

- NVIDIA GPU（Ampere 及以上体验更佳）
- PyTorch ≥ 2.1
- Triton ≥ 3.x
- CUDA ≥ 12.x

> 在支持更先进架构（如 Hopper / Blackwell）的设备上，可进一步探索更激进的优化配置。

---

## 📚 如何阅读？

强烈建议 **按 Phase 顺序阅读**，而不是跳着看代码。

这是一个**教学型仓库**，不是代码片段合集。

如果你已经熟悉 FlashAttention，可以：
- 快速浏览 Phase 0–2
- 重点阅读 Phase 4（Backward）和 Phase 5（性能调优）

---

## 🧭 下一步你可以做什么？

- 给当前实现加上 **dropout**
- 支持 **variable-length sequence（cu_seqlens）**
- 尝试在 CUDA / PTX 层实现更激进的调度优化
- 对照官方实现，分析哪些优化超出了 Triton 的表达能力

---

## 📄 License

MIT License 
你可以自由使用、修改和引用本项目中的代码与内容。

---

## 🙌 致谢

- FlashAttention 原论文作者
- Triton / PyTorch 社区
- 所有愿意认真理解底层机制的人

如果这套教程对你有帮助，欢迎 ⭐️ Star / Fork / Discussion！
