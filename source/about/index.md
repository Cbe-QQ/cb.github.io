---
title: 关于
layout: about
eyebrow: ABOUT THIS BLOG
role: AI Infra 学习者
intro: 这里记录我从 Attention 到 NPU 算子的学习过程：读论文、拆算子、跑实验，然后把理解写成可以复用的笔记。
interests:
  - AI Infra
  - Attention
  - KV Cache
  - MoE
  - CANN / Ascend
links:
  - label: GitHub
    url: https://github.com/Cbe-QQ
  - label: RSS
    url: /rss.xml
---

## 这个博客是什么

这是一个围绕 AI Infra 的学习笔记库。内容不追求面面俱到，而是把每个主题拆到可以复述、可以动手验证的程度：先理解问题，再推导方法，最后落到算子与硬件。

## 写作原则

- **从第一性原理出发**：先问「为什么需要它」，再看「它是怎么做的」。
- **公式与代码并重**：推导之外，尽量给出可运行的实现或伪代码。
- **保留推导过程**：不只贴结论，把中间步骤和踩过的坑一起写下来。
- **持续修订**：笔记会随着理解加深而更新，过时的结论会被改掉。

## 内容主线

目前有两条主线：**DeepSeek 系列**从 FlashAttention、MLA 到 MoE，串起注意力算子、KV Cache 压缩和稀疏架构；**Ascend 系列**则从 CANN-Bench 的精度验证出发，进入昇腾算子开发与验证。

如果这些笔记对你有帮助，欢迎通过 RSS 订阅更新。
