---
title: MLA (Multi-head Latent Attention)
date: 2026-07-11
categories: DeepSeek
tags:
  - MLA
  - Multi-head Latent Attention
  - DeepSeek
  - KV Cache
mathjax: true
---

在聊MLA作用之前，先说一下推理的流程，总共分为2个流程：

- **prefill阶段**：是模型对全部的Prompt tokens一次性并行计算，最终会生成第一个输出token
- **decode阶段**：每次生成一个token，直到生成EOS（end-of-sequence）token，产出最终的response

因此对于初始的MHA，需要存放所有的KV-Cache，这会造成内存使用率爆炸性增长，为了减少KV存储的数据量，有各种各样的方法：

- **共享KV**：MQA/GQA
- **量化压缩**：通过量化的方式，将KV值以低bit的方式存储
- **计算优化**：FA

### MLA流程

**论文流程图**:

![image-20260709213727426](/images/MLA-paper.png)

MLA整体流程可以分为部分，输入$x_t$

- $KV$计算流程：

  1. 输入低秩投影得到$c_t^{kv}$：$c_t^{kv}=W^{DKV}x_t$， 该向量是存储的KV向量，用于推理
     - $W^{DKV}\in \R^{d_c\times d}$
       - $d_c$: KV低秩投影维度，$d_c=4\times d_h$
       - $d_h$: 单个head的维度
       - $n_h$: 总head的维度
       - $d$: 隐藏层维度，$d=d_h\times n_h$
  2. 计算得到KV矩阵: 将KV矩阵维度扩展到原多头维度
     - $K=W_kc_t^{qk}$
     - $V=W_vc_t^{qk}$
     - $W_k, W_v\in \R^{d_hn_h\times d_c}$

- $Q$计算流程：

  1. 低秩投影：$c_t^{q}=Wx_t$
     - $W\in \R^{d_q\times d}$
     - $d_q$: Q低秩投影维度，是$d_c$ 3倍
  2. 升维：$Q=W_qc_t^q$
     - $W_q \in \R^{d_hn_h\times d_q}$

- RoPE位置编码得到最后QK：

  - $q_t^R=RoPE(W^{QR}c_t^q)$

  - $k_t^R=RoPE(W^{KR}x_t)$

    > 1. $q_t^R$和$k_t^R$都是较小的向量维度
    > 2. $k_t^R$是所有head共享一个k

  - 使用矩阵拼接得到最后的qk

    - $q=[Q;q_t^R]$
    - $k=[K;k_t^R]$



**计算流程图**：

![img](/images/MLA-flow.jpg)



### 参考资料：

> https://zhuanlan.zhihu.com/p/16730036197
>
> https://spaces.ac.cn/archives/10091
>
> https://zhuanlan.zhihu.com/p/1911795330434986569
