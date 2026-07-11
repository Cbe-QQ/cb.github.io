---
title: DeepSeek MoE
date: 2026-07-11
categories: DeepSeek
tags:
  - MoE
  - DeepSeek
  - Mixture of Experts
  - Deep Learning
mathjax: true
---

#### 概念：

**MoE(Mixtrue of Experts)**: 混合专家模型，使用更少的计算资源来完成有效的预训练。

**核心思想**：通过Router或者Gate Network决定每个token应该由哪个专家处理

> 在自注意力机制中进一步使用注意力机制



#### 整体架构

![img](/images/MoE.png)

- **门控网络(Gate Network)**：决定不同token应该由哪个专家处理
- **专家(Experts)**：每个专家实际是一个FFN

计算公式如下：

1. 计算得到输入参数的0~1分布阈值
2. 计算得到专家的输出值，本质是FFN
3. 加权求和得到最后的输出结果

$$
Gate(x) = softmax(xW) \\
Experts(x)=(xW_1)W_2 \\
Output(x)=\sum_{i=1}^nGate(x)_i Experts_i(x)
$$

#### 存在的问题

- 负载不均衡：会造成token分配不均衡，部分专家被分配的token过少导致训练、利用得不够充分；部分专家被分配的token过多，但由于内存限制只能选择一定数量的token使用，导致token资源的浪费。
- 冗余专业化：每个MoE层都使用一个门控网络来学习令牌与专家的亲和力。理想情况下，学习的门控网络应该产生亲和力，以便将相似或相关的令牌路由到同一个专家。然而，如果门控网络是次优策略，则可能会产生冗余的专家和/或不够专业的专家。

#### DeepSeek中MoE

##### DS V1

![img](/images/DS-V1.jpg)

由于当前MoE架构中存在知识混杂和知识冗余，为了解决该问题，引入了：

- 细粒度/垂类专家：通过减少FFN中间隐藏层维度来增加专家数量
- 共享专家：将某些专家隔离出来，作为始终激活的共享专家，旨在捕获不同上下文中的共同知识。通过将共同知识压缩到这些共享专家中，可以减轻其他路由专家之间的冗余，这可以提高参数效率，确保每个路由专家专注于不同方面而保持专业化。

**计算公式如下**:
$$
h_t^l = \sum_{i=1}^{K_s}FFN_i{x_t^l} + \sum_{K_s+1}^{m^N}g_{i,t}FFN_i{x_t^l} + x_t^l \\
g_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} \in \mathrm{Topk}\left(\{s_{j,t}\mid K_s + 1 \le j \le mN\},\, mK - K_s\right) \\
0, & \text{otherwise}
\end{cases} \\

s_{i,t} = \mathrm{Softmax}_i\left(x_t^l e_i\right)
$$

> $K_s$表示的共享专家的数量

##### DS V2

DeepSeek-V2进一步扩大了细粒度专家选择，采用了路由专家160选6，加上2个共享专家的做法，同时新增了一个路由机制和两个负载均衡方法。

- MoE Layer包含162专家数，其中2个共享专家，160个路由专家，每个token激活8个专家



##### DS V3

相比于前两版框架，V3主要增加了路由专家数，修改了门控网络的激活函数

- MoE Layer包含256个专家，每个token激活专家数8个

由于增加了专家数，softmax激活函数的区分度降低，计算误差加大，会导致专家选择误差增大，因此采用了sigmoid函数

**计算公式如下**
$$
h'_t = u_t + \sum_{i=1}^{N_s} \mathrm{FFN}_i^{(s)}(u_t)
+ \sum_{i=1}^{N_r} g_{i,t}\,\mathrm{FFN}_i^{(r)}(u_t), \\
g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}, \\
g'_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} \in \mathrm{Topk}\left(\{s_{j,t}\mid 1 \le j \le N_r\},\, K_r\right) \\
0, & \text{otherwise}
\end{cases} \\
s_{i,t} = \mathrm{Sigmoid}\left(u_t^T e_i\right).
$$
