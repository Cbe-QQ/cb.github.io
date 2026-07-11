---
title: Flash Attention
date: 2026-07-11
categories: DeepSeek
tags:
  - Flash Attention
  - Attention
  - Deep Learning
mathjax: true
---

### 基础Attention

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

  1. 相似度计算：将 Query 矩阵 $Q$ 与 Key 矩阵 $K$ 做点积（$QK^T$），得到注意力得分矩阵。每个元素表示某个 query 与某个 key 之间的相关性。

  2. 缩放（Scale）：将得分矩阵除以 $\sqrt{d_k}$。因为 $d_k$ 较大时点积值会很大，导致 softmax 梯度趋于 0（饱和区），缩放可以保持梯度稳定。

  3. Softmax 归一化：对每行做 softmax，将得分转换为概率分布（每行之和为 1）。此时每个值代表对应位置 Value 的注意力权重。

    4. 加权求和：将注意力权重矩阵与 Value 矩阵 $V$ 相乘。每个输出位置是所有 Value 向量的加权和，权重由 Query-Key 的相关性决定。

 直观理解：对于序列中的每个位置，Attention 机制动态地决定应该"关注"序列中哪些其他位置，并聚合它们的 Value 信息。越相关的位置，分配的权重越大。



### softmax流程

基础计算公式：

- softmax :  用于计算每个位置的概率
  - 公式：$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$
- safe softmax: 防止某个概率出现溢出的场景，因此需要减去最大值
  - 公式： $\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$

- online softmax：若采用上述的公式，需要较大的缓存来进行计算，因此采用分块的方式来计算softmax

  - 公式：
    $$
    m_{new} = Max(m_{old}, m_{block}) \\
    l_{new} = \sum_{i=0}^{2n}e^{i - m_{new}}\\
    =\sum_{i=0}^{n}e^{i - m_{old} + m_{old} - m_{new}} + \sum_{i=n+1}^{2n}e^{i-m_{block} + m_{block} -m_{new}}\\
    =e^{m_{old} - m_{new}}*l_{old} + e^{m_{block}-m_{new}}*l_{block}
    $$


### Flash Attention - V1

重点循环：KV外循环，Q内循环

公式如下：

![img](/images/FA_2.jpg)

从以下的公式推断：

- Q第一个循环：
  - 计算以下的内容：
    - $Q_1K_1$矩阵相乘：$S_1=Q_1K_1$
    - 当前块最大值： $m_1 = max(S_1)$
    - 减去最大值的矩阵：$P_1=e^{S_1 - m_1}$
    - 减去最大值的总和： $l_1=sum(e^{s_1 - m_1})$
    - 计算得到的Output： $O_1=\frac{e^{s_1 - m_1}}{l_1}V_1$

- Q第二个循环

  - 计算以下的内容：

    - $Q_2K_2$矩阵乘：$S_2=Q_1K_2$
    - 当前块最大值： $m_2=max(S2)$
    - 减去最大值的矩阵：$P_2=e^{S_2 - m_2}$
    - 减去当前块最大值总和：$l_2=sum(e^{S_2 - m_2})$

  - 额外计算得到两个分块的最大值与和：

    - 两块矩阵的最大值：$m_{new} = max(m_1, m_2)$
    - 两块矩阵总和：$l_{new}=e^{m_1-m_{new}}*l_1+e^{m_2 - m_{new}}*l_2$

  - 计算得到最后的Output：
    $$
    Ouput = \frac{e^{S-m_{new}}}{l}V \\
    =\frac{e^{S_1 - m_{new}}}{l}V_1 + \frac{e^{S_2} - m_{new}}{l}V_2 \\
    =\frac{e^{S_1-m_1}e^{m_1-m_{new}}l_1 / l_1V_1}{l} + \frac{e^{S_2 - m_{2}}e^{m_2 - m_{new}}V_2}{l} \\
    =\frac{O_1e^{m_1 - m_{new}}l_1 + e^{S_2 - m_{new}}P_2V_2}{l}
    $$


**缺点**：需要使用数组存放内循环产生的O/l/m这三个参数用于外循环遍历计算

**代码**：

```python
import torch

torch.manual_seed(456)

N, d = 16, 8

Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

# 执行标准的pytorch softmax和attention计算
expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
expected_attention = expected_softmax @ V_mat


# 分块（tiling）尺寸，以SRAM的大小计算得到
Br = 4
Bc = d

# flash attention算法流程的第2步，首先在HBM中创建用于存储输出结果的O，全部初始化为0
O = torch.zeros((N, d))
# flash attention算法流程的第2步，用来存储softmax的分母值，在HBM中创建
l = torch.zeros((N, 1))
# flash attention算法流程的第2步，用来存储每个block的最大值，在HBM中创建
m = torch.full((N, 1), -torch.inf)

# 算法流程的第5步，执行外循环
for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc
    # line 6, load a block from matmul input tensor
    # 算法流程第6步，从HBM中load Kj, Vj的一个block到SRAM
    Kj = K_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    Vj = V_mat[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    # 算法流程第7步，执行内循环
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
        # 算法流程第8行，从HBM中分别load以下几项到SRAM中
        mi = m[block_start_Br:block_end_Br, :]  # shape Br x 1
        li = l[block_start_Br:block_end_Br, :]  # shape Br x 1
        Oi = O[block_start_Br:block_end_Br, :]  # shape Br x d
        Qi = Q_mat[block_start_Br:block_end_Br, :]  # shape Br x d

        # 算法流程第9行
        Sij = Qi @ Kj.T  # shape Br x Bc

        # 算法流程第10行，计算当前block每行的最大值
        mij_hat = torch.max(Sij, dim=1).values[:, None]

        # 算法流程第10行，计算softmax的分母
        pij_hat = torch.exp(Sij - mij_hat)
        lij_hat = torch.sum(pij_hat, dim=1)[:, None]

        # 算法流程第11行，找到当前block的每行最大值以及之前的最大值
        mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]

        # 算法流程第11行，计算softmax的分母，但是带了online计算的校正，此公式与前面说的online safe softmax不一致，但是是同样的数学表达式，只是从针对标量的逐个计算扩展到了针对逐个向量的计算
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        # 算法流程第12行，计算每个block的输出值
        Oi = (li * torch.exp(mi - mi_new) * Oi / li_new) + (torch.exp(mij_hat - mi_new) * pij_hat / li_new) @ Vj

        # 算法流程第13行
        m[block_start_Br:block_end_Br, :] = mi_new  # row max
        l[block_start_Br:block_end_Br, :] = li_new  # softmax denominator
        # 算法流程第12行，将Oi再写回到HBM
        O[block_start_Br:block_end_Br, :] = Oi
```



### Flash Attention - V2

为了解决上述存在的缺点，不保存多余变量，因此提出了V2版本

重点循环：Q外循环，KV内循环

![img](/images/FA_1.jpg)

从以下公式推断：

- Q第一个循环：

  - 计算以下的内容： 
    - $Q_1k_1$矩阵乘：$S_1=Q_1K_1$
    - 当前块最大值： $m_1 = max(S_1)$
    - 减去最大值的矩阵：$P_1=e^{S_1 - m_1}$
    - 减去最大值的总和： $l_1=sum(e^{s_1 - m_1})$
    - 当前计算得到的最后输出(不除总和): $O_1=e^{s_1 - m_1}V_1$

- Q第二个循环

  - 计算以下的内容：

    - $Q_2K_2$矩阵乘: $S_2=Q_1K_2$
    - 当前块最大值： $m_2=max(S2)$
    - 两块相比最大值： $m_{new} = max(m_1, m_2)$
    - 减去最大值的矩阵：$P_2=e^{S_2 - m_{new}}$
    - 减去最大值的总和：  $l_2=sum(e^{S_2 - m_{new}})$

  - 计算最后的Output:
    $$
    Output = e^{S-m_{new}}V \\
    =e^{S_1 - m_1}e^{m_1 - m_{new}}V_1 + e^{S_2 - m_{new}}V_2 \\
    =O_1e^{m_1 - m_{new}}+ P_2V_2
    $$

  - 不断更新l的值，在最后一个ouput时除以l，就是最后的结果

**代码**：

```python
import torch

torch.manual_seed(456)

N, d = 16, 8
Q_mat = torch.rand((N, d))
K_mat = torch.rand((N, d))
V_mat = torch.rand((N, d))

expected_softmax = torch.softmax(Q_mat @ K_mat.T, dim=1)
expected_attention = expected_softmax @ V_mat

# 分块（tiling）尺寸，以SRAM的大小计算得到
Br = 4
Bc = d

O = torch.zeros((N, d))

# 算法流程第3步，执行外循环
for block_start_Br in range(0, N, Br):
    block_end_Br = block_start_Br + Br
    # 算法流程第4步，从HBM中load Qi 的一个block到SRAM
    Qi = Q_mat[block_start_Br:block_end_Br, :]
    # 算法流程第5步，初始化每个block的值
    Oi = torch.zeros((Br, d))  # shape Br x d
    li = torch.zeros((Br, 1))  # shape Br x 1
    mi = torch.full((Br, 1), -torch.inf)  # shape Br x 1

    # 算法流程第6步，执行内循环
    for block_start_Bc in range(0, N, Bc):
        block_end_Bc = block_start_Bc + Bc

        # 算法流程第7步，load Kj, Vj到SRAM
        Kj = K_mat[block_start_Bc:block_end_Bc, :]
        Vj = V_mat[block_start_Bc:block_end_Bc, :]

        # 算法流程第8步
        Sij = Qi @ Kj.T
        # 算法流程第9步
        mi_new = torch.max(torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1).values[:, None]
        Pij_hat = torch.exp(Sij - mi_new)
        li = torch.exp(mi - mi_new) * li + torch.sum(Pij_hat, dim=1)[:, None]
        # 算法流程第10步
        Oi = Oi * torch.exp(mi - mi_new) + Pij_hat @ Vj
$
        mi = mi_new

    # 第12步
    Oi = Oi / li

    # 第14步
    O[block_start_Br:block_end_Br, :] = Oi
```



#### 参考资料

> https://www.cnblogs.com/rossiXYZ/p/18798185
