---
title: Attention 數學結構
date: 2024-10-10 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2023-02-18-Attn_All_U_Need_Visual]]"
categories:
  - Science
tags:
  - Attention
  - LLM
  - Math
  - Science
  - Transformer
---

[[2023-03-26-Transformer_LLM]]
[[2024-10-11-Linear_Attention]]


## Transformer Series

<img src="/media/image-20241010115242.png" alt="20241010115242" style="zoom:60%;" />


## Transformer Layer Block Diagram (1-head)

<img src="/media/image-20241010115103.png" alt="20241010115103" style="zoom:60%;" />
### Math

<img src="/media/image-20241010115214.png" alt="20241010115214" style="zoom:60%;" />


## Multi-Head Attention

<img src="/media/image-20241010115415.png" alt="20241010115415" style="zoom:60%;" />

<img src="/media/image-20241010131619.png" alt="20241010131619" style="zoom:60%;" />
### Multi-Head Attention  數學表示：

$$
\begin{aligned}
x_0 & =U W_e+W_p \\
x_l & =\text{transformer\_block}(h_{l-1}) \quad \forall \, l \in [1, n_{layers}] 
\end{aligned}
$$

where $U=\left(u_{1}, \ldots, u_{k}\right)$  ($k=1024$) is the context vector of tokens, $n$ is the number of layers, $W_e$ is the token embedding matrix, and $W_p$ is the position embedding matrix.

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) & =\text{Concat}\left(\text{head}_1, \ldots, \text{head}_{n_{heads}}\right) W_o \\
\text { where \quad head}_i = \text{softmax}(Q_i K_i^T +M)V_i & =\text {Attention}\left(x W_i^Q, x W_i^K, x W_i^V\right)
\end{aligned}
$$

$M$ 是 attention mask, 就是要計算 attention 的部分，會收到兩個 factor 影響
-  **Token length < context length**:  有 token 部分才需要計算，padding 不用，特別是在 pre-fill stage.
-  **Causal mask:**  diagonal and lower triangle 才需要計算。主要是在 pre-fill stage 所有 tokens (token length x embed size).  **等到 generation stage, 每個進來的 token (1x embed size), 都需要和自己，以及之前的 KV cache 算 attention.  M 可以 ignore.**   

假設 $K, Q, V$ 都有一樣多 heads,  $d_{model} = d_{embed} = 768 = head_{num} \times d_{head}  \to  768 = 12 \times 64$
$$
\begin{aligned}
Q =&\left[Q_1, Q_2, ... Q_{12}\right] =& x \left[W^Q_1, W^Q_2, ...W^Q_{12}\right] =& x W^Q  \\
K =&\left[K_1, K_2, ... K_{12}\right] =& x \left[W^K_1, W^K_2, ...W^K_{12}\right] =& x W^K \\
V =&\left[V_1, V_2, ... V_{12}\right] =& x \left[W^V_1, W^V_2, ...W^V_{12}\right] =& x W^V \\
\end{aligned}
$$

Where the projections are parameter matrices $W_i^{Q,K,V} \in \mathbb{R}^{d_{\text {model }} \times d_{head}} = \mathbb{R}^{768 \times 64}$ and $W^{Q,K,V,O} \in \mathbb{R}^{d_{model} \times d_{model}}= \mathbb{R}^{768\times 768}$.

但是 $x, Q, K, V \in \mathbb{R}^{k \times d_{model}} = \mathbb{R}^{1024 \times 768}$.

#### 接下來的 FF Network:
$$
\operatorname{FFN}(x)=\max \left(0, x W_1+b_1\right) W_2+b_2
$$


## 變形 1:   MQA, GQA to save Weight+KV Cache Size and BW

<img src="/media/image-20241010131748.png" alt="20241010131748" style="zoom:60%;" />

<img src="/media/image-20241010132038.png" alt="20241010132038" style="zoom:60%;" />



## 變形 2:   KV Cache for Interence to Save Memory BW


<img src="/media/image-20241010132145.png" alt="20241010132145" style="zoom:60%;" />









#### Embedding - 參數 $W_{e}$ : $n_{vocab} \times d_{model}$，$W_{p}$ : $n_{ctx} \times d_{model}$

* $U$ 的大小是 ($n_{ctx} \times n_{vocab}$)，$W_{e}$ 的大小是 ($n_{vocab} \times d_{model}$)，$W_{p}$ 的大小是 ($n_{ctx} \times d_{model}$)。

* Vocabulary size ($n_{vocab}$) 是 50257。乍看很大，但是 one-hot，也就是 $U$ 的 element 只有 0 或 1.  $W_e$ 基本就是一本字典，每一個 vocabulary in 50257 都對應一個 embedding of vector length $d_{model}$.

* 所有的 models 都用同一個 context window, 其大小 $n_{ctx} = 2048$ tokens.  這個 context window size 決定這個 model 的記憶範圍。

* $h_0$ 的大小是 $(n_{ctx} \times d_{model})$。


#### Attention Block - 參數 $W_i^{Q,K,V}$ : $3 d_{model} d_{head} n_{head} = 3 (d_{model})^2$，$W^{O} = (d_{model})^2$ , total = $4  (d_{model})^2$, need to add 4  d bias? yes  =>  $4 (d_{model})^2 + 4 d_{model}$

* $Q, K$ 都是 $h_0$, 大小都是 $(n_{ctx} \times d_{model})$
* $V$ 是 output shifted right,  因爲 FFN 保持 input size 到 output size,  所以大小也是  $(n_{ctx} \times d_{model})$
* 對於 MultiHead attention, 每一個 head 的長度是 $d_{head}$，而且 $d_{model} = d_{head} \times n_{heads}$
* 每一個 head 都是三個矩陣乘法，$Q W_i^Q, K W_i^K, V W_i^V$，每個矩陣乘法大小是 $(n_{ctx} \times d_{model})\times (d_{model}\times d_{head})$，所以 head output 大小是 $n_{ctx}\times d_{head}$。但是因爲有 $n_{head}$ 而且 concat 在一起再做一次矩陣乘法 with $W^O$，所以 $\text{MultiHead}(Q,K,V)$ 的大小是 ($n_{ctx}\times d_{model}$).

#### Feed-Forward Network (FFN) - 參數 $W_{1,2}: 2\times d_{model} \times 4 d_{model} = 8 (d_{model})^2$ and $b_{1,2} = 5 d_{model}$ => $8 (d_{model})^2 + 5 d_{model}$

* Feed-forward network (FFN) 的 input 和 output 都一樣大小 $d_{model}$, 而且只有一層 hidden layer, $d_{ff} = 4 d_{model}$.   這一層 hidden layer 和 input 以及和 output 都是 fully connected network. 所以兩個的參數量都是 $d_{model} \times 4 d_{model}$  再加上兩個 bias  $4 d_{model} + 1 d_{model} = 5 d_{model}$.
* FFN 的最後大小和 input 一樣：  ($n_{ctx}\times d_{model}$).



#### Layer Norm - 參數  $\gamma, \beta$  :   $4d_{model}$

elf-attention块和MLP块各有一个layer normalization，包含了2个可训练模型参数：缩放参数 $\gamma$ 和平移参数 $\beta$，形状都是 [ℎ] 。2个layer normalization的参数量为 4ℎ 。



### GPT/Llama 總參數量：

一層的 transformer block:

$W_i^{Q,K,V},W_{1,2}, b_{1,2} = 4(d_{model})^2+4 d_{model} + 8(d_{model})^2 + 5 d_{model} + 4d_{model}  = 12(d_{model})^2 + 13 d_{model}$

$n_{layers}$ **多層以及加上 $W_e, W_p$ 總參數量：**

$n_{vocab}d_{model}+n_{ctx}d_{model}+n_{layers} \times (12 (d_{model})^2+ 13 d_{model})$



### Variable 縮寫

<img src="/media/image-20230721114220162.png" alt="image-20230721114220162" style="zoom: 80%;" />

<img src="/media/image-20230721114303954.png" alt="image-20230721114303954" style="zoom:67%;" />

<img src="/media/image-20230723205534577.png" alt="image-20230723205534577" style="zoom: 80%;" />

所以總共的參數有：$vy+uy+4xyzw+8xy^2 + 13xy = y(v + u) + x (4yzw+8y^2+13y)$.

一般 $y = zw$ 所以也可以寫成 $y(v+u) + x (12y^2+13y)$.  

這也是上式： $P= d_{model} \cdot (n_{vocab}+n_{ctx})+n_{layers} \cdot (12 (d_{model})^2+ 13 d_{model})$

* **注意參數量 (不含 $W_p$) 和 token length 也和 batch 無關。這和 activation 不同！！！** 
* 如果使用相對位置編碼，例如 RoPE (Llama) or ALiBi, 不包含可訓練的參數，$W_p$ 可以忽略。
* 另一種寫法是 $P = Vh + l (12 h^2 + 13 h)$ 

<img src="/media/image-20230723202649745.png" alt="image-20230723202649745" style="zoom: 67%;" />



<img src="/media/image-20240115212837183.png" alt="image-20240115212837183" style="zoom:80%;" />


##### Self attention

b = batch  在 training 時，會有 batch input,  在 inference 是 batch = 1 for ChatGPT model.

s = n_ctx (input tokens)

h = d_model ( = num_head * d_head)

* 此處考慮 batch size, 因爲 training.   
* ChatGPT inference 的 batch size = 1.    以及在某些 network (大小網絡) batch size > 1 可以加速。

**K, Q, V: Mapping Matrix** :  $Q = x_{in} W_Q，K = x_{in} W_K， V = x_{in} W_V$

* input 和 output shape  [b, s, h] x [h, h] -->  [b, s, h]  
* Input activation 的量是:  $p b s h$,  p 是 precision, 如果是 FP16, p 是 2 個 byte.

**QK: Attention matrix**   $x_{out} = \text{softmax}\left( \frac{Q K^T}{\sqrt{h}}\right) \cdot V \cdot W_O + x_{in}$

* 此處要考慮 multi-heads,  因此把 3D 的 Q, K, V [b, s, h] reshape 成 4D [b, head_num, s, per_head_hidden_size]  where h = d_model = head_num * per_head_hidden_size 
* $Q K^T$  矩陣的 input 和 output [b, head_num, s, per_head_hidden_size] x [b, head_num, per_head_hidden_size, s] --> [b, head_num, s, s]
* Input activation 的量是:  $2 p b s h$,  

**Softmax**   $x_{out} = \text{softmax}\left( \frac{Q K^T}{\sqrt{h}}\right) \cdot V \cdot W_O + x_{in}$

* input 和 output shape 都是: [b, head_num, s, s]
* Input activation 的量是:  $p b s^2 a$,  
* 计算完 softmax 函数后，会进行dropout操作。需要保存一个mask矩阵，mask矩阵的形状与 softmax 相同，占用显存大小为   $ b s^2 a$。Make 只需要 1 byte, 不用乘 p.

**Score**   $x_{out} = \text{softmax}\left( \frac{Q K^T}{\sqrt{h}}\right) \cdot V \cdot W_O + x_{in}$

* input 和 output shape 都是: [b, head_num, s, s] x [b, head_num, s, per_head_hidden_size] --> [b, head_num, s, per_head_hidden_size]
* Input activation 的量是:  $p b s^2 a + p b s h$,  

**Output Mapping**   $x_{out} = \text{softmax}\left( \frac{Q K^T}{\sqrt{h}}\right) \cdot V \cdot W_O + x_{in}$

* input 和 output shape 都是: [b, s, h] x [h, h] --> [b, s, h]
* Input activation 的量是:  $p b s h$,  再加上一個 dropout $b sh$, total $(p+1) b s h$



##### Self-Attention 的 activation

$p b s h + 2 p b s h + p b s^2 a + p b s^2 a + b s^2 a + p b sh + (p+1)bsh = (5p+1) b s h + (2p+1) b s^2 a$



##### MLP

$$ x_{mlp} = f_{gelu} (x_{out} W_1) W_2 + x_{out}$$

* 第一個 FC (W1)，
* Input activation 的量是:  $p b s h$
* GELU 需要保存輸入：$4p b s h$ 
* 第二個 FC (W2)，矩陣乘法的輸入和輸出 [b, s, 4h] x [4h, h] --> [b, s, h]
* Input activation 的量是:  $4p b s h$
* 最後有一個 dropout, 需要保存 mask 矩陣, 大小是 $bsh$

##### MLP 的 activation

$(9p+1) b s h $

另外，self-attention块和MLP块分别对应了一个layer normalization。每个layer norm需要保存其输入，大小为 $pbsh$ 。2个layer norm需要保存的中间激活为 $2pbsh$.



综上，**每个transformer层需要保存的中间激活占用显存大小为** $(16p+2) bsh + (2p+1) b s^2 a$ 。对于 $l$ 层transformer模型，还有embedding层、最后的输出层。embedding层不需要中间激活。总的而言，当隐藏维度 ℎ 比较大，层数 $l$ 较深时，这部分的中间激活是很少的，可以忽略。因此，**对于 $l$ 层transformer模型，中间激活占用的显存大小可以近似为** $((16p+2)bsh + (2p+1) b s^2 a)*l$ 。





在推断阶段，transformer模型加速推断的一个常用策略就是使用 KV cache。一个典型的大模型生成式推断包含了两个阶段：

1. **预填充阶段**：输入一个prompt序列，为每个transformer层生成 key cache和value cache（KV cache）。
2. **解码阶段**：使用并更新KV cache，一个接一个地生成词，当前生成的词依赖于之前已经生成的词。



第 $i$个transformer层的权重矩阵为 $W_Q^i, W_K^i, W_V^i, W_O^i, W_1^i, W_2^i$。

其中 self-attention 的 4 個權重矩陣  $W_Q^i, W_K^i, W_V^i, W_O^i \in R^{h \times h}$。

并且MLP块的2个权重矩阵 $W_1^i \in R^{h \times 4h}, W_2^i \in R^{4h \times h}$。



**预填充阶段**

假设第 $i$个transformer层的输入为 $x^i$ ，self-attention块的key、value、query和output表示为 $x_K^i, x_V^i, x_Q^i, x_{out}^i$ 其中 $x_K^i, x_V^i, x_Q^i, x_{out}^i \in R^{b\times s\times h}$。

Key cache 和 value cache 的計算過程為

$x_K^i = x^i \cdot W_K^i$

$x_V^i = x^i \cdot W_V^i$

第 $i$ 個 transformer 層剩餘的計算過程為

<img src="/media/image-20231022220017738.png" alt="image-20231022220017738" style="zoom: 67%;" />

**解码阶段**

给定当前生成词在第 $i$ 个transformer层的向量表示为 $t^i \in R^{b \times 1 \times h}$.  推理計算分兩部分：更新 KV cache 和計算第 $i$ 個 transformer 層的輸出。

更新 key cache 和 value cache 的計算過程如下：

<img src="/media/image-20231022220801197.png" alt="image-20231022220801197" style="zoom: 67%;" />



### Attention is what you need, Memory is the Bottleneck

Attention 已經是必備的 core network.   相較於 CNN,  attention 最大的問題是 memory bandwidth.

主要在計算 K, Q 的 correlation, 以及 softmax.  以下是 GPT1/2/3 的參數。

下圖應該畫錯了！ GPT 應該是 decoder only (右邊)。所以對應的方塊圖是沒有 encoder (左邊)，只有 decoder (右邊)。所以打叉的地方相反。BERT 才是 encoder only (左邊)。不過兩者的架構非常類似。不過 decoder only 架構 output 會 shift right 再接回 input, 稱爲 auto-regression.

<img src="/media/image-20230723204336707.png" alt="image-20230723204336707" style="zoom:80%;" />


## Source

* [Efficient Memory Management for Large Language Model Serving with PagedAttention (arxiv.org)](https://arxiv.org/pdf/2309.06180.pdf)
* Flash Attention with attention bias:  https://zhuanlan.zhihu.com/p/567167376
* Flash attention 2: https://tridao.me/publications/flash2/flash2.pdf
* Flash Decoder: https://princeton-nlp.github.io/flash-decoding/
* 詳細的 GPT2/3 參數計算: https://www.lesswrong.com/posts/3duR8CrvcHywrnhLo/how-does-gpt-3-spend-its-175b-parameters
* LLM1,  https://finbarr.ca/how-is-llama-cpp-possible/
* [FlashAttention图解（如何加速Attention） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/626079753)
* [NLP（十七）：从 FlashAttention 到 PagedAttention, 如何进一步优化 Attention 性能 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/638468472)
* [大模型推理性能优化之KV Cache解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/630832593)
* The KV Cache: Memory Usage in Transformers  https://www.youtube.com/watch?v=80bIUggRJf4&ab_channel=EfficientNLP
