---
title: Llama3 70B Distributed Inference Code
date: 2024-10-19 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-10-18-Distributed_Inference]]"
categories:
  - GenAI
tags:
  - Attention
  - Distributed_Inference
  - LLM
  - Llama
  - Transformer
---

[[2023-03-26-Transformer_LLM]]
[[2024-10-11-Linear_Attention]]
[[2024-10-10-Attention_Math]]
[[2023-10-21-LLM_Memory]]

## Source

The following are some notes on how the Llama 3.1 70B model works in distributed environment. Will focus only on the pure PyTorch implementation which uses the _fairscale_ library. _Fairscale_ library is a fork of the Nvidia _Megatron-LM_ library. The Llama 3 and 3.1 models are the same, hence in the following will refer to them as Llama 3.


<img src="/media/image-20241020001941.png" alt="20241020001941" style="zoom:60%;" />


## 我們用 Llama 70B 3.1 爲例：

Embedding: 1 layer
Transformer:  80 layers
最後 linear layer: 1 layer

d_model: 8K (8192)
**context_window: 128K**  這是 Llama3.1 和 Llama3 最大的差別
output max output:  2K
Q head number: 64
KV head number: 8, i.e.  8 Q heads share 1 K,V head.



Allreduce 點:  就是 GPU 需要同步全部資訊的點。
- Embedding 結束一次。
- Attention 結束一次 x layer number (80)
- FFN 結束一次 x layer number (80)
- 最後 linear layer 再一次
一共 80x2 + 2 = 162 次 allreduce.

<img src="/media/image-20241027214924.png" alt="20241027214924" style="zoom:60%;" />

#### Pesudo Code

```json
Transformer(  
	(tok_embeddings): VocabParallelEmbedding()  
	(layers): ModuleList(  
		(0-79): 80 x TransformerBlock(  
			(attention): Attention(  
				(wq): ColumnParallelLinear()  
				(wk): ColumnParallelLinear()  
				(wv): ColumnParallelLinear()  
				(wo): RowParallelLinear()  
			)  
			(feed_forward): FeedForward(  
				(w1): ColumnParallelLinear()  
				(w2): RowParallelLinear()  
				(w3): ColumnParallelLinear()  
			)  
			(attention_norm): RMSNorm()  
			(ffn_norm): RMSNorm()  
		)  
	)  
	(norm): RMSNorm()  
	(output): ColumnParallelLinear()  
)
```


## 如何切 70B Model?

我們是利用 Tensor Parallelism on multiple (8) GPUs.
什麽 Tensor，如何分割?  我們從一層 transformer layer 分析。   
包含兩個連結的 blocks: Attention and MLP (or FFN: feedforward network)

<img src="/media/image-20241019083502.png" alt="20241019083502" style="zoom:60%;" />

### 先看比較複雜 Attention block,  假設 batch = 1

結論：
- Batch parallelism 是 no brainer, 不過有 system level 的 concerns 和無法沒有任何 memory BW sharing/reuse gain.  基本是搭配其他 parallelism.  一般在 Inferencing + generation mode 會盡量拉高 batch number reuse the memory BW sharing gain 以拉大 generation throughput at the cost of latency.  
- Token parallelism:  因爲 attention block 就是在算 token 之間的 correction 有強相關。Token parallelism 不是很好的切法。但是在 MLP 部分就非常有效。**這也是爲什麽 MOE 可以利用 token parallelism.**   
- Tensor parallelism 在 K, Q, V 有強相關。不 parallel.  但是在 multi-heads 則是理想的 partition.  只有在 input 和 output 需要做 synchronization.

| **Partitioning**                                        | **Training/Inferecing**                                                               | **Memory**                                               | Communication                                                                         | Others                                                |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Batch Parallelism** (e.g. 10 batch each GPU)          | - Good for training with fixed batch<br>- NG for inferencing (batch=1, dynamic batch) | - Each GPU has complete MLP weights, no memory BW saving | - Zero                                                                                | - combined with other methods                         |
| **Token Parallelism**                                   | - NG for generation<br>- Good for prompt mode                                         | -                                                        | -                                                                                     | - Not use for attention<br>- Use for embedding and FF |
| **Tensor Parallelism** (not Q,K,V, but multi-heads)<br> | - both training and inferencing<br>                                                   | - Reduces per-GPU memory usage for MLP weights           | - Allreduce only at input/output <br>- **Requires synchronization after every layer** | **- Most favorable**                                  |

Attension Layer 主要包含三個 tensors:  Q, K, V $\in \mathbb{R}^{l\times d} = \mathbb{R}^{128k \times 8k}$ of 1GB parameter.  
If using FP16, 每個 tensor 都是 2GB at maximum.
對於 80 層，Q: 2G x 80 (layers) = 160 GB.  如果 8 個 GPU, 各 20GB 内存。

但是 K, V 使用 GQA 只有 8 heads, vs. 64 heads, 所以 K, V 各自 160GB/8 = 20GB.  
Q+K+V= 160GB + 20GB + 20GB = 200GB
- Training: 需要 200 GB
- Inferencing:  
	- Pre-fill phase:  需要先產生 seq_len 的  Q, K, V.  所以是 200 GB x seq_len / 128K.  除非是非常長 input (例如一篇小説 128K),  此時需要 200 GB.  不然只要一個比例。
	- Generation phase:  KV cache = 40 GB (max)，Q 只需要前一個 1 token, 不需要 128K.
		- KV = 40 GB,  Q 一般是前一個 1 token, 不需要 之前的 128K 


#### 切 K, Q, V

**很直覺的切法是 K, Q, V 在不同的 GPUs.  不過這不是好方法，因爲 Q, K, V, 有很強的相關性，如下圖左。如果是在不同的 GPU 或是 core, Q, K 會有很強的相關性。Q, K 的結果和 V 也有很強的相關性。**

#### 切 Multi-Head

**再來看下圖右，Q, K, V 可以分成 multi-heads! 不同 heads 之間只有在 input 和 output (concat and linear) 才有相關性。**

以 70B 有 64 heads,  所以 Q, 可以分解成 [Q1, Q2, ... Q64],  每個 head 的 embed_size = 8192 / 64 = 128.   也就是 Qi $\in \mathbb{R}^{context \times size} = \mathbb{R}^{128k \times 128}$ = 16M parameter
一個 GPU 分到 8 heads Q1-8:  16M x 8 = 128M parameter per layer. 
如果 80 layers:  Q1-8 = 10G parameter.  或是 20GB 内存 for FP16 per GPU for Q1-8.  這和上面一致。

70B  使用 GQA,  所以 K, V 各自只有 8 heads!  每個 GPU 只要 1 個 heads.
K, V 都是16M parameter per later:  KV Cache: 16M x 80 層 x 2 (K, V) = 2.5G parameter or 5 GB 内存.

每個 GPU:
Q1-8:  10G parameter, 20GB
K1+V1:  2.56G parameter, 5 GB
Q1-8 + K1 + V1: 25 GB. 
如果是 8 個 GPU: 200GB.
如前所述，在 training Q, K, V 全部都要。但是在 inference 只有一開始 prefill Q 需要考慮全部 length.  除了 long context input.  在 generative stage, 只有 K, V cache 和一個 token Q.  
- 5GB + Q prefill length ~ 5 GB + 0~20GB ~ 5-25GB per GPU.

一般 GPU 會留 30% for KV cache: 80GB/3 = 27GB.  



<img src="/media/image-20241019083903.png" alt="20241019083903" style="zoom:60%;" />
### Weight Memory:

70B x 2 = 140B,   140B / 8 GPU = 18 GB 左右。

### Batch = k > 1

在 batch = k > 1,  weight是一樣！因此可以 share 内存和頻寬！
但是 Q, K, V 則各自負責！以 KV cache:
5GB x K = 5K GB.  如果 64 users: 320GB! per GPU.   顯然不可能。只能 share KV cache among different batch?

## MLP or FFN block

這個部分就是 tensor multiplication,  是最容易可以做 tensor parallel 的部分。而且有好幾個方法，我們分別描述。

| **Partitioning**                                              | **Training/Inferecing**                                                               | **Memory**                                               | Communication                                                               | Others                        |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------- |
| **Batch Parallelism** (e.g. 10 batch each GPU)                | - Good for training with fixed batch<br>- NG for inferencing (batch=1, dynamic batch) | - Each GPU has complete MLP weights, no memory BW saving | - Zero                                                                      | - combined with other methods |
| **Token Parallelism** <br>(e.g. 10 tokens each GPU)           | - NG for inference (token=1 in generation)<br>- OK for MLP                            | - Each GPU has complete MLP weights, no memory BW saving | - Zero for MLP                                                              | - Not use                     |
| **Tensor Parallelism**<br>(e.g. splits large weight matrices) | - both training and inferencing<br>                                                   | - Reduces per-GPU memory usage for MLP weights           | - Only at input/output <br>- **Requires synchronization after every layer** | **- Most favorable**          |

簡單說：
- Batch parallelism 是 no brainer, 不過有 system level 的 concerns 和無法 gain 任何 memory BW saving.  
- Token parallelism 在 MLP 和 Attention 不同，在 MLP 是完全 parallelism 而且 no communication.   基本和 Batch parallelism 一樣。 
- Tensor parallelism 則是非常理想的 partition.  只有在 input 和 output 需要做 synchronization.


### 再看 MLP block,  這是比較容易的 block, 假設 batch=1 and token=1

<img src="/media/image-20241019200821.png" alt="20241019200821" style="zoom:60%;" />


MLP/FF Layer 主要 tensors:  X, X2 $\in \mathbb{R}^{l\times d} = \mathbb{R}^{128k \times 8k}$ of 1GB parameter.  X1 則是  $\in \mathbb{R}^{l\times d} = \mathbb{R}^{128k \times 8k\times 4}$ of 4GB parameters.
If using FP16, 這些 tensor 是 2GB, 2GB, and 8GB.

對於 80 層，X, X2: 2G x 80 (layers) = 160 GB x 2 = 320 GB.  X1: 8GB x 80 = 640GB  
Total: 320GB + 640 GB = 800 GB
如果 8 個 GPU, 各 100GB 内存。顯然不切實際。 X and X2 應該是必要的中間值。但是 X1 可以用時間 trade-off.  一般可能讓 X1 2GB.  所以 total = 2GB x 3 x 80 / 8 = 60 GB?

重點是在 weights,  Wv1 and Wv2 都是 $\in \mathbb{R}^{d\times 4d} = \mathbb{R}^{8k \times 8k \times 4}$ = 256M parameter per tensor.  
一共有 256M x 2 (v1, v2) x 80 = 40G parameter or 80 GB 内存。
因此每個 GPU 需要 10GB 内存。


## Embeding

Embeding and Position Encode
$W_e \in \mathbb{R}^{vocab \times d_{model}} = \mathbb{R}^{128k \times 8k} = 1G$  parameter.  不過一般是字典 (i.e. one-hot)，所以是用查表法而不用矩陣乘法，節省算力，但沒有節省 memory and memory BW.

這裡可以用切 token length 分給不同 GPUs 做 embedding.  雖然無法節省 DRAM 頻寬，但可以節省算力。結束後 allreduce 做 data sync, 因為 attention block 需要所有的 tokens 計算。但是 FF 的確可以分不同 token.




#### Llama 70B Inferencing over 8 GPUs

|                  | Type          | Parameter          | Layer      | GB     | GB/GPU      | Must | Note       |
| ---------------- | ------------- | ------------------ | ---------- | ------ | ----------- | ---- | ---------- |
| **Weight**       |               |                    |            |        |             |      |            |
| Attn             | WQ,WO weight  | 8Kx8Kx80x2=10G     | 80         | 20GB   | 2.5GB       | V    |            |
|                  | WK, WV weight | 8x1Kx1Kx80x2=1.25G | 80         | 2.5G   | 0.3125GB    | V    |            |
|                  | K, V cache    | 1Kx128Kx80x2=160G  | 80         | 40GB   | 5GB         | V    | GQA        |
| MLP              | W1, W2        | 8Kx32Kx80x2 =40G   | 80         | 80GB   | 10GB        | V    |            |
| Embedding        | We            | 128K x 8K = 1G     | 1          | 2GB    | 2GB         | V    | no sharing |
| Position         | Wp            | 8K x 8K = 64M      | 1          | 0.12GB | 0.12GB      | V    | no sharing |
|                  |               |                    |            |        |             |      |            |
| **Intermediate** |               |                    |            |        |             |      |            |
| Attn             | X             | 128Kx8Kx80=80G     |            | 160GB  | 160GB?      |      | must sync? |
| MLP              | X             | 128Kx8Kx80=80G     |            | 160GB  | 20GB?       |      | can split? |
|                  |               |                    |            |        |             |      |            |
| Allreduce        |               |                    | 80x2+2=162 |        |             |      |            |
|                  |               |                    |            |        |             |      |            |
| Total            |               |                    |            |        | 60GB-100GB? |      |            |



## Reference

*  Under the Hood of Llama 3.1 70B:  https://whatdhack.medium.com/under-the-hood-of-llama-3-1-70b-distributed-inference-8b3c03886f22
* [Llama (LLM) (devopedia.org)](https://devopedia.org/llama-llm#:~:text=Since%20this%20model%20uses%20GQA%2C%20there%20are%2064,heads%20but%20only%208%20K%20and%20V%20heads.)


