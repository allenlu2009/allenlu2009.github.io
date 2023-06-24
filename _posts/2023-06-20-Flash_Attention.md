---
title: Flash Attention
date: 2023-06-20 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---



## Attention is all you need

Attention 已經是必備的 core network.   相較於 CNN,  attention 最大的問題是 memory bandwidth.

主要在計算 K, Q 的 correlation, 以及 softmax.





### Flash Attention

FA 的主要思路就是通過 tile 技術減少在 DRAM 和 on-chip SRAM 讀寫實際。GPT2 有三倍加速 (seq length = 1024)



核心就是

1. 計算 softmax 不需要全部數據 (768x768),  可以分段計算。
2.  Back-propagate 不存儲 attention matrix (768x768), 只需要存儲 softmax normalization 的係數



<img src="/media/image-20230621224412245.png" alt="image-20230621224412245" style="zoom:67%;" />

如果以 ChatGPT-2 small 為例

K: 768x768 and Q: 768x768

每次 tokens 進來先做一次mapping to 最大是 Q 768 tokens (assuming 1 head), and K 768 tokens.  

Perform (768x768)  correlation = S = QK^T,  

接下來要做 768 個 **P = softmax(S)** softmax operation (with token number, 以 row 為單位)。

最後 O = P V. 768x768 x (768x1) = 768 x 1  



<img src="/media/image-20230620203324060.png" alt="image-20230620203324060" style="zoom:80%;" />

一般 GPU 會把 S and P 放在 HBM, which takes O(N^2) memory.  通常 N >> d (GPT2 medium, N = 1024, and d = 64)

<img src="/media/image-20230620203746623.png" alt="image-20230620203746623" style="zoom:80%;" />

<img src="/media/image-20230620201108355.png" alt="image-20230620201108355" style="zoom:80%;" />

#### 如何引入 Trainable Parameter?

* 如何做到？非常容易！  重組 input 引入 V (value) matrix.  引入 K, Q matrix and similarity/attention matrix!!
* 使用 K, Q 計算 similarity matrix (512x512)，then softmax for attention matrix.   V 通過 attention matrix 得到 output!
* 因為 attention matrix 的遠小於 768!  所以有類似 low rank 的功效。 
* V, K, Q 的 dimension:
  * V: 768x768,  K: 768x768,  Q: 768x768.  Total:  3 x (768x768)

<img src="/media/image-20230402220443383.png" alt="image-20230402220443383" style="zoom:80%;" />

* 最後再加上一個 MLP layer. Hidden layer 的 dimension 是 4x768!,  所以 parameter = 768^2 x 4 x 2? = 5M parameters!

<img src="/media/image-20230402220500305.png" alt="image-20230402220500305" style="zoom:80%;" />

* 一個 transformer block 的 parameter = V,K,Q x 768^2 = 3 x 768^2 = 2M param + 5 M= 7.1 M parameters

<img src="/media/image-20230402220521798.png" alt="image-20230402220521798" style="zoom:80%;" />

* BERT 有 12 blocks, giving ~ 85M params (再加上 25M for token embedding, 512 x 768 = 0.39M??)
* GPT2 = ? 



<img src="/media/image-20230402220542789.png" alt="image-20230402220542789" style="zoom:80%;" />

* BERT 和 GPT training 方式不同： BERT 使用 masked token.   GPT 使用 predicted token.









NLP 的問題是 input text string 一般是變動的，例如：“Hello, world",  或是 "This is a test of natural language processing!"

Input 是 text string, 切成 tokens ($\le$512).  儘量塞 sentence 或是 0 padding.  每個 token 是 768-dim (feature) vector. 也就是 (input) token embedding 是一個 arbitrary width ($\le$ 512) 2D matrix X.  最終希望做完 attention 運算還是得到同樣的 shape.



#### 先看不好的方法 for Natural Language Porcessing

* 如果 input-output 是 inter-token 和 inter-feature 的 fully-connected network, 顯然不可行！
  * 因為是一個 $(512\cdot 768)^2 = 154$ B weights，同時 computation 也要 154 T operation!
  * Input 是變動的長度, 所以固定的 154B weights 無法得到不同 width 的結果。


<img src="/media/image-20230402210446588.png" alt="image-20230402210446588" style="zoom:80%;" />

* 如果 input-output 只作用或處理在 embedding dimension (i.e shared weight for all input tokens!), 例如 1-dimension convolution, kernel 就是 1x768, channel length = 768.  假設 input 是 3-channel (e.g. RGB or KQV), parameter = 3 x 768^2 = 2M parameters.  顯然也不夠力。同時 each token 都是獨立處理，缺乏 temporal information!  

<img src="/media/image-20230402220358950.png" alt="image-20230402220358950" style="zoom:80%;" />



#### Catch

* **我們需要找一個方法介於 fully connected model and 1-d convolution network!!!**
  * Fully connect network size:  (512x768)^2 = 154B
  * 1-dimension network size: (768x768) < 1M
* 所以需要如同下面的方法！計算  $f(X) = X X^T X$.  假設 X 是 m x n -> (m x n)  (n x m) (m x n) = m x n 得到和原來一樣的 shape! 
* 此時 token 和 token 之間 interact, 但又不像 fully connected 這麼多 interaction!!!  這就是 attention 的原理！
* Rank = min(m, n)！ 如果 width 很小，例如短句。或是 attention 範圍小。rank 就小。計算量就小，也避免 overfit!
* **問題是以下的方法，沒有任何 trainable parameter!!!!**

<img src="/media/image-20230402220424409.png" alt="image-20230402220424409" style="zoom:80%;" />

#### 如何引入 Trainable Parameter?

* 如何做到？非常容易！  重組 input 引入 V (value) matrix.  引入 K, Q matrix and similarity/attention matrix!!
* 使用 K, Q 計算 similarity matrix (512x512)，then softmax for attention matrix.   V 通過 attention matrix 得到 output!
* 因為 attention matrix 的遠小於 768!  所以有類似 low rank 的功效。 
* V, K, Q 的 dimension:
  * V: 768x768,  K: 768x768,  Q: 768x768.  Total:  3 x (768x768)

<img src="/media/image-20230402220443383.png" alt="image-20230402220443383" style="zoom:80%;" />

* 最後再加上一個 MLP layer. Hidden layer 的 dimension 是 4x768!,  所以 parameter = 768^2 x 4 x 2? = 5M parameters!

<img src="/media/image-20230402220500305.png" alt="image-20230402220500305" style="zoom:80%;" />

* 一個 transformer block 的 parameter = V,K,Q x 768^2 = 3 x 768^2 = 2M param + 5 M= 7.1 M parameters

<img src="/media/image-20230402220521798.png" alt="image-20230402220521798" style="zoom:80%;" />

* BERT 有 12 blocks, giving ~ 85M params (再加上 25M for token embedding, 512 x 768 = 0.39M??)
* GPT2 = ? 



<img src="/media/image-20230402220542789.png" alt="image-20230402220542789" style="zoom:80%;" />

* BERT 和 GPT training 方式不同： BERT 使用 masked token.   GPT 使用 predicted token.





## Convolution is all you need

Input 大多是 256x256x3 (RGB) pixel (or 考量 data augmentation +/-16 pixel: 224x224 pixel ).

如果是 generative model, 例如 de-noise, de-blur, output 和 input dimension 一樣。

因爲不是分類問題，沒有用 fully-connected layer.



#### 先看不好的方法 for Computer Vision

* Fully connected trainable parameter:  (256x256x3)^2 = 38B!  雖然大力出奇蹟。但這只是小圖。如果是大圖 4K x 4K,  參數增加 16 倍。基本上 fully connected network 不 scalable!
* 如果是一次 depth-wise + point-wise trainable parameter = (256x256)^2 x 3 + 3x3 = 12.9B!  雖然比較小，還是非常大，not scalable!



#### 可行的方法

三種方法：(1) convolution;  (2) transformer; (3) hybrid.  

#### **(1) 第一種方法 convolution + hierarchy (down-sampling/up-sampling)**

Kernel: 3x3,  Cin = 3,  Cout=64 => first layer: (shared) trainable weights = 3x3x3x64 = 1.7K!! 非常小。

* 但是 計算量 (要掃過所有圖區域)  >> trainable parameters.  就是計算量 dominate!   好處是 power efficiency 比較好，可以不用一直拿 weights.
* Kernel 小的缺點是 receptive field 非常小。需要 down-sampling and up-sampling 增加 receptive field! 但同時增加 channel depth (feature) 避免 loss information.
  * 所以後面的 kernel trainable parameters 會增加:  **3x3x512x1024 = 4.7M,  還在可控範圍！就算是 20 layers 也不過 ~80M** parameters!  而且如果是 4K x 4K,  parameter 是 linear 增加，而不是幾何增加！ 
* **如果還要更少 trainable weights, 可以用 depth-wise + point-wise (例如 MobileNet) => 3x3x512 + 1x1x512x1024 = 529K, 大約只有 4.7M 的 11%。** 



#### **(2) 第二種方法 Vision Transformer (ViT)**

如果是用 3x3 pixel 為一個單位 (patch)，256x256 picture 就會有 7281 patches, 對於 transformer model 顯然太多。

ViT 使用 16x16 pixel 為一個 patch,  256x256 picture 就有 256 patches/tokens, 剛好可以給 transformer 使用！

如果一個 layer 是 7.1M parameter,   **12 layers 就是 84M pixel.  好像還是在可接受範圍！**  



* ViT 原理一樣！1-patch = 16x16x3 = 768 dimension.  224x224 image, 一共有 196 patches (<512 token).

  <img src="/media/image-20230402220606024.png" alt="image-20230402220606024" style="zoom:80%;" />

* ViT 另一個簡化是使用 1-convolution 可以 work!!  也就是 shared weight!

  

<img src="/media/image-20230402220640057.png" alt="image-20230402220640057" style="zoom:80%;" />



##### ViT-22B





<img src="/media/image-20230406213629587.png" alt="image-20230406213629587" style="zoom:67%;" />



ViT-22B 是一個基于Transformer架構的模型，和原版ViT架構相比，研究人員主要做了三處修改以提升訓練效率和訓練穩定性。

##### 並行層（parallel layers）


ViT-22B**並行執行**注意力塊和MLP塊，而在原版Transformer中為順序執行。

<img src="/media/image-20230406214114945.png" alt="image-20230406214114945" style="zoom:67%;" />

Google 的 PaLM模型的訓練也採用了這種方法，可以將大模型的訓練速度提高15%，並且性能沒有下降。

##### query/key (QK) normalization


在擴展ViT的過程中，研究人員在80億參數量的模型中觀察到，在訓練幾千步之後訓練損失開始發散(divergence)，**主要是由於注意力logits的數值過大引起的不穩定性，導致零熵的注意力權重（幾乎one-hot）**。


為瞭解決這個問題，研究人員在點乘注意力計算之前對Query和Key使用 LayerNorm

<img src="/media/image-20230406215726984.png" alt="image-20230406215726984" style="zoom:67%;" />



##### 刪除QKV 投影和 LayerNorms 上的 bias


和PaLM模型一樣，ViT-22B從QKV投影中刪除 bias，並且在所有LayerNorms中都沒有 bias 和centering，使得硬件利用率提高了3%，並且質量沒有下降。

不過與PaLM不同的是，ViT-22B對（內部和外部）MLP稠密連接層使用了 bias，可以觀察到質量得到了改善，並且速度也沒有下降。


ViT-22B的編碼器模組中，嵌入層，包括抽取patches、線性投影和額外的 position embedding 都與原始ViT中使用的相同，並且使用多頭注意力pooling來聚合每個頭中的per-token表徵。


ViT-22B的patch尺寸為14×14，圖象的分辨率為224×224（通過inception crop和隨機水平翻轉進行預處理），一共有 224x224/(14x14)=16x16=256 patches。



#### **(3) 第三種方法 Hybrid**:  

convolution 對於 local feature 效果不錯。但是 transformer 對於 global feature 很好。是否可以結合？

(A) 比如前幾層是用 convolution, 之後用 transformer!

(B) 或是 convolution 的 block 直接用 transformer block 替代或 vice versa 

(C) 或是把 hierarchy (down-sampling) 用在 transformer.



#### (3C) SWIN Transformer

基本是利用 CNN 的 windows (local attention) + shifted windows + patch merge + hierarchy 

只是先做 attention, 再用 local attention window + shifted attention windows (類似 CNN sliding windows!)

Vision Transformer應用到圖象領域主要有兩大挑戰：

- 視覺實體變化大，在不同場景下視覺Transformer性能未必很好
- 圖象分辨率高，像素點多，Transformer基于全局自注意力的計算導致計算量較大

針對上述兩個問題，SWIN Transformer 提出了一種**包含滑窗操作，具有"層級"設計**的Swin Transformer。

其中滑窗操作包括**不重疊的local window，和重疊的cross-window**。將注意力計算限制在一個窗口中，**一方面能引入CNN卷積操作的局部性，另一方面能節省計算量**。

ViT 是 16x16 pixel/patch, 所以有 16x16 (256x256 pixels) patch.

SWIN 用更小的 pixel 為 patch, 這樣有很多的 tokens?  (16倍) -> 使用更小的 (local) window attention.

**Window attention + shifted window attention 取代 global attention!** 

用 patch merge 取代 pooling!

<img src="/media/image-20230406221600548.png" alt="image-20230406221600548" style="zoom: 67%;" />





<img src="/media/image-20230406222418591.png" alt="image-20230406222418591" style="zoom:67%;" />



Math and figure:

* Input 

#### 





## 