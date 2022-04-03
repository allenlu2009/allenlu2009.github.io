---
title: Vision Transformer
date: 2022-02-27 09:28:08
categories: 
- Language
tags: [Transformer, Vision]
description: Transformer is Used for Vision
typora-root-url: ../../allenlu2009.github.io
---



**先説結論**

* 深度學習的 tasks 像 vision, speech, NLP 的 backbone network 包含兩個 steps：(global) attention + (local) feature extraction.

* Attention 是一種 global information “identification”: (1) 只聚焦在重要的 features (對於大腦的好處是節省計算資源，但此處沒有這個好處); (2) 這些重要 features 的相對位置 (relative position embedding) 或關係。  **Attention 是 general concept,  可以應用在 NLP, vision, speech.** 之後有機會成爲 multi-modal applications 的 backbone.

* Feature extraction 是 local and application specific.  例如 multi-layer CNN 非常適合 vision (but not for language) 因爲它利用或榨取 (exploit) vision object 的 spatial locality 特性。 RNN like LSTM 非常適合 speech 因爲它榨取 speech 的 temporal locality 特性。Feedforward (FF)/MLP/FC network 比較 general, 用來處理 NLP 或是 vision or speech 也 OK.  所以 pure transformer 也是可以處理 NLP, Vision, Speech 問題。

  

## Attention Mechanism



#### Why Attention?



#### Attention 自然語言例子和 hand waving 解釋 

給一個句子： "The animal didn't cross the street because it was too wide." 包含 12 個 tokens (words, punctuation marks).

Attention 的概念 (self-attention in this case, cross-attention in translation) 就是選擇任一個 token, 可以計算出 score (between 0 and 1) of correlation.  

這個 attention score 這有什麽用？ 

* mimic 大腦的機制，把有限的資源 (computation resource) 放在重要的部分。一方面 maximize resource, 另一方面避免資訊 overload (overfit).

* 當然 attention 只是其中一步。Downstream task 例如 Q&A, sentimental analysis, 以及 translation 會利用 attention score 作爲 input 訓練出這些功能。

  



<img src="/media/image-20220228084102554.png" alt="image-20220228084102554" style="zoom:50%;" />



#### Attention Vision 例子和 hand waving 解釋 

什麽是人臉 (1) 要有人臉重要的 features，例如眉、眼、鼻、嘴；(2) 重要的 features 位置很重要。不重要的 features 像是痣、鬍子、沒有或是放在不對的位置也沒關係。

CNN 可以做到 1, 但很難學習到 2.  除非有比較大的 global information.



Vision dominant 的 NN 是 CNN.  

* CNN is good at local feature extraction, but not on the global information.

例如下圖左和右從 CNN 的角度都是一張人臉。 



<img src="/media/image-20220227233505107.png" alt="image-20220227233505107" style="zoom:50%;" />

相反 transformer 比較 focus on global information.  large receptive fields are required to track long-range dependencies.

* CNN 雖然可以藉助更深的網絡增加 receptive field.  但只是一個逐漸擴大範圍提取特徵過程，CNN 本身沒有 idea 什麽是重要的特徵 (e.g. 眼睛，鼻子，嘴巴)，更不用說重要特徵的相對位置加入 training.  而且 CNN feature map 本身就 embed 絕對位置，似乎也很難在重新加入相對位置的 embedding.   **CNN 的強處是影像的特徵提取，使用標準的 image filter。同時因爲 share weight/filter, 可以大量節省 computation!**  How about use transformer in voice?

* **Attention 剛好和 CNN 相反。Attention 本身是完全 permutation (i.e. position) invariant or insensitive.  乍看這是一個問題，但也正是因爲如此，可以外加 position embedding, 不論是絕對位置或是相對位置。**  **Attention 的缺點是沒有特別強的特徵提取，只是用 general feedforward network (FE/MLP/FC). 雖然 general, 但對於不同應用如影像或是聲音的特徵提取，就不如 CNN or LSTM/GRU 更強！**

  

### How Attention Work: 1. Query, Key-Value pair (Q, K, V)

我們可以這樣來看待Attention機制 (參考下圖) : 將Source (e.g. 一句話)中的構成元素想像成是由一系列的<Key,Value>數據對構成，此時給定Target中的某個元素Query，通過計算Query和各個Key的相似性或者相關性，得到每個Key對應Value的權重係數，然後對Value進行加權求和，即得到了最終的Attention數值。所以本質上Attention機制是對Source中元素的Value值進行加權求和，而Query和Key用来计算对应Value的权重系数。即可以将其本质思想改写为如下公式：

$$ \text{Attention}(Query, Source) = \Sigma_{i=1}^{L_x} \text{Similarity}(Query, Key_i) * Value_i $$

* Input: Query, Key-Value 聽起來像是 database;  output: attention weight for the query? or key-value pairs?
* Key-Value

<img src="/media/image-20220228151559231.png" alt="image-20220228151559231" style="zoom:50%;" />



<img src="/media/image-20220228154213580.png" alt="image-20220228154213580" style="zoom:50%;" />



**階段 1:  計算 Query 和 Key 的相似性。Input：Query, Key;  Output：Similarity **

這裏可以用各種不同的函數如下：

Dot-Product Similarity : $\text{Similarity}(Query, Key_i) = \text{Query} \cdot \text{Key}_i $

Cosine Similarity : $\text{Similarity}(Query, Key_i) = \frac{\text{Query} \cdot \text{Key}_i}{\|\text{Query}\|  \|\text{Key}_i\|} $

MLP 網路 : $\text{Similarity}(Query, Key_i) = \text{MLP}(\text{Query} , \text{Key}_i) $



**階段 2:  Normalize Score。Input: Similarity.  Output: Normalized Similarity **

第一階段產生的分值根據具體產生的方法不同其數值取值範圍也不一樣，第二階段引入類似SoftMax的計算方式對第一階段的得分進行數值轉換，一方面可以進行歸一化，將原始計算分值整理成所有元素權重之和為1的概率分布；另一方面也可以通過SoftMax的內在機制更加突出重要元素的權重。即一般採用如下公式計算：

$$ a_i = \text{Softmax} (\text{Sim}_i) = \frac{e^{Sim_i}}{\Sigma^{L_x}_{j=1} e^{Sim_j}} $$



**階段 3:  Normalize Score。Input: Normalized Similarity, Value.  Output: Attention **

$$ \text{Attention}(Query, Source) = \Sigma^{L_x}_{i=1} a_i \cdot Value_i$$



<img src="/media/image-20220301215444704.png" alt="image-20220301215444704" style="zoom: 80%;" />

通過如上三個階段的計算，即可求出針對Query的Attention數值，目前絕大多數具體的注意力機制計算方法都符合上述的三階段抽象計算過程。

**總結 attention：**
$$
\operatorname{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})=\operatorname{softmax}\left(\frac{\boldsymbol{Q} \boldsymbol{K}^{\top}}{\sqrt{d_{k}}}\right) \boldsymbol{V}
$$
where $\boldsymbol{Q} \in \mathbb{R}^{n \times d_{k}}, \boldsymbol{K} \in \mathbb{R}^{m \times d_{k}}, \boldsymbol{V} \in \mathbb{R}^{m \times d_{v}}$ 。如果忽略 activation function softmax，事實上這就是三個矩陣相乘，最後的結果是 $n\times d_v$ 的矩陣。所以我們可以認爲：這是一個 attention layer, 將  $n\times d_k$ 的序列 ***Q*** 編碼成一個新的  $n\times d_v$ 的序列。



### 2. Multi-Head Attention (Google)

这个是Google提出的新概念，是Attention机制的完善。它的觀念非常簡單，就是對於同樣的 ***(Q, K, V)***, 同時有多組的 attentions output.  這似乎也合理，同樣一句話，因爲上下文不同，可能有超過一種 attentions。有點像一般自然語言的歧義的意味。 For example,

The animal didn't cross the **street** because **it** was too wide.  中的 it 是指 street (highest attention), but

The **animal** didn't cross the street because **it** was too tired.  中的 it 是指 animal (highest attention).  



Multi-Head Attention (MHA, 多頭注意力) 就是爲了增加學習多種可能的 attentions.

做法直觀暴力，就是把 ***Q,K,V*** 通過參數矩陣映射一下，然後再做Attention，把這個過程重複做 $h$ 次  (2-5 depends on applications?) ，結果拼接 (concatenate) 起來就行了，可謂「大道至簡」了，如下圖。 



<img src="/media/image-20220305162924195.png" alt="image-20220305162924195" style="zoom: 67%;" />

數學表示式：

$$
\text {MultiHead}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})=\text {Concat}\left(\text {head}_{1}, \ldots, \text {head}_{h}\right) \boldsymbol{W}^O
$$

where  $\boldsymbol{W}^{O} \in \mathbb{R}^{h d_{v} \times d_{v}}$ and  

$$
\text{head}_i = \text{Attention}(\boldsymbol{Q} \boldsymbol{W}^Q_i, \boldsymbol{K} \boldsymbol{W}^K_i, \boldsymbol{V}\boldsymbol{W}^V_i)
$$

where $\boldsymbol{W}_{i}^{Q} \in \mathbb{R}^{d_{k} \times \bar{d}_{k}}, \boldsymbol{W}_{i}^{K} \in \mathbb{R}^{d_{k} \times \bar{d}_{k}}, \boldsymbol{W}_{i}^{V} \in \mathbb{R}^{d_{v} \times \bar{d}_{v}}$  and $\bar{d}_{k} = \bar{d}_{v} = d_v/h$ 



實際看一個 $h=5$ 的例子如下。不同的 $head_i$ 會學到不同的 attention.

<img src="/media/image-20220305100538823.png" alt="image-20220305100538823" style="zoom:50%;" />

#### Ex1:  Translation: Encoder to Decoder

用實際的例子來看 Q, K, V (all vectors).   以下 encoder 的 compatibility function 就是 similarity + normalization.

Decoder 的  $z_0$ 是 query，而 key 就是 encoder 的 input  $h^1, h^2, ...$     所以 $z_0$ 會和 $h_1, h_2, ....$ 都會通過 compatibility function 算出吻合程度 $a_0^1, a_0^2, ...$ 就是權重。這個權重在和對應的 value 做 linear combination。 這裏爲了簡單讓 key 和 value 一樣，都是 $h^1, h^2, ...$  其實可以是不同的東西。計算的結果 $c^1$ 就是 context vector,  作爲 decoder 的輸入，與 $z_0$ 一起計算出 $z_1$。

這樣算是完成一輪 attention mechanism。下次再繼續用 $z_1$ 當成 query 進行比對。 

<img src="/media/image-20220228211615187.png" alt="image-20220228211615187" style="zoom:67%;" />

如此一來，就可以以動態的方式去產生序列了。Encoder 負責的是將輸入的序列轉成固定大小的向量，decoder 將這樣的向量轉換回序列，而中間需要動態調整的部份就像人的注意力一樣，會去掃視跟比對哪個部份的翻譯是最吻合的，然後將他做一個線性組合的調整。



#### Ex2:  Self Attention (Google): Encoder Only

而在Google的论文中，大部分的Attention都是Self Attention，即“自注意力”，或者叫内部注意力。

Self attention 其實是 $Attention(\boldsymbol{X}, \boldsymbol{X}, \boldsymbol{X})$.  也就是說，在序列內部做Attention，尋找序列內部的聯繫。**Google論文的主要貢獻之一是它表明了內部注意力在機器翻譯（甚至是一般的Seq2Seq任務）的序列編碼上是相當重要的，而之前關於Seq2Seq的研究基本都只是把注意力機制用在解碼端。**類似的事情是，目前SQUAD閱讀理解的榜首模型R-Net也加入了自注意力機制，這也使得它的模型有所提升。

當然，更準確來說，Google所用的是Self Multi-Head Attention.



Q: 最小單位是 token, 可以是一個字或是標點符號。 How about for voice or image or video? 

Q: should we make this permutation dependent? or permutation indepedent?

A: by nature, the pair-wise is permutation independent, 因爲我們只 care 比較大值，而不 care position.  但實際文法需要, e.g.  I eat tiger is different from tiger eats me.  所以需要 position encode to make it permutation variant.



所以最後有 12 x 12 scores for self-dimension with input 12 dimension?  這只是 network structure with input sensitive to context, 並不是固定存了這些 scores.  



### 3. Positional Encoding

To be added, very important:  No position encoding: permutation invariant.   With position encoding: permutation sensitive.  基本上一個是排列 (permutation sensitive); 另一個是組合 (permutation invariant).

Transformer 目前使用 a fixed sinusoidal function as the positional encoder (PE)。看似聰明，但只是一個 patch to solve the permutation invariant problem。希望之後有更合理的 solution.

How about partially permutation sensitive?  e.g. eyes can permute; but eye and mouth cannot permute? similarly use position encoder + concatenate no position encoder?   **Use different symmetric function (odd function, even function?)**



### 4. Transformer 合體

![image-20220305234923675](/media/image-20220305234923675.png)



## Vision Transformer

是否能把 transformer 作爲 vision backbone network, 類似 CNN 在 vision backbone network 的角色。可以用於 vision 底層 task 如 detection, segmentation, 甚至 quality enhancement；以及中高層的 vision task 像是 classification,  image caption, 等等。

CNN 的優點是：computation efficiency, low level feature extraction (filter-like building block), translation equivariant for low level vision。藉著 bottom up to increase the receptive field and pooling to offer scaling invariant and translation invariant.



|                                             | CNN in Vision                        | Transformer in Vision | Transformer in NLP               |
| ------------------------------------------- | ------------------------------------ | --------------------- | -------------------------------- |
| Low level vision<br>translation equivariant | Yes                                  | Need to solve         | No need                          |
| Low level vision <br>scale equivariant      | CNN Pyramid                          |                       | No need                          |
| High level<br>translation invariant         | feature pyramid+FC layer             |                       | Yes                              |
| High level <br>scale invariant              | feature pyramid+FC layer             |                       | NO                               |
| High level<br>Permutation sensitive         | Position embedding                   |                       | Position encoding                |
| Vision charateristics                       | best fit for vision spatial locality |                       | No constraint by vision locality |



CNN Pro and Con

|                 | CNN                     | Transformer/Attention |
| --------------- | ----------------------- | --------------------- |
| Receptive field | small, need deep layers | Wide, use FC layer    |
| Feature         | local feature           | long range feature    |
| Scope           | bottom up               | Top down              |

combine feature extraction and ...?



<img src="/media/image-20220319101124736.png" alt="image-20220319101124736" style="zoom: 67%;" />

本文主要討論 attention 用於 vision.  有兩種類型:  (I) pure transformer, Vision Transformer or ViT;  (II) combined CNN + attention （本文不討論）.

Why introduce transformer or attention to vision?  (1) better performance for classification, detection, segmentation, or quality;  (2)  vision + NLP tasks such as vision caption, etc.







### Self Attention Module (Type II, Use CNN feature map, 本文不討論)

Straightforward:

1. image or feature patch patch use for self-attention

<img src="/media/image-20220227234913682.png" alt="image-20220227234913682" style="zoom: 67%;" />

2. Feature patch use for self-attention



三部曲：

1. image 
2. Image pyramid (spatial pyramid) for scaling invariant
3. Image + feature pyramid (CNN) for scaling invariant and receptive field 



Transformer Pro and Con

Separate feature extraction and ...?

<img src="/media/image-20220319091140496.png" alt="image-20220319091140496" style="zoom:80%;" />

Key milestones in the development of transformer.  The vision transformer models are marked in red.



<img src="/media/image-20220320221125054.png" alt="image-20220320221125054" style="zoom:80%;" />

Timeline of existing vision transformer studies including pure-attention, convolutional with attention, multi-layer perceptron, vision transformer with multi-layer perception, and self-supervise ViTs.  As presented in the figure, the progress has started from the last quarter of 2020.  ViT and DeiT is available to the community for 2020.  In 2021, a lot of vision transformers have been emerged enhancing training strategies, attention mechanisms and positional encoding techniques.  Major contributions to the ImageNet dataset for classification task is shown in this figure. 





### ViT (Vision Transformer, Transformer/no CNN, Google 2020)

我們接下來看如何應用 transformer 應用在 vision，幾個 key points: 

* Token -> patch (becomes vector)

* Positional encoder

* Self attention 

  

Image is permutation sensitive, not permuation invariant

Need positional encoder -> ViT learn the positional encoder instead of pre-determined



#### Token as Patch (注意沒有用 CNN)

想要套用 transformer model 在 vision 的應用，**第一步就是要解決什麽是 token in vision.** 在 NLP 的應用，token 基本上就是 words (含標點符號)。在語音的應用，token 就是 phoneme (音素)。

在 vision 的應用 token 就是 patch (of pixels)。到底應該多少 pixels 為一個 patch/token?  [@dosovitskiyImageWorth2021] 使用 fixed-size patch, 最好的 patch 大小是 14x14(xC, pixels including channel length) 。ImageNet 的 input image pixel 是 256x256, 不過 crop 之後是 224x224 pixels。所以一張圖對應的 token = (224/14 x 224/14) = 16 x 16 = 256 tokens。 也因此該篇論文的標題是 "AN IMAGE IS WORTH 16X16 WORDS".



<img src="/media/image-20220227235041989.png" alt="image-20220227235041989" style="zoom: 50%;" />



#### Position Embedding to Solve Permutation Invariant

下一步要解決的問題是 positional encoder.  更正確是解決 transformer encoder 或是 attention 機制本身是 permutation invariant 的問題。 例如下圖左和右從 transformer encoder or attention 的角度是一樣的。對於大腦當然不同，除非是在玩拼圖。

ViT 並沒有采用 a fixed positional encoder, 而是用 **learnable position embedding.**  同樣作者發現 1D position embedding 就足夠，因爲 2D position embedding 似乎沒有什麽 performance improvement.

最後 train 出的結果也顯示有學習到 position feature.  這是 ViT 的一個優點 over CNN.  就是 permutation sensitive.  對於 Fig. 2 的人臉或任何影像，我們希望相對位置也要正確。  Can CNN add position embedding?



<img src="/media/image-20220305235502703.png" alt="image-20220305235502703" style="zoom:67%;" />



[@radhakrishnanWhyTransformers2021]



#### Self Attention

Structure:  **直接使用 transformer encoder** train 出分類網路。

#### ViT Attention Result

下圖顯示 ViT attention 的確有找到應該注意的地方。

<img src="/media/image-20220312005935024.png" alt="image-20220312005935024" style="zoom: 67%;" />





### Swin Transformer (Shifted Windows, Transformer/no CNN, Microsoft 2021)

ViT 是第一個把 transformer 用於 image classification 有不錯成果的嘗試。類似的嘗試是 FAIR 的 DETR，用 transformer+CNN 在 object detection.  **下一步要考慮的是 transformer 用於 vision 的汎化性**。是否能把 transformer 作爲 vision backbone network, 類似 CNN 在 vision backbone network 的角色。可以用於 vision 底層 task 如 detection, segmentation, 甚至 quality enhancement；以及中高層的 vision task 像是 classification,  image caption, 等等。



**ViT 最大的問題是直接套用 NLP 的 encoder model, 完全沒有考慮 image 的特性。**

Transformer 用於 vison 的汎化性，有幾個問題要解決：

1. 受限於圖像的矩陣性質，一個能表達信息的圖片往往至少需要幾百個像素點，而建模這種幾百個長序列的數據恰恰是Transformer的天生缺陷，e.g. ViT 16 (patch in H) x16 (patch in W)=256 1D tokens，會造成龐大的算力需求。這不是最大的問題。

2. 建模能力方面，強行分割patch破壞了原有的鄰域結構，也不再具有卷積的那種空間不變性。
3. 複雜度方面，之前的 ViT 是在每層都做全局 (global) 自注意力。如果保持每個 Patch 的大小不變 (e.g. 14 pixel x14 pixel = 1 patch in ViT)，隨著圖片尺寸的變大，Patch的個數會增加，而Patch的個數等於進入Transformer的Token個數，且Transformer的時間複雜度是 $O(n^2)$。也沒有如 CNN 感受域 (receptive field) 逐步增加的架構。patch 和 patch 之間的 interaction 都只在同一個 patch size.   14 pixel x14 pixel 的大小可能對於底層視覺如 object detection or image segmentation 不夠細膩。但對於高層視覺如 classification or image caption 又有感受域不夠的問題。
4. 易用性方面，由於Embedding（結構是全連接）和圖片大小是綁定的，所以預訓練、精調和推理使用的圖片必須是完全同等的尺寸。
5. 目前的基於 Transformer 框架更多的是用來進行 image classification。但是對 image segmentation 這種密集預測的場景Transformer並不擅長解決。

Swin (Shifted Window) transformer [@liuSwinTransformer2021]  address 以上問題，**並且在分類，檢測，分割任務上都取得了SOTA的效果。**Swin Transformer的最大貢獻是提出了一個可以廣泛應用到所有計算機視覺領域的backbone，並且大多數在CNN網絡中常見的超參數在Swin Transformer中也是可以人工調整的，例如可以調整的網絡塊數，每一塊的層數，輸入圖像的大小等等。該網絡架構的設計非常巧妙，是一個非常精彩的將Transformer應用到圖像領域的結構，值得深入鑽研。

在Swin Transformer之前的ViT和iGPT，它們都使用了小尺寸的圖像作為輸入，這種直接resize的策略無疑會損失很多信息。與它們不同的是，Swin Transformer的輸入是圖像的原始尺寸，例如ImageNet的224*224。另外Swin Transformer使用的是CNN中最常用的層次的網絡結構，在CNN中一個特別重要的一點是隨著網絡層次的加深，節點的感受野也在不斷擴大，這個特徵在Swin Transformer中也是滿足的。**Swin Transformer的這種層次結構，也賦予了它可以像FPN[6]，U-Net[7]等結構實現可以進行分割或者檢測的任務。**Swin Transformer和ViT的對比如圖1。



<img src="/media/image-20220312190856873.png" alt="image-20220312190856873" style="zoom:50%;" />

Figure 1. (a) The proposed Swin Transformer builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red). It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks. (b) In contrast, previous vision Transformers [20] produce feature maps of a single low resolution and have quadratic computation complexity to input image size due to computation of selfattention globally.



#### Swin Transformer 結構

Swin Transformer 共提出四個網路框架，從小到大依次是 Swin-T, Swin-S, Swin-B, Swin-L.  下圖以 Swin-T 爲例。核心部分是 4 stage 的 Swin Transformer Block.  **下一個 stage 是把前一個 stage 的 patch merge, 增加感受域 (receptive field).  這裏的概念和 CNN encoder 非常類似：每後一層的感受地域都比前一層更大。只是 CNN 是用 stride=2 和 max pooling 提升感受域。Swin transformer 是用 patch merge 的方法達成。** 

這是 backbone network 必須要有的特性。因爲底層視覺功能 (detection, segmentation) 是 translation equivariant.  但是高層視覺功能 (classification) 卻是 translation invariant. 

##### Swin 分成幾個階段

1. Embedding Stage（stage1）。將圖片劃分為若干 4x4 的patch，使用線性變換來將 patch 變為 Embedding向量，這一步和 ViT 是一樣的。但是注意，這裡的 patch 比 ViT 的 14x14 小了很多。
2. 若干個使用 Swin Transformer 的 Stage（stage2-4）。這裡模仿了經典 CNN backbone的結構，在每個Stage都將feature map（對應到 ViT 就是 Patch 或 Token 的個數）變成原來的四分之一。這是通過簡單地將2x2 patch 合併成一個來完成的。同時，用 Swin Transformer 替代了原來的標準 Transformer。
3. Swin transformer 的 block 如下圖 (b)。LN: Layer Norm ; MSA: Multi-head Self Attention ; W-MSA: Window (MxM) MSA ; SW-MSA: Shift Window MSA ; MLP: Multi-Layer Perceptron, 就是 feed-forward or fully connected network.  Swin transformer 的基本結構和 transformer encoder 一樣，只是把 LN 放到前面。其他主要變化如下
   * 用 MxM 大小的 W-MSA 代替 global MSA。因為自注意力機制時間複雜度是 $O(n^2)$，通過減少參加自注意力的元素，將原來關於patch數平方複雜度的計算變為關於patch數線性複雜度。這應該是對的 approach, 因為 **vision objects tend to have spatial locality。不需要 global self-attention.**
   * 用對角線方向的 shift 來使 Swin Transformer 里的每一層 window 都是不同的，也就是 SW-MSA。這樣一個 patch 有機會和不同的 patch 交互。這裡還使用了一個mask trick來使得這種shift的自注意力計算更高效。如下下圖。
   * 添加了相對位置 bias (RPB, relative position bias)，對比發現這比使用絕對位置 embedding 效果好很多。



<img src="/media/image-20220312212605133.png" alt="image-20220312212605133" style="zoom:80%;" />



<img src="/media/image-20220315215116327.png" alt="image-20220315215116327" style="zoom: 67%;" />

簡單 summarize Swin Transformer vs. ViT

* Token 不再是 fixed size patch (14pixel x 14pixel or 16pixel x 16pixel)，而是隨著 layer 變深，patch 會 (shift) 移動，而且會 merge 增加 receptive field.
* 絕對位置的 embedding 改成 parameterized 相對位置 bias (RPB, Relative Position Bias).
* Transformer encoder 修改成 Swin transformer。主要是采用 MxM window 可以**大幅減少計算量**。



從結果看，Swin-T 相比于 ViT-B/16 (16pixel x 16 pixel token) parameter size, FLOP 都減少很多，但 top-1 accuracy 反而變好。但比起 CNN based 的 EffNet-B3 還是有所不如。

![image-20220313221559508](/media/image-20220313221559508.png)



##### Ablation Result

綜合消融實驗的結果可以對比三種不同的attention方式: fixed window (in ViT)、sliding window和shifted window的性能。他們的imagenet top1 acc分別是80.2， 81.4和81.3。從中可以看出類似於卷積的sliding window性能是最好的，無奈太慢了。fixed window丟失了很多有用的窗口間交互，性能最差。shifted window性能相比sliding window下降微弱，但速度提升了好幾倍。同樣可視為fixed window的ViT只能得到0.78的top1 acc，我想這是小 patch 帶來的差別，因為現在的線性變換 embedding 實在太弱了，**patch越大帶來的信息丟失就越多。**



### Swin Transformer v2 (Transformer/no CNN, Microsoft 2021)

Swin Transformer v2 主要是解決模型上規模的問題，有幾個主要的改動，如下圖：

1. 把每個Block里的 LN (Layer Norm) 從前面換到了後面，來解決深度增加之後訓練不穩定的問題。這好像回到原來 transformer attention 的結構 $\to$ minor modification.
2. 把原來的scaled dot attention換成了scaled cosine attention，也是為了解決訓練不穩定的問題（否則可能被某些像素對的相似度主導）$\to$ minor modification.
3. 改進 RPB 相對位置 bias。V1版里這個模塊是用一個規模跟窗口大小M相關可學習參數矩陣來處理的，如果預訓練和finetune時M大小改變，就用插值來生成原來不存在的值。V2版首先是引入了一個小網絡來取代參數矩陣，其次是將相對位置從線性空間換到了對數空間，通過取對數壓縮空間差距來讓M變化時的過渡更加順滑。$\to$ **major modification.**

<img src="/media/image-20220318174153907.png" alt="image-20220318174153907" style="zoom:50%;" />



RPB 的 ablation result 如下表。Log-Spaced CPB (Continuous Position Bias) > Linear-Spaced CPB > RPB.

<img src="/media/image-20220318180345460.png" alt="image-20220318180345460" style="zoom:67%;" />



從結果來看，更大的網絡確實帶來了更好的性能，3B 參數版的 SwinV2-G 比 88M 參數版的 SwinV2-B 性能提升了不少。同樣參數量的 V2 版也比 V1 版提升了一些。

<img src="/media/image-20220318220249478.png" alt="image-20220318220249478" style="zoom: 67%;" />



In summary,  Swin V1/V2 針對 ViT 最大的改善：

1. Fixed size patch (14x14 or 16x16)$\to$ variable size (4x4 and above) and shifted patch **with increased receptive field.**  這也讓 Swin 適合做 backbone for classification, detection, segmentation.
2. Absolute position embedding $\to$ **relative position bias (RPB)** or log-spaced continuous position bias (CPB).  增加準確度，也更 robust.
3. Global image self attention $\to$ windowed self attention **大幅減少計算量**。OK for image (with spatial locality)，但有可能損失準確度。



[@hanSurveyVision2020] 討論其他很多的 models.  基本上都是針對以上幾點的改進。例如

**TNT** 是把 ViT 的 fixed patch divides into a number of sub-patches and introduce transformer-in-transformer to model the relationship between sub-patches. - (1)

**KVT** 引入 k-NN attention to utilize locality of images patches and ignore noisy token by only computing attentions with top-k similar tokens.  - (3)

**CrossFormer** 引入 CEL (Cross-scale embedding layer) and LSDA (Long Short Distance Attention) 彌補之前架構在建立跨尺度 attention 方面的缺陷 (1) and (3).  另外又引入 DPB (Dynamic Position Bias) 讓相對位置 bias 更加靈活，更好適合不定尺寸的 image and window.  





**Image+feature pyramid 非常重要 for low level vision detection/attention/image segmentation, optical flow!  => Scale Invariant! =>  face detection**

在 MTCNN,   



## Reference

