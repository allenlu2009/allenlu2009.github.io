---
title: Large Multimodality Model
date: 2024-11-04 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-08-03-LLM_Trend]]"
categories:
  - GenAI
tags:
  - Agent
  - LLM
  - Multimodality
---


## Source

Sabastian Raschka: https://magazine.sebastianraschka.com/p/understanding-multimodal-llms: excellent introduction of two LMM architecture!

## Takeaway

有兩種主要的方法來構建多模態大型語言模型（LMMs - Large Multimodality Models）：

- 方法 A：統一嵌入 Decoder-only 架構
- 方法 B：Cross-modality Attention 架構

|      | A: 統一嵌入 Decoder-only            | B: Cross-attention           |
| ---- | ------------------------------- | ---------------------------- |
| 訓練   | 固定 LLM，ViT，只調 adaptor，成本低       | 固定 ViT 調 LLM 或固定 LLM 調 ViT   |
| 推理   |                                 |                              |
| 標記長度 | 文本標記 + 圖像標記                     | 文本標記（// 圖像注意力）               |
| 參數大小 | LLM + ViT                       | LLM + ViT +  cross-attention |
| 示例   | Llava1.5, Llava1.6 (+ ViT 0.6B) | Llama3.2 8B+3B, 70B+20B      |

## 兩種架構

簡要描述可以是 “**Decoder-only**” 和 “**Cross-Attention**” 的方法。

<img src="/media/image-20241104103206.png" alt="20241104103206" style="zoom:60%;" />

_**統一嵌入 decoder-only 架構**_ : 利用單一的 decoder 模型，類似於未經修改的 LLM 架構，如 GPT-2 或 Llama 3.2。在這種方法中，圖像被轉換為與原始文本標記相同嵌入大小的標記，允許 LLM 在串接後一起處理文本和圖像輸入標記。

_**Cross-modality attention 架構**_ : 採用 cross-attention 機制，直接在注意力層中整合圖像和文本嵌入。

## **方法 A：統一嵌入 decoder-only 架構**

在 decoder-only 架構中，圖像被轉換為嵌入向量，類似於標準文本 LLM 中輸入文本轉換為嵌入的方式。

<img src="/media/image-20241104105644.png" alt="20241104105644" style="zoom:60%;" />

對於處理文本的典型文本 LLM，文本輸入通常會進行 tokenization (例如，使用字節對編碼），然後通過嵌入層，如下圖所示。

類似於文本的tokenization 和嵌入，圖像嵌入是使用 Image encoder 模塊生成的（而不是 tokenizer ），如下面的圖所示。

<img src="/media/image-20241104104646.png" alt="20241104104646" style="zoom:60%;" />

上面顯示的 Image encoder 內部發生了什麼？為了處理圖像，我們首先將其劃分為較小的 patch ，就像在 tokenization 過程中將單詞拆分為子詞一樣。這些 patch 然後由預訓練的 vision transformer (ViT) 編碼，如下圖所示。

<img src="/media/image-20241104104743.png" alt="20241104104743" style="zoom:60%;" />

請注意，ViT 通常用於分類任務，因此在上面的圖中包含了分類頭。然而，在這種情況下，我們只需要 Image encoder 部分。

### **線性投影模塊的作用**

上面圖中的“線性投影”由單個線性層組成（即完全連接層）。該層的目的是將展平為向量的圖像 patch 投影到與 transformer  encoder 兼容的嵌入大小。這一線性投影在下面的圖中得到了說明。一個展平為256維向量的圖像 patch 被上投影到768維向量。

### **圖片與文本tokenization 比較**

現在我們簡要討論了 image encoder（以及作為 encoder 一部分的線性投影）的目的，讓我們回到之前提到的文本tokenization 類比，並並排比較文本和圖像的tokenization 及嵌入，如下圖所示。

<img src="/media/image-20241104105158.png" alt="20241104105158" style="zoom:60%;" />

還有一個額外的 _**projector**_ 模塊跟隨在 image encoder  之後。這個 _projector_ 通常只是另一個與之前解釋過的一樣的 _**線性投影**_ 層。其目的在於將 image encoder  輸出映射到與嵌入文本標記尺寸相匹配的一個維度，如下图所示。（正如我們稍後會看到，「projector」有時也稱作「adaptor」、「連接器」。）

<img src="/media/image-20241104105338.png" alt="20241104105338" style="zoom:60%;" />

現在，由於圖片 patch 嵌入具有與文本標記嵌入相同的維度，我們可以簡單地將它們拼接在一起作為 LLM 的輸入，如本節開頭所示。

順便提一下，我們在本節討論過的 image encoder  通常是一個預訓練過的vision transformer。一個受歡迎選擇是 [CLIP](https://github.com/openai/CLIP) 或 [OpenCLIP](https://github.com/mlfoundations/open_clip)。

然而，也有一些方法 A 的版本直接操作於 patch 上，例如 [Fuyu](https://www.adept.ai/blog/fuyu-8b)，如下图所示。

<img src="/media/image-20241104105517.png" alt="20241104105517" style="zoom:60%;" />

如上图所示，Fuyu 將輸入patch直接傳遞給線性投影（或嵌入層），以學習自己的圖片patch嵌入，而不是依賴其他模型和方法中的額外預訓練圖片encoder。這大大簡化了架構和訓練設置。

## **方法 B：跨模態注意力架構**

現在我們已經討論了統一嵌入 decoder 架構的方法來建立多模態 LLM 並理解圖片編碼背後基本概念，讓我們來談談通過跨注意力實現多模態 LLM 的替代方法，如下面總結所示


<img src="/media/image-20241104105615.png" alt="20241104105615" style="zoom:60%;" />


在上圖所示的 cross-modality attention 架構方法中，我們仍然使用之前討論的相同image encoder設置。然而，與其將patch編碼為 LLM 的輸入，我們通過 cross-attention 機制將輸入patch連接到多頭注意力層中。

這個想法與原始的 transformer 架構有關，並追溯到 2017 年的論文《Attention Is All You Need》，在下圖中突出顯示。

<img src="/media/image-20241104105808.png" alt="20241104105808" style="zoom:60%;" />

注意上圖所示的原始“Attention Is All You Need” transformer 最初是為語言翻譯而開發的。因此由一個文本encoder（圖的左側）組成，該encoder接收要翻譯的句子並通過文本 decoder （圖的右側）生成翻譯。在多模態 LLM 的上下文中，encoder 是 image encoder，而不是文本 encoder，但相同的原理適用。

Cross-attention 是如何工作的？讓我們看看常規self-attention機制內部發生的概念圖。

<img src="/media/image-20241104105951.png" alt="20241104105951" style="zoom:60%;" />

在上圖中，x 是輸入，_Wq_ 是用於生成查詢 _Q_ 的權重矩陣。類似地，_K_ 代表鍵，_V_ 代表值。A 代表注意力分數矩陣，_Z_ 是轉換為輸出上下文向量的輸入（x）。

與self-attention相比，在 cross-attention 中，我們有兩個不同的輸入來源，如下圖所示。

<img src="/media/image-20241104110142.png" alt="20241104110142" style="zoom:60%;" />

如前兩個圖所示，在self-attention 中，我們處理的是相同的輸入序列。在 cross-attention 中，我們混合或結合兩個不同的輸入序列。

**在《Attention Is All You Need》論文中的原始 transformer 架構中，兩個輸入 _x1_ 和 _x2_ 對應於左側encoder模塊返回的序列（_x2_）和右側 decoder 部分正在處理的輸入序列（_x1_）。在多模態 LLM 的上下文中，_x2_ 是image encoder的輸出。**

**請注意，"查詢/query"通常來自 decoder ，而"鍵/key"和"值/value"通常來自 encoder。**

在 cross-attention 中，兩個輸入序列 _x1_ 和 _x2_ 可以具有不同數量的元素。然而，它們的嵌入維度必須匹配。如果我們設置 _x1 = x2_，這等同於self-attention。

# Decoder-only 和 Cross-attention 模型訓練

現在我們已經討論了兩種主要的多模態設計選擇，接下來簡要談談在模型訓練過程中如何處理三個主要組件，這些組件在下圖中進行了總結。

<img src="/media/image-20241104111126.png" alt="20241104111126" style="zoom:60%;" />

與傳統的 text-only 大型語言模型（LLMs）的開發類似，多模態 LMM 的訓練也涉及兩個階段：預訓練和指令微調。然而，與從零開始不同，多模態 LMM 的訓練通常以一個預訓練的、經過指令微調的text-only LLM 作為基礎模型開始。

對於image encoder，CLIP 通常被使用，並且在整個訓練過程中通常保持不變，儘管也有例外，稍後我們將探討。保持 LLM 部分在預訓練階段固定也是常見的，專注於訓練projector，一個線性層或一個小型多層感知器。考慮到 projector 的學習能力有限，通常僅由一到兩層組成，因此在多模態指令微調（第二階段）期間，LLM 通常會被解凍，以便進行更全面的更新。然而，請注意，在基於 cross-attention 的模型（方法 B）中，cross-attention 層在整個訓練過程中都是解凍的。

在介紹了兩種主要方法（方法 A：統一嵌入 decoder-only 架構和方法 B：cross-modality attention 架構）之後，您可能會想知道哪一種更有效。答案取決於具體的權衡。

統一嵌入 decoder-only 架構通常更容易實現，因為它不需要對 LLM 架構本身進行任何修改。

**Cross-modality attention 架構通常被認為在計算上更高效，因為它不會用額外的圖像標記來過載輸入上下文，而是在 cross-attention 層中稍後引入這些標記。此外，如果在訓練過程中保持 LLM 參數不變，這種方法還能保持原始 LLM 的 text-only 性能。**

檢視幾篇最近實施這些方法的研究論文提供實際的視角。
## **Llama 3 模型系列**

_[Llama 3 模型系列](https://arxiv.org/abs/2407.21783)_ 論文（2024年7月31日）由 Meta AI 發表，然而很晚才描述但未發布他們的多模態模型，Llama 3.2 模型於 9 月 25 日正式宣布並提供。

多模態 Llama 3.2 模型有 11B 和 90B 參數版本，是使用之前描述的基於 cross-attention 的方法的圖像-文本模型，如下圖所示。

<img src="/media/image-20241104120105.png" alt="20241104120105" style="zoom:60%;" />

請注意，雖然圖中還描繪了視頻和語音作為可能的模態，但截至本文撰寫時發布的模型僅專注於圖像和文本。

Llama 3.2 使用基於 cross-attention 的方法。然而，它與之前所述的有所不同，即在多模態 LLM 開發中，我們通常會固定圖像 encoder，並僅在預訓練期間更新 LLM 參數。

在這裡幾乎採取了相反的方法：他們更新 image encoder，但不更新語言模型的參數。這是故意的，旨在保留 text-only 的能力，以便 11B 和 90B 的多模態模型可以作為 Llama 3.1 8B 和 70B text-only 模型在文本任務中的替代品。

訓練本身是通過多次迭代進行的，從 Llama 3.1 文本模型開始。在添加 image encoder和投影（這裡稱為“adaptor”）層之後，他們在圖像-文本數據上進行預訓練。類似於 Llama 3 模型的 text-only 訓練，他們隨後進行指令和偏好微調。

研究人員沒有採用像 CLIP 這樣的預訓練模型作為image encoder，而是使用了一個從零開始預訓練的vision transformer。具體來說，他們採用了經典vision transformer架構的 ViT-H/14 變體（630M參數） ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))。然後，他們在 2.5 billion 圖像-文本對的數據集上進行了五個階段的預訓練；這是在將 image encoder連接到 LLM 之前完成的。（image encoder接受 224×224 分辨率的圖像，並將其劃分為 14×14 的 patch 網格，每個 patch大小為 16×16 像素。）

由於 cross-attention 層增加了大量參數，**它們僅在每第四個 transformer 塊中添加。（對於 8B 模型，這增加了 3B 參數，對於 70B 模型，這增加了 20B 參數。）**

## **Molmo 和 PixMo：最先進多模態模型的開放權重和開放數據**

_[Molmo 和 PixMo：最先進多模態模型的開放權重和開放數據](https://www.arxiv.org/abs/2409.17146)_ 論文（2024年9月25日）值得注意，因為它承諾不僅開源模型權重，還開放數據集和源代碼，類似於僅語言的 OLMo LLM。（這對於 LLM 研究非常有利，因為它使我們能夠查看確切的訓練過程和代碼，並且還可以進行消融研究並在相同數據集上重現結果。）

論文標題中有兩個名稱，Molmo 指的是模型（多模態開放語言模型），而 PixMo（Pixels for Molmo）則是數據集。

<img src="/media/image-20241104121306.png" alt="20241104121306" style="zoom:60%;" />

如上圖所示，image encoder使用了一個現成的 vision transformer，具體來說是 CLIP。這裡的“connector”一詞指的是一個“projector”，用於將圖像特徵與語言模型對齊。

Molmo 通過避免多個預訓練階段來簡化訓練過程，而選擇一個簡單的管道，統一更新所有參數，包括基礎 LLM、connector 和 image encoder 的參數。

Molmo 團隊提供了幾個基礎 LLM 的選擇：

- OLMo-7B-1024（完全開放的模型骨幹），
- OLMoE-1B-7B（混合專家架構；最有效的模型），
- Qwen2 7B（性能優於 OLMo-7B-1024 的開放權重模型），
- Qwen2 72B（開放權重模型，性能最佳）

## **NVLM：開放前沿級多模態 LLM**

NVIDIA 的 _[NVLM: Open Frontier-Class Multimodal LLMs](https://arxiv.org/abs/2409.11402)_ 論文（2024年9月17日）特別有趣，因為它不僅專注於單一方法，而是探索了兩種方法：

- 方法 A，統一嵌入 decoder 架構（“僅 decoder 架構”，NVLM-D），和
- 方法 B， cross-modality attention 架構（“基於 cross-attention 的架構”，NVLM-X）。

此外，他們還開發了一種混合方法（NVLM-H），並提供了三種方法的直接比較。

如下圖所示，NVLM-D 對應於方法 A，NVLM-X 對應於方法 B，如前所述。混合模型（NVLM-H）的概念是結合兩種方法的優勢：提供圖像縮略圖作為輸入，然後通過 cross-attention 傳遞動態數量的patch，以捕捉更細緻的高分辨率細節。

<img src="/media/image-20241104121524.png" alt="20241104121524" style="zoom:60%;" />

簡而言之，研究團隊發現：

- NVLM-X 在高分辨率圖像上顯示出優越的計算效率。    
- NVLM-D 在與 OCR 相關的任務中達到更高的準確性。
- NVLM-H 結合了兩種方法的優勢。

與 Molmo 和其他方法類似，他們從 text-only 的 LLM 開始，而不是從零開始預訓練多模態模型（因為這通常表現更好）。此外，他們使用的是經過指令調整的 LLM，而不是基礎 LLM。具體來說，骨幹 LLM 是 Qwen2-72B-Instruct（Molmo 使用的是 Qwen2-72B 基礎模型）。

在 NVLM-D 方法中訓練所有 LLM 參數時，他們發現對於 NVLM-X，將原始 LLM 參數固定並僅在預訓練和指令微調期間訓練 cross-attention 層效果良好。

對於image encoder，他們沒有使用典型的 CLIP 模型，而是使用了 [InternViT-6B](https://arxiv.org/abs/2312.14238)，該模型在所有階段中保持固定。

projector是一個多層感知器，而不是單一的線性層。

## **Qwen2-VL：在任何分辨率下增強視覺-語言模型對世界的感知**

前兩篇論文和模型，Molmo 和 NVLM，都是基於 Qwen2-72B LLM。在這篇論文中，Qwen 研究團隊自己宣布了一個多模態 LLM，_[Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)_

這項工作的核心是他們所謂的“Naive 動態分辨率”機制（“Naive”這個術語是故意的，而不是“Native”的錯字）。這個機制允許模型處理不同分辨率的圖像，而不僅僅是簡單的下採樣，從而使圖像能以其原始分辨率輸入。

<img src="/media/image-20241104122304.png" alt="20241104122304" style="zoom:60%;" />
如上圖所示，原生分辨率輸入是通過修改的 ViT 實現的，具體是去除了原始的絕對位置嵌入，並引入了 2D-RoPE。

他們使用了一個經典的 vision encoder，擁有 6.75 億參數，並且 LLM 的骨幹有不同的大小，如下表所示。

<img src="/media/image-20241104122347.png" alt="20241104122347" style="zoom:60%;" />

訓練本身分為三個階段：（1）僅預訓練image encoder，（2）解凍所有參數（包括 LLM），以及（3）固定 image encoder並僅對 LLM 進行指令微調。

## **Pixtral 12B**

_[Pixtral 12B](https://mistral.ai/news/pixtral-12b/)_（2024年9月17日）使用方法 A：統一嵌入 decoder-only 架構，是 Mistral AI 的第一個多模態模型。不幸的是，沒有技術論文或報告可用，但 Mistral 團隊在他們的[博客文章](https://mistral.ai/news/pixtral-12b/)中分享了一些有趣的細節。

有趣的是，他們選擇不使用預訓練的 image encoder，而是從零開始訓練了一個擁有 4 億參數的encoder。對於 LLM 的骨幹，他們使用了 120 億參數的 [Mistral NeMo](https://mistral.ai/news/mistral-nemo/) 模型。

與 Qwen2-VL 類似，Pixtral 也原生支持可變圖像大小，如下圖所示。

<img src="/media/image-20241104122443.png" alt="20241104122443" style="zoom:60%;" />
# 結論

此處幾乎完全跳過了建模和計算性能的比較。首先，因為普遍存在數據污染，將 LLM 和多模態 LLM 在公共基準上的性能進行比較是具有挑戰性的，這意味著測試數據可能已包含在訓練數據中。

此外，架構組件的差異如此之大，以至於進行直接比較變得困難。因此，對 NVIDIA 開發不同版本的 NVLM 使得 decoder-only 和 cross-attention 方法之間的比較成為可能。

本文的主要結論是，多模態 LLM 可以以多種不同的方式成功構建。以下是一個總結本文中涵蓋的模型不同組件的圖示。

<img src="/media/image-20241104122540.png" alt="20241104122540" style="zoom:60%;" />





## Appendix A: English Version from Source
## Takeaway


There are two main approaches to building multimodal LLMs:

- Method A: Unified Embedding Decoder Architecture approach
- Method B: Cross-modality Attention Architecture approach

|                | A: Decoder-only                                | B: Cross-attention                                   |
| -------------- | ---------------------------------------------- | ---------------------------------------------------- |
| Training       | LLM, ViT frozen, only adjust adaptor, low cost | Adjust LLM for fixed ViT or Adjust ViT for fixed LLM |
| Inferencing    |                                                |                                                      |
| Token length   | text token + Image token                       | text token (// image attention)                      |
| Parameter size | LLM + ViT                                      | LLM + ViT + cross-attention                          |
| Examples       | Llava1.5, Llava1.6 (+ ViT 0.6B)                | Llama3.2 8B+3B, 70B+20B                              |

## Two Architectures

Briefer descriptions may be "**decoder-only**" and "**cross-attention-based**" approaches.

<img src="/media/image-20241104103206.png" alt="20241104103206" style="zoom:60%;" />

The _**Unified Embedding-Decoder Architecture**_ utilizes a single decoder model, much like an unmodified LLM architecture such as GPT-2 or Llama 3.2. In this approach, images are converted into tokens with the same embedding size as the original text tokens, allowing the LLM to process both text and image input tokens together after concatenation.


The _**Cross-Modality Attention Architecture**_ employs a cross-attention mechanism to integrate image and text embeddings directly within the attention layer.


## **Method A: Unified Embedding Decoder Architecture**

In the unified embedding-decoder architecture, an image is converted into embedding vectors, similar to how input text is converted into embeddings in a standard text-only LLM.

<img src="/media/image-20241104105644.png" alt="20241104105644" style="zoom:60%;" />

For a typical text-only LLM that processes text, the text input is usually tokenized (e.g., using Byte-Pair Encoding) and then passed through an embedding layer, as shown in the figure below.

Analogous to the tokenization and embedding of text, image embeddings are generated using an image encoder module (instead of a tokenizer), as shown in the figure below.

<img src="/media/image-20241104104646.png" alt="20241104104646" style="zoom:60%;" />

What happens inside the image encoder shown above? To process an image, we first divide it into smaller patches, much like breaking words into subwords during tokenization. These patches are then encoded by a pretrained vision transformer (ViT), as shown in the figure below.

<img src="/media/image-20241104104743.png" alt="20241104104743" style="zoom:60%;" />

Note that ViTs are often used for classification tasks, so I included the classification head in the figure above. However, in this case, we only need the image encoder part.

### **The role of the linear projection module**

The "linear projection" shown in the previous figure consists of a single linear layer (i.e., a fully connected layer). The purpose of this layer is to project the image patches, which are flattened into a vector, into an embedding size compatible with the transformer encoder. This linear projection is illustrated in the figure below. An image patch, flattened into a 256-dimensional vector, is up-projected to a 768-dimensional vector.

### **Image vs text tokenization**

Now that we briefly discussed the purpose of the image encoder (and the linear projection that is part of the encoder), let's return to the text tokenization analogy from earlier and look at text and image tokenization and embedding side by side, as depicted in the figure below.

<img src="/media/image-20241104105158.png" alt="20241104105158" style="zoom:60%;" />

an additional _**projector**_ module that follows the image encoder. This _projector_ is usually just another _**linear projection**_ layer that is similar to the one explained earlier. The purpose is to project the image encoder outputs into a dimension that matches the dimensions of the embedded text tokens, as illustrated in the figure below. (As we will see later, the projector is sometimes also called adapter, adaptor, or connector.)

<img src="/media/image-20241104105338.png" alt="20241104105338" style="zoom:60%;" />

Now that the image patch embeddings have the same embedding dimension as the text token embeddings, we can simply concatenate them as input to the LLM, as shown in the figure at the beginning of this section. 

By the way, the image encoder we discussed in this section is usually a pretrained vision transformer. A popular choice is [CLIP](https://github.com/openai/CLIP) or [OpenCLIP](https://github.com/mlfoundations/open_clip).

However, there are also versions of Method A that operate directly on patches, such as [Fuyu](https://www.adept.ai/blog/fuyu-8b), which is shown in the figure below.

<img src="/media/image-20241104105517.png" alt="20241104105517" style="zoom:60%;" />

As illustrated in the figure above, Fuyu passes the input patches directly into a linear projection (or embedding layer) to learn its own image patch embeddings rather than relying on an additional pretrained image encoder like other models and methods do. This greatly simplifies the architecture and training setup.




## **Method B: Cross-Modality Attention Architecture**

Now that we have discussed the unified embedding decoder architecture approach to building multimodal LLMs and understand the basic concept behind image encoding, let's talk about an alternative way of implementing multimodal LLMs via cross-attention, as summarized in the figure below.

<img src="/media/image-20241104105615.png" alt="20241104105615" style="zoom:60%;" />

In the Cross-Modality Attention Architecture method depicted in the figure above, we still use the same image encoder setup we discussed previously. However, instead of encoding the patches as input to the LLM, we connect the input patches in the multi-head attention layer via a cross-attention mechanism.

The idea is related and goes back to the original transformer architecture from the 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, highlighted in the figure below.

<img src="/media/image-20241104105808.png" alt="20241104105808" style="zoom:60%;" />

Note that the original "Attention Is All You Need" transformer depicted in the figure above was originally developed for language translation. So, it consists of a text encoder (left part of the figure) that takes the sentence to be translated and generates the translation via a text decoder (right part of the figure). In the context of multimodal LLM, the encoder is an image encoder instead of a text encoder, but the same idea applies.

How does cross-attention work? Let's have a look at a conceptual drawing of what happens inside the regular self-attention mechanism.

<img src="/media/image-20241104105951.png" alt="20241104105951" style="zoom:60%;" />

In the figure above, x is the input, and _Wq_ is a weight matrix used to generate the queries (_Q_). Similarly, _K_ stands for keys, and _V_ stands for values. A represents the attention scores matrix, and _Z_ are the inputs (x) transformed into the output context vectors.

In cross-attention, in contrast to self-attention, we have two different input sources, as illustrated in the following figure.

<img src="/media/image-20241104110142.png" alt="20241104110142" style="zoom:60%;" />

As illustrated in the previous two figures, in self-attention, we work with the same input sequence. In cross-attention, we mix or combine two different input sequences. 

**In the case of the original transformer architecture in the _Attention Is All You Need_ paper, the two inputs _x1_ and _x2_ correspond to the sequence returned by the encoder module on the left (_x2_) and the input sequence being processed by the decoder part on the right (_x1_). In the context of a multimodal LLM, _x2_ is the output of an image encoder.** 

**Note that the queries usually come from the decoder, and the keys and values typically come from the encoder.**

Note that in cross-attention, the two input sequences _x1_ and _x2_ can have different numbers of elements. However, their embedding dimensions must match. If we set _x1 = x2_, this is equivalent to self-attention.

# Unified decoder and cross-attention model training

Now that we have talked a bit about the two major multimodal design choices, let's briefly talk about how we deal with the three major components during model training, which are summarized in the figure below.

<img src="/media/image-20241104111126.png" alt="20241104111126" style="zoom:60%;" />

Similar to the development of traditional text-only LLMs, the training of multimodal LLMs also involves two phases: pretraining and instruction finetuning. However, unlike starting from scratch, multimodal LLM training typically begins with a pretrained, instruction-finetuned text-only LLM as the base model.

For the image encoder, CLIP is commonly used and often remains unchanged during the entire training process, though there are exceptions, as we will explore later. Keeping the LLM part frozen during the pretraining phase is also usual, focusing only on training the projector—a linear layer or a small multi-layer perceptron. Given the projector's limited learning capacity, usually comprising just one or two layers, the LLM is often unfrozen during multimodal instruction finetuning (stage 2) to allow for more comprehensive updates. However, note that in the cross-attention-based models (Method B), the cross-attention layers are unfrozen throughout the entire training process.

After introducing the two primary approaches (Method A: Unified Embedding Decoder Architecture and Method B: Cross-modality Attention Architecture), you might be wondering which is more effective. The answer depends on specific trade-offs.

The Unified Embedding Decoder Architecture (Method A) is typically easier to implement since it doesn't require any modifications to the LLM architecture itself.

**The Cross-modality Attention Architecture (Method B) is often considered more computationally efficient because it doesn't overload the input context with additional image tokens, introducing them later in the cross-attention layers instead. Additionally, this approach maintains the text-only performance of the original LLM if the LLM parameters are kept frozen during training.**

We will revisit the discussion on modeling performance and response quality in a later section, where we will discuss NVIDIA's NVLM paper.

This marks the end of what turned out to be a rather extensive introduction to multimodal LLMs. As I write this, I realize that the discussion has become lengthier than initially planned, which probably makes this a good place to conclude the article. 

However, to provide a practical perspective, it would be nice to examine a few recent research papers that implement these approaches. So, we will explore these papers in the remaining sections of this article.


# Recent multimodal models and methods

## **The Llama 3 Herd of Models**

_[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)_ paper (July 31, 2024) by Meta AI came out earlier this summer, which feels like ages ago in LLM terms. However, given that they only described but did not release their multimodal models until much later, I think it's fair to include Llama 3 in this list. (Llama 3.2 models were officially announced and made available on September 25.)

The multimodal Llama 3.2 models, which come in an 11-billion and 90-billion parameter version, are image-text models that use the previously described cross-attention-based approach, which is illustrated in the figure below.


<img src="/media/image-20241104120105.png" alt="20241104120105" style="zoom:60%;" />

Note that while the figure also depicts video and speech as possible modalities, the models that were released as of this writing focus only on image and text.

Llama 3.2 uses the cross-attention-based approach. However, it differs a bit from what I wrote about earlier, namely that in multimodal LLM development, we usually freeze the image encoder and only update the LLM parameters during pretraining.

Here, the researchers almost take the opposite approach: they update the image encoder but do not update the language model's parameters. They write that this is intentional and done to preserve the text-only capabilities so that the 11B and 90B multimodal models can be used as drop-in replacements for the Llama 3.1 8B and 70B text-only model on text tasks.

The training itself is done in multiple iterations, starting with the Llama 3.1 text models. After adding the image encoder and projection (here called "adapter") layers, they pretrain the model on image-text data. Then, similar to the Llama 3 model text-only training (I wrote about it in [an earlier article](https://magazine.sebastianraschka.com/i/147749119/llama-overview)), they follow up with instruction and preference finetuning.

Instead of adopting a pretrained model such as CLIP as an image encoder, the researchers used a vision transformer that they pretrained from scratch. Specifically, they adopted the  ViT-H/14 variant (630 million parameters) of the classic vision transformer architecture ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)). They then pretrained the ViT on a dataset of 2.5 billion image-text pairs over five epochs; this was done before connecting the image encoder to the LLM. (The image encoder takes 224×224 resolution images and divides them into a 14×14 grid of patches, with each patch sized at 16×16 pixels.)

As the cross-attention layers add a substantial amount of parameters, **they are only added in every fourth transformer block. (For the 8B model, this adds 3B parameters, and for the 70B model, this adds 20 billion parameters.)**


## **Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models**

_[The Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](https://www.arxiv.org/abs/2409.17146)_ paper (September 25, 2024) is notable because it promises to open source not only the model weights but also the dataset and source code similar to the language-only OLMo LLM. (This is great for LLM research as it allows us to take a look at the exact training procedure and code and also lets us run ablation studies and reproduce results on the same dataset.)

If you are wondering why there are two names in the paper title, Molmo refers to the model (Multimodal Open Language Model), and PixMo (Pixels for Molmo) is the dataset.

<img src="/media/image-20241104121306.png" alt="20241104121306" style="zoom:60%;" />

As illustrated in the figure above, the image encoder employs an off-the-shelf vision transformer, specifically CLIP. The term "connector" here refers to a "projector" that aligns image features with the language model.

Molmo streamlines the training process by avoiding multiple pretraining stages, choosing instead a simple pipeline that updates all parameters in a unified approach—including those of the base LLM, the connector, and the image encoder.

The Molmo team offers several options for the base LLM:

- OLMo-7B-1024 (a fully open model backbone),
    
- OLMoE-1B-7B (a mixture-of-experts architecture; the most efficient model),
    
- Qwen2 7B (an open-weight model that performs better than OLMo-7B-1024),
    
- Qwen2 72B (an open-weight model and the best-performing model)


## **NVLM: Open Frontier-Class Multimodal LLMs**

NVIDIA's _[NVLM: Open Frontier-Class Multimodal LLMs](https://arxiv.org/abs/2409.11402)_ paper (September 17, 2024) is particularly interesting because, rather than focusing on a single approach, it explores both methods: 

- Method A, the Unified Embedding Decoder Architecture ("decoder-only architecture," NVLM-D), and 
    
- Method B, the Cross-Modality Attention Architecture ("cross-attention-based architecture," NVLM-X). 
    

Additionally, they develop a hybrid approach (NVLM-H) and provide an apples-to-apples comparison of all three methods.

<img src="/media/image-20241104121524.png" alt="20241104121524" style="zoom:60%;" />

As summarized in the figure below, NVLM-D corresponds to Method A, and NVLM-X corresponds to Method B, as discussed earlier. The concept behind the hybrid model (NVLM-H) is to combine the strengths of both methods: an image thumbnail is provided as input, followed by a dynamic number of patches passed through cross-attention to capture finer high-resolution details.

In short, the research team find that:

- NVLM-X demonstrates superior computational efficiency for high-resolution images.
    
- NVLM-D achieves higher accuracy in OCR-related tasks.
    
- NVLM-H combines the advantages of both methods.
    

Similar to Molmo and other approaches, they begin with a text-only LLM rather than pretraining a multimodal model from scratch (as this generally performs better). Additionally, they use an instruction-tuned LLM instead of a base LLM. Specifically, the backbone LLM is Qwen2-72B-Instruct (to my knowledge, Molmo used the Qwen2-72B base model).

While training all LLM parameters in the NVLM-D approach, they found that for NVLM-X, it works well to freeze the original LLM parameters and train only the cross-attention layers during both pretraining and instruction finetuning.

For the image encoder, instead of using a typical CLIP model, they use [InternViT-6B](https://arxiv.org/abs/2312.14238), which remains frozen throughout all stages.

The projector is a multilayer perceptron rather than a single linear layer.

## **Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution**

The previous two papers and models, Molmo and NVLM, were based on Qwen2-72B LLM. In this paper, the Qwen research team itself announces a multimodal LLM, _[Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)_ (October 3rd, 2024).

At the core of this work is their so-called "Naive Dynamic Resolution" mechanism (the term "naive" is intentional and not a typo for "native," though "native" could also be fitting). This mechanism allows the model to handle images of varying resolutions without simple downsampling, enabling the input of images in their original resolution.

<img src="/media/image-20241104122304.png" alt="20241104122304" style="zoom:60%;" />

The native resolution input is implemented via a modified ViT by removing the original absolute position embeddings and introducing 2D-RoPE.

They used a classic vision encoder with 675M parameters and LLM backbones of varying sizes, as shown in the table below.

<img src="/media/image-20241104122347.png" alt="20241104122347" style="zoom:60%;" />

The training itself consists of 3 stages: (1) pretraining only the image encoder, (2) unfreezing all parameters (including LLM), and (3) freezing the image encoder and instruction-finetuning only the LLM.

## **Pixtral 12B**

_[Pixtral 12B](https://mistral.ai/news/pixtral-12b/)_ (September 17, 2024), which uses the Method A: Unified Embedding Decoder Architecture approach, is the first multimodal model from Mistral AI. Unfortunately, there is no technical paper or report available, but the Mistral team shared a few interesting tidbits in their [blog post](https://mistral.ai/news/pixtral-12b/).

Interestingly, they chose not to use a pretrained image encoder, instead training one with 400 million parameters from scratch. For the LLM backbone, they used the 12-billion-parameter [Mistral NeMo](https://mistral.ai/news/mistral-nemo/) model.

Similar to Qwen2-VL, Pixtral also supports variable image sizes natively, as illustrated in the figure below.

<img src="/media/image-20241104122443.png" alt="20241104122443" style="zoom:60%;" />

# Conclusion

As you may have noticed, I almost entirely skipped both the modeling and the computational performance comparisons. First, comparing the performance of LLMs and multimodal LLMs on public benchmarks is challenging due to prevalent data contamination, meaning that the test data may have been included in the training data.

Additionally, the architectural components vary so much that making an apples-to-apples comparison is difficult. So, big kudos to the NVIDIA team for developing NVLM in different flavors, which allowed for a comparison between the decoder-only and cross-attention approaches at least.

In any case, the main takeaway from this article is that multimodal LLMs can be built successfully in many different ways. Below is a figure that summarizes the different components of the models covered in this article.


<img src="/media/image-20241104122540.png" alt="20241104122540" style="zoom:60%;" />

