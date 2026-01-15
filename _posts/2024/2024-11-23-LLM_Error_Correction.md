---
title: How to Improve LLM Error Resillence
date: 2024-11-23 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1117-Causal_Attention_Kernel]]"
categories:
  - LLM
tags:
  - AI
  - Attention
  - Causal
  - Cumsum
  - Kernel
  - LLM
  - Transformer
---



## Introduction

There are several cases where large language models (LLMs) may encounter errors:

1. **Input errors**: These include typos, scrambled encoding, or incorrect sentence breaks, which can lead to misinterpretations by the model.
2. **Yield issues**: In efforts to improve chip yield, increasing the number of dies can introduce error-prone components, impacting model performance.
3. **Communication issues**: Errors may arise during distributed processing, such as the loss of intermediate results in all-reduce layers.

Given that machine learning inherently relies on probabilistic methods, and network weights are optimized based on data and with built-in redundacy, there are opportunities to design neural networks that can tolerate errors—similar to the human brain. For instance, individuals with significant brain injuries, or even those with only half a brain, can still perform intact functions due to the brain's adaptability and redundancy. This inspires potential approaches to make neural networks robust against errors while maintaining functionality.



Here’s a summarized table of the three errors, their characteristics, and potential methods to address them:

|**Error Type**|**Characteristics**|**Potential Methods to Fix Errors**|
|---|---|---|
|**Input Errors**|Typos, scrambled encoding, incorrect sentence breaks.|- Preprocessing steps like spell-checking and encoding standardization.|
|||- Implementing robust tokenization algorithms to handle inconsistencies.|
|||- Use context-aware models to infer correct meanings despite input issues.|
|**Yield Issues**|Error-prone components due to increased chip yield (e.g., manufacturing tolerances).|- Employ error-correcting codes (ECC) to mitigate hardware-induced errors.|
|||- Introduce fault-tolerant architectures or redundant components.|
|||- Optimize chip design for reliability while balancing performance and yield.|
|**Communication Issues**|Loss of intermediate results during distributed processing (e.g., all-reduce layers).|- Use checkpointing techniques to save partial computations and resume from failures.|
|||- Improve communication protocols to ensure data integrity and minimize packet loss.|
|||- Leverage asynchronous training techniques to handle partial updates without significant performance degradation.|

This table captures the core ideas in a concise format, making it easy to reference. Let me know if you’d like further refinements!


## Other Errors (基本是 training 就有問題)

我們想解決的問題是 training 沒有問題，但是 inferecing 遇到問題。一是影響有多大，二是可否在 training 幫忙多做一些事，或是 fine-tuning 幫忙解決？

Here's a table summarizing the additional error types, their characteristics, and potential methods to fix them:

| **Error Type**                                                   | **Characteristics**                                                                            | **Potential Fixes**                                       |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Overfitting**<br>too few data, training                        | Model performs well on training data but poorly on unseen data due to memorization.            | - Regularization (e.g., dropout, L2)                      |
|                                                                  |                                                                                                | - Increase data diversity                                 |
|                                                                  |                                                                                                | - Use data augmentation                                   |
| **Underfitting** <br>too small model, training                   | Model fails to capture patterns in data, leading to poor training set performance.             | - Increase model complexity (e.g., layers)                |
|                                                                  |                                                                                                | - Improve feature engineering                             |
|                                                                  |                                                                                                | - Extend training time                                    |
| **Data Bias/Imbalance**<br>Data, training                        | Skewed predictions due to unrepresentative training data.                                      | - Collect diverse and balanced datasets                   |
|                                                                  |                                                                                                | - Use fairness-aware training                             |
|                                                                  |                                                                                                | - Post-process predictions                                |
| **Catastrophic Forgetting**<br>training                          | Loss of performance on old tasks during fine-tuning for new tasks.                             | - Use Elastic Weight Consolidation (EWC)                  |
|                                                                  |                                                                                                | - Employ regularization                                   |
|                                                                  |                                                                                                | - Use rehearsal methods                                   |
| **Gradient Issues**<br>Training                                  | Gradients become too large (explosion) or too small (vanishing), hindering effective training. | - Gradient clipping for large gradients                   |
|                                                                  |                                                                                                | - Careful weight initialization                           |
|                                                                  |                                                                                                | - Use suitable activation functions (e.g., ReLU variants) |
| **Hallucination**<br>Data, training                              | Model generates plausible but incorrect or nonsensical information.                            | - Integrate retrieval-augmented generation (RAG)          |
|                                                                  |                                                                                                | - Fine-tune on specific datasets                          |
|                                                                  |                                                                                                | - Verification steps or hybrid systems                    |
| **Numerical Instabilities**<br>Training, and Inferencing (quant) | Errors like NaNs or infinities due to extreme computational values.                            | - Use numerically stable algorithms                       |
| Label smoothing                                                  |                                                                                                | - Normalize input data                                    |
| Z-loss, overflow, underflow                                      |                                                                                                | - Monitor gradients and weights                           |

## Parallelsim

Data parallellism (batches)
Tensor parallellism (heads)
Pipline parallellism (layers)

## Some Solution

Training:  dropout training
Fine-tuning:  ? How ? use on-device LoRA per chip to fix errors?
Topology:  massive connections:  L to L+1 and L+2 layers



### Error Resilient LLM (Use Ai2 OpenScholar-8B)

Large language models (LLMs) have shown remarkable capabilities in various natural language processing (NLP) tasks, but their resilience to errors and noise in input text is a critical concern. Recent studies have investigated the ability of LLMs to handle different types of errors, including automatic speech recognition (ASR) errors, OCR errors, grammatical mistakes, typographical errors, and distractive content (Wang et al. 2024).
大型語言模型 （LLMs） 在各種自然語言處理 （NLP） 任務中表現出卓越的能力，但它們對輸入文本中的錯誤和雜訊的彈性是一個關鍵問題。最近的研究調查了 LLMs 處理不同類型錯誤的能力，包括自動語音識別 （ASR） 錯誤、OCR 錯誤、語法錯誤、印刷錯誤和分散注意力的內容（Wang 等人，2024 年）。

One study found that while some LLMs show a degree of resistance to certain types of noise, their overall performance significantly suffers when faced with these errors (Wang et al. 2024). Another study proposed a "re-pass" strategy to purify the instructions of noise before the LLMs process them, which showed promise in improving model performance (Wang et al. 2024).

What is re-pass:  This approach involves a two-step process: initially, we employ an LLM to conduct zero-shot text normalization to purify the noisy instructions. Next, we prompt the model to process upon the cleaned instruction. Our findings reveal that not all models are adept at fulfilling this role of data normalization. The exception is ChatGPT, which demonstrates a comprehensive understanding of the text and can recover the instruction with different types of noises.

一項研究發現，雖然一些 <代碼id=g1001>LLM 對某些類型的雜訊表現出一定程度的抵抗力，但在面對這些錯誤時，它們的整體性能會受到嚴重影響（Wang等人，2024年）。另一項研究提出了一種 「re-pass」 策略，在 LLMs，這在提高模型性能方面顯示出希望（Wang et al. 2024）。

什麼是 re-pass?這種方法涉及兩個步驟的過程：首先，我們使用大型語言模型（LLM）進行zero-shot文本規範化，以淨化雜訊指令。接著，我們提示模型根據清理過的指令進行處理。我們的研究結果顯示，並非所有模型都擅長執行這一數據規範化的角色。例外的是ChatGPT，它對文本具有全面的理解能力，並能夠在不同類型的噪聲中恢復指令。

A novel experimental setup, the Scrambled Bench, was designed to measure the capacity of LLMs to handle scrambled input, including recovering scrambled sentences and answering questions given scrambled context (Cao et al. 2023). The results showed that most powerful LLMs, such as GPT-4, demonstrate the capability to understand the meaning of words even when the letters within those words are scrambled, as long as the first and last letters remain in place (Cao et al. 2023).
一種新穎的實驗設置，即 Scrambled Bench，旨在測量 LLMs 處理亂序輸入的能力，包括恢復亂序句子和在給定亂序上下文中回答問題（Cao 等人，2023 年）。結果表明，最強大的 LLMs，例如 GPT-4，只要第一個和最後一個字母保持不變，即使這些單詞中的字母被打亂，也能理解單詞的含義（Cao 等人，2023 年）。

A framework called Resilient Guardrails for Large Language Models (RigorLLM) was introduced to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs (Yuan et al. 2024). RigorLLM employs a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on the data augmentation (Yuan et al. 2024). The experimental evaluations demonstrated that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks (Yuan et al. 2024).
引入了一個名為大型語言模型的彈性護欄 （RigorLLM） 的框架，以有效地調節 LLMs（Yuan 等人，2024 年）。RigorLLM 採用多方面的方法，包括通過 Langevin 動力學進行基於能量的訓練數據增強，通過最小極大優化優化輸入的安全後綴，以及集成基於融合的模型，該模型結合了穩健的 KNN 和 <代碼 id=g1002>LLM（Yuan 等人，2024 年）.實驗評估表明，RigorLLM 不僅在檢測有害內容方面優於 OpenAI API 和 Perspective API 等現有基線，而且還表現出無與倫比的越獄攻擊彈性（Yuan 等人，2024 年）。

A study investigated the resilience of LLMs against morphological variations in text by artificially introducing varying levels of noise into a diverse set of datasets and systematically evaluating LLMs' robustness against the corrupt variations of the original text (Singh et al. 2024). The findings showed that generative LLMs are quiet robust to noisy perturbations in text, which is a departure from pre-trained models like BERT or RoBERTa whose performance has been shown to be sensitive to deteriorating noisy text (Singh et al. 2024).
一項研究通過人為地將不同級別的雜訊引入不同的數據集並系統評估 LLMs（Singh 等人，2024 年）。研究結果表明，生成<代碼 id=g1003>LLM 對文本中的嘈雜擾動具有安靜的魯棒性，這與 BERT 或 RoBERTa 等預訓練模型不同，其性能已被證明對惡化的嘈雜文本很敏感（Singh 等人，2024 年）。

The ability of LLMs to act as speech recognition post-processors was explored, and the results showed that rescoring only by in-context learning with frozen LLMs achieves results that are competitive with rescoring by domain-tuned LMs (Yang et al. 2023). Additionally, a task-activating prompting method was evaluated, which combines causal instructions and demonstration to increase the context window (Yang et al. 2023).
探索了 <代碼id=g1001>LLM充當語音辨識後處理器的能力，結果表明，僅使用凍結的 <代碼id=g1002>LLM通過上下文學習進行重新評分，其結果與通過域調整的 LM 進行重新評分具有競爭力（Yang 等人，2023年）.此外，還評估了一種任務啟動提示方法，該方法結合了因果指令和演示以增加上下文視窗（Yang 等人，2023 年）。

Large-scale language models have shown remarkable capability in various NLP tasks, but their performance on automatic evaluation metrics falls short of the previous state-of-the-art models in English grammatical error correction (GEC) tasks (Qu et al. 2023). A study explored how large language models perform on Chinese GEC tasks and found that the performances of LLMs on automatic evaluation metrics fall short of the previous state-of-the-art models because of the problem of over-correction (Qu et al. 2023). Furthermore, notable variations in the performance of LLMs were discovered when evaluated on different data distributions (Qu et al. 2023).
大規模語言模型在各種 NLP 任務中表現出卓越的能力，但它們在自動評估指標上的表現不如以前在英語語法糾錯 （GEC） 任務中最先進的模型（Qu et al. 2023）。一項研究探討了大型語言模型在中文 GEC 任務上的表現，發現由於過度糾正的問題，LLMs 在自動評估指標上的性能不如以前最先進的模型（Qu et al. 2023）。此外，在不同的數據分佈上進行評估時，發現 LLMs（Qu 等人，2023 年）。

A method was introduced to dramatically reduce fine-tuning VRAM requirements and rectify quantization errors in quantized LLMs (Chai et al. 2023). The method, called extremely memory-efficient fine-tuning (EMEF), uses Low-Rank Adaptation (LoRA) to reduce the memory requirements by up to 5.6 times, which enables fine-tuning a 7 billion parameter LLM on consumer laptops (Chai et al. 2023). The method also proposes a Low-Rank Error Correction (LREC) method that exploits the added LoRA layers to ameliorate the gap between the quantized model and its float point counterpart, leading to a fully functional INT2 quantized LLM with the capacity to generate coherent English text (Chai et al. 2023).
引入了一種方法來顯著降低微調 VRAM 要求並糾正量化 <代碼 id=g1001>LLM 中的量化錯誤 （Chai et al. 2023）。該方法稱為極其節省記憶體的微調 （EMEF），使用低秩自適應 （LoRA） 將記憶體需求降低多達 5.6 倍，從而能夠在消費類筆記型電腦上微調 70 億個參數<代碼 id=g1002>LLM）（Chai 等人，2023 年）。該方法還提出了一種低秩糾錯 （LREC） 方法，該方法利用添加的LoRA層來縮小量化模型與其浮點對應物之間的差距，從而產生功能齊全的 INT2 量化 <代碼 id=g1003>LLM，能夠生成連貫的英文文本（Chai 等人，2023 年）。

An efficient error detection technique called Concurrent Linguistic Error Detection (CLED) was proposed for LLMs that does not require access to the model's internal nodes (Zhu et al. 2024). CLED exploits the rules of the language and the patterns of valid texts to detect texts generated by language models when they suffer errors (Zhu et al. 2024). The proposed scheme was evaluated for two different models and two different tasks, showing that most errors, close to 90%, are detected at a very low recomputation overhead (of 1%) (Zhu et al. 2024). The results also showed that the proposed CLED can trade off detection rate and overhead by just changing the decision threshold of the concurrent classifier, enabling a dynamic adjustment of the protection to the application or input text at run time (Zhu et al. 2024).
針對不需要訪問模型內部節點的 <代碼 id=g1001>LLM）提出了一種稱為併發語言錯誤檢測 （CLED） 的有效錯誤檢測技術（Zhu et al. 2024）。CLED 利用語言規則和有效文本的模式來檢測語言模型在出現錯誤時生成的文本（Zhu et al. 2024）。針對兩種不同的模型和兩種不同的任務對所提出的方案進行了評估，結果表明，大多數錯誤（接近90%）是在非常低的重新計算開銷（1%）下檢測到的（Zhu et al. 2024）。結果還表明，所提出的 CLED 可以通過改變併發分類器的決策閾值來權衡檢測率和開銷，從而可以在運行時動態調整對應用程式或輸入文本的保護（Zhu et al. 2024）。

In conclusion, LLMs demonstrate resilience to errors and noise in input text, but their performance significantly suffers when faced with certain types of errors (Wang et al. 2024). Various strategies have been proposed to improve model resilience, including the "re-pass" strategy (Wang et al. 2024), the Scrambled Bench (Cao et al. 2023), RigorLLM (Yuan et al. 2024), and error detection techniques like CLED (Zhu et al. 2024). These studies highlight the importance of further investigation into enhancing model resilience and developing more secure and reliable LLMs (Wang et al. 2024).
總之，LLMs 表現出對輸入文本中錯誤和雜訊的彈性，但在面對某些類型的錯誤時，它們的性能會受到嚴重影響（Wang et al. 2024）。已經提出了各種策略來提高模型彈性，包括 “re-pass” 策略 （Wang et al. 2024）、Scrambled Bench （Cao et al. 2023）、RigorLLM （Yuan et al. 2024）以及錯誤檢測技術，如 CLED （Zhu et al. 2024） 中。這些研究強調了進一步研究增強模型彈性和開發更安全可靠的 <代碼id=g1002>LLM的重要性（Wang等人，2024年）。

### References

- [Yuji Chai](https://www.semanticscholar.org/author/148385111), [John Gkountouras](https://www.semanticscholar.org/author/2214279295), [Glenn G. Ko](https://www.semanticscholar.org/author/2054461), [D. Brooks](https://www.semanticscholar.org/author/1896817), [Gu-Yeon Wei](https://www.semanticscholar.org/author/2113825170). [INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation](https://www.semanticscholar.org/p/259165162). 2023. arXiv.org.  
- [Fanyi Qu](https://www.semanticscholar.org/author/2127464628), [Yunfang Wu](https://www.semanticscholar.org/author/2115377800). [Evaluating the Capability of Large-scale Language Models on Chinese Grammatical Error Correction Task](https://www.semanticscholar.org/p/259501459). 2023. arXiv.org. 
- [Chao-Han Huck Yang](https://www.semanticscholar.org/author/46962482), [Yile Gu](https://www.semanticscholar.org/author/2112579196), [Yi-Chieh Liu](https://www.semanticscholar.org/author/51224607), [Shalini Ghosh](https://www.semanticscholar.org/author/2247854926), [I. Bulyko](https://www.semanticscholar.org/author/2575459), [A. Stolcke](https://www.semanticscholar.org/author/1762744). [Generative Speech Recognition Error Correction With Large Language Models and Task-Activating Prompting](https://www.semanticscholar.org/p/263152585). 2023. Automatic Speech Recognition & Understanding.  
- [Qi Cao](https://www.semanticscholar.org/author/2268816164), [Takeshi Kojima](https://www.semanticscholar.org/author/2081836120), [Yutaka Matsuo](https://www.semanticscholar.org/author/2241471533), [Yusuke Iwasawa](https://www.semanticscholar.org/author/1715282). [Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Unnatural Scrambled Text](https://www.semanticscholar.org/p/265506770). 2023. Conference on Empirical Methods in Natural Language Processing.  
- [Zhuowen Yuan](https://www.semanticscholar.org/author/2269690464), [Zidi Xiong](https://www.semanticscholar.org/author/2155965725), [Yi Zeng](https://www.semanticscholar.org/author/2273035656), [Ning Yu](https://www.semanticscholar.org/author/2282538184), [Ruoxi Jia](https://www.semanticscholar.org/author/2291965197), [D. Song](https://www.semanticscholar.org/author/2242706269) et al. [RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content](https://www.semanticscholar.org/p/268536710). 2024. International Conference on Machine Learning.  
- [Jinhua Zhu](https://www.semanticscholar.org/author/2146271833), [Javier Conde](https://www.semanticscholar.org/author/2230852635), [Zhen Gao](https://www.semanticscholar.org/author/2243685502), [Pedro Reviriego](https://www.semanticscholar.org/author/2243081043), [Shanshan Liu](https://www.semanticscholar.org/author/48641583), [Fabrizio Lombardi](https://www.semanticscholar.org/author/2060432692). [Concurrent Linguistic Error Detection (CLED) for Large Language Models](https://www.semanticscholar.org/p/268680321). 2024. arXiv.org.  
- [Bin Wang](https://www.semanticscholar.org/author/2256857690), [Chengwei Wei](https://www.semanticscholar.org/author/1780581845), [Zhengyuan Liu](https://www.semanticscholar.org/author/2223323387), [Geyu Lin](https://www.semanticscholar.org/author/2297433621), [Nancy F. Chen](https://www.semanticscholar.org/author/2271409954). [Resilience of Large Language Models for Noisy Instructions](https://www.semanticscholar.org/p/269148809). 2024. Conference on Empirical Methods in Natural Language Processing.  
- [Ayush Singh](https://www.semanticscholar.org/author/2303795047), [Navpreet Singh](https://www.semanticscholar.org/author/2311196517), [Shubham Vatsal](https://www.semanticscholar.org/author/1491705917). [Robustness of LLMs to Perturbations in Text](https://www.semanticscholar.org/p/271162255). 2024. arXiv.org.




