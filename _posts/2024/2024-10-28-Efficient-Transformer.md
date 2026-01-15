---
title: Efficient (Still) Transformer
date: 2024-10-28 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-08-14-LLM_Adapting]]"
categories:
  - LLM
tags:
  - LLM
  - Transformer
  - prompt
---




## Source

https://arxiv.org/pdf/2404.14294 A Survey on Efficient Inference for Large Language Models

https://arxiv.org/pdf/2006.16236 Efficient Transformers: A Survey 2022


https://arxiv.org/pdf/2112.05682: memory efficient attention for inferencing!



# Title: 2024-1028-Efficient-Transformer

## Overview
The document presents an overview of advancements in efficient transformer architectures, particularly focusing on methods and techniques aimed at improving inference efficiency for large language models. The survey synthesizes current research, methodologies, and applications related to efficient transformers.

## Key Topics

1. **Introduction to Transformers**
   - Understanding the foundational architecture of transformers.
   - Significance of transformers in natural language processing.

2. **Challenges with Large Language Models**
   - High computational costs associated with inference.
   - Memory limitations when deploying large models.

3. **Efficient Transformer Architectures**
   - Overview of various architectures designed for efficiency.
   - Comparison of different approaches and their trade-offs.

4. **Memory Efficient Attention Mechanisms**
   - Techniques proposed in the literature to reduce memory usage during attention computations.
   - Discussion on specific methods like sparse attention, low-rank approximations, and kernel-based approaches.

5. **Quantization and Pruning Techniques**
   - Methods for reducing model size and increasing inference speed without significantly impacting performance.
   - Strategies for quantizing weights and pruning less important connections in the model.

6. **Hardware Acceleration**
   - The role of specialized hardware (e.g., TPUs, GPUs) in enhancing transformer efficiency.
   - Frameworks that enable optimized execution on hardware platforms.

7. **Applications and Use Cases**
   - Practical implementations of efficient transformers in real-world scenarios.
   - Examples from various domains such as healthcare, finance, and customer service.

8. **Future Directions**
   - Emerging trends in transformer research focusing on both theoretical advancements and practical applications.
   - Potential areas for further exploration to improve efficiency further.

## References

1. [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/pdf/2404.14294)  
2. [Efficient Transformers: A Survey 2022](https://arxiv.org/pdf/2006.16236)  
3. [Memory Efficient Attention for Inferencing](https://arxiv.org/pdf/2112.05682)  

## Conclusion
The survey highlights significant progress made toward achieving efficient inference with large language models through innovative architectural designs, memory management techniques, and hardware optimizations. As these methods continue to evolve, they will play a critical role in making advanced AI systems more accessible and deployable across various applications.

--- 

This summary captures the essential points discussed within the context of efficient transformers while directing readers to pertinent resources for deeper insights into each topic mentioned.

## Transformer Attention 比較表 
| Stage                                     | Memory Requirement                              | Memory Bandwidth                    | Computation Requirement                   | Primary Bottleneck                                              | RNN                                |
| ----------------------------------------- | ----------------------------------------------- | ----------------------------------- | ----------------------------------------- | --------------------------------------------------------------- | ---------------------------------- |
| **Training**                              | Attn matrix with $L^2$                          | Weight only, shared by token length | Attn matrix grows with $L^2$              | **Computation-bound** (heavy matrix multiplications), 但可平行處理加速！ | Grows with $L$， Recursive, 無法平行加速！ |
| **Inference Prefill (one-shot training)** | same as above, 但可分段 to build causal attn matrix | same as above                       | Attn matrix grows with $L^2$, impact TTFT | **Computation-bound**, but **Memory-bound** for long prompts    | Grows with $L$                     |
| **Inference Generation (AR)**             | KV cache grows with $L$                         | Weight + KV cache for every token!  | Attn matrix grows with $L$                | **Memory bandwidth-bound** due to KV cache and weights          | Constant                           |



<img src="/media/image-20241031120636.png" alt="20241031120636" style="zoom:60%;" />

<img src="/media/image-20241027201735.png" alt="20241027201735" style="zoom:60%;" />


<img src="/media/image-20241027201707.png" alt="20241027201707" style="zoom:60%;" />


## Efficient Transformer


<img src="/media/image-20241028142313.png" alt="20241028142313" style="zoom:60%;" />


<img src="/media/image-20241028142344.png" alt="20241028142344" style="zoom:60%;" />