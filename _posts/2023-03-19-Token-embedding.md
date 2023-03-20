---
title: Token and Embedding, Query-Key-Value
date: 2023-03-19 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---

## Reference

Cohere 在 natural language processing 出了幾篇介紹的文章都非常有幫助。

[@serranoWhatAre2023]  :  excellent introduction of embedding

[@serranoWhatSemantic2023] : excellent introduction of semantic search



## Introduction

NLP 最有影響力的幾個觀念。之後也被其他領域使用。例如 computer vision, ....



Embedding:! starting from word2vec paper.

QKV: query, key, value 這似乎最早是從 database 或是 recommendation system 而來。從 transformer paper 開始聲名大噪，廣汎用於 NLP, vision, voice, etc.

attention: 特別是 self-attention,  然後才是 cross-attention.  雖然在 RNN 已經有 attention 的觀念，但是到 transformer 才發揚光大。 RNN 其實已經有時序的 attention, 在加上 attention 似乎有點畫蛇添足。Transformer attention 是 spatial attention, 並且範圍更大。







## Reference

Serrano, Luis. “What Are Word and Sentence Embeddings?” Context by Cohere. January 18, 2023.

<https://txt.cohere.ai/sentence-word-embeddings/>.



———. “What Is Semantic Search?” Context by Cohere. February 27, 2023.

<https://txt.cohere.ai/what-is-semantic-search/>.

