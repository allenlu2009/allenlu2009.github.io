---
title: Mixed Language Output
date: 2023-03-05 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---

## Reference





## Introduction

混合語言如中英混合，中日混合，英法混合都算是常見的例子。

最近 LLM (Large Language Model) 火紅，對於混合語言的支持如何？

因爲 LLM 從網絡抓下來的資料，混合語言只會有極小的部分。**對於 training mixed language as embedding 有幫助。**

**所以 LLM 一般可以理解輸入混合語言，只要一些 mixed language for trailing.**

**但是輸出混合語言基本不可能，因爲 predict next word 基本都是用機率最大的字。混合語言出現的機率非常小。**

這裏有幾個問題或應用：

* 需要解決混合語言輸入問題:  混合語言輸入法一直是麻煩的問題。如果是鍵盤需要切換輸入法 (中英文)，如果是語音輸入需要自動偵測不同語言。因爲資料量不足，常常準確率不好 (用 synthesized data?)。即使是已經輸入的混合語言
  * keyboard
  * Voice
  * NLP with mixed language
*   output
  * 專有名詞
  * Personalization





解法 (之一) : 把混合語言視爲 translation 問題





但在某些情況似乎還是需要混合語言輸出。

