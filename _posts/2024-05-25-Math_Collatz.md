---
title: 考拉兹猜想
date: 2024-05-25 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
description: 採樣的常見性和重要性不言而喻
typora-root-url: ../../allenlu2009.github.io


---



## Source

[78分钟无禁忌全方位讨论考拉兹猜想｜3N+1问题｜角谷猜想｜陶哲轩 (youtube.com)](https://www.youtube.com/watch?v=FkF13JnkBfU)



Binary:

Even :  n/2   平均 x 0.5

Odd:  3n+1 (even/2)  平均 x 1.5



使用二進位非常有意思。

$n = 2^k = 1000...00$  to $1$

所有的偶數的尾部 000, 都可以去掉，之剩下尾數 1,  也就是奇數。

一般偶數基本把尾部 0 去掉。 11110  : remove 0。  0.5 倍。

3n +1 = (2+1) n + 1 = (2n + 1) + n 

2n + 1 就是 left-shift append with 1,  再加上原數 n.

$ n = 1011$  to $10111 + 1011 = 100010$ to $10001$  比兩倍小。約為 1.5 倍。



如果奇數和偶數機率一樣，簡單的倍率是 x1.5 x0.5 = 0.75 所以每次的倍率基本是 0.75 變小。

因此趨勢是越來越小。
