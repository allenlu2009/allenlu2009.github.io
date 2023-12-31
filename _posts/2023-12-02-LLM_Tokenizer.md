---
title: LLM Tokenizer
date: 2023-12-02 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
description: LLM Tokenizer
typora-root-url: ../../allenlu2009.github.io


---





## Source

* https://juejin.cn/post/7234795667477561402

  

## Takeaway



### Tokenizer

1. character level encoder:  codebook 65
2. BPE (byte-pair encoder)
   * GPT2-3:   codebook 50541?
   * Tiktoken (OpenAI)

基本是 trade-off of the codebook vs. the token length!

[Hii, hello world]:  character tokenizer: 12 tokens;  BPE:  3 tokens

@guodongLLMTokenizer2023

## 背景

随着ChatGPT迅速出圈，最近几个月开源的大模型也是遍地开花。目前，开源的大语言模型主要有三大类：ChatGLM衍生的大模型（wenda、[ChatSQL](https://link.juejin.cn?target=https%3A%2F%2Fgithub.com%2Fyysirs%2FChatSQL)等）、LLaMA衍生的大模型（Alpaca、Vicuna、BELLE、Phoenix、Chimera等）、Bloom衍生的大模型（Bloomz、BELLE、Phoenix等）。其中，ChatGLM-6B主要以中英双语进行训练，LLaMA主要以英语为主要语言的拉丁语系进行训练，而Bloom使用了46种自然语言、13种编程语言进行训练。

| 模型       | 训练数据量                                           | 模型参数  | 训练数据范围               | 词表大小 | 分词算法 | 分词器（Tokenizer）后端                       |
| ---------- | ---------------------------------------------------- | --------- | -------------------------- | -------- | -------- | --------------------------------------------- |
| LLaMA      | 1T～1.4T tokens(其中，7B/13B使用1T，33B/65B使用1.4T) | 7B～65B   | 以英语为主要语言的拉丁语系 | 32000    | BBPE     | 基于SentencePiece工具实现                     |
| ChatGLM-6B | 约 1T tokens                                         | 6B        | 中英双语                   | 130528   | BBPE     | 基于SentencePiece工具实现                     |
| Bloom      | 1.6TB预处理文本，转换为 350B 唯一 tokens             | 300M~176B | 46种自然语言，13种编程语言 | 250680   | BBPE     | HuggingFace 的 tokenizers （类SentencePiece） |

目前来看，在开源大模型中，LLaMA无疑是其中最闪亮的星。但是，与ChatGLM-6B和Bloom原生支持中文不同。LLaMA 原生仅支持 Latin 或 Cyrillic 语系，对于中文支持不是特别理想。原版LLaMA模型的词表大小是32K，而多语言模型（如：XLM-R、Bloom）的词表大小约为250K。以中文为例，LLaMA词表中的中文token比较少（只有几百个）。这将导致了两个问题：

- LLaMA 原生tokenizer词表中仅包含少量中文字符，在对中文字进行tokenzation时，一个中文汉字往往被切分成多个token（2-3个Token才能组合成一个汉字），显著降低编解码的效率。
- 预训练中没有出现过或者出现得很少的语言学习得不充分。

为了解决这些问题，我们可能就需要进行中文词表扩展。比如：在中文语料库上训练一个中文tokenizer模型，然后将中文 tokenizer 与 LLaMA 原生的 tokenizer 进行合并，通过组合它们的词汇表，最终获得一个合并后的 tokenizer 模型。

本文将介绍使用`SentencePiece`工具如何使用中文语料训练一个分词模型。

## 预备知识

讲解 SentencePiece 之前，我们先讲解下分词器（Tokenizer）。

那什么是分词器？简单点说就是将字符序列转化为数字序列，对应模型的输入。

通常情况下，Tokenizer有三种粒度：word/char/subword

- word: 按照词进行分词，如: `Today is sunday`. 则根据空格或标点进行分割`[today, is, sunday, .]`
- character：按照单字符进行分词，就是以char为最小粒度。 如：`Today is sunday.` 则会分割成`[t， o， d，a，y， .... ，s，u，n，d，a，y， .]`
- subword：按照词的subword进行分词。如：`Today is sunday.` 则会分割成`[to， day，is ， s，un，day， .]`

可以看到这三种粒度分词截然不同，各有利弊。

对于word粒度分词：

- 优点：词的边界和含义得到保留；
- 缺点：1）词表大，稀有词学不好；2）OOV（可能超出词表外的词）；3）无法处理单词形态关系和词缀关系，会将两个本身意思一致的词分成两个毫不相同的ID，在英文中尤为明显，如：cat， cats。

对于character粒度分词：

- 优点：词表极小，比如：26个英文字母几乎可以组合出所有词，5000多个中文常用字基本也能组合出足够的词汇；
- 缺点：1）无法承载丰富的语义，英文中尤为明显，但中文却是较为合理，中文中用此种方式较多。2）序列长度大幅增长；

最后为了平衡以上两种方法， 又提出了基于 subword 进行分词：它可以较好的平衡词表大小与语义表达能力；常见的子词算法有Byte-Pair Encoding (BPE) / Byte-level BPE（BBPE）、Unigram LM、WordPiece、SentencePiece等。

- BPE：即字节对编码。其核心思想是从字母开始，不断找词频最高、且连续的两个token合并，直到达到目标词数。
- BBPE：BBPE核心思想将BPE的从字符级别扩展到子节（Byte）级别。BPE的一个问题是如果遇到了unicode编码，基本字符集可能会很大。BBPE就是以一个字节为一种“字符”，不管实际字符集用了几个字节来表示一个字符。这样的话，基础字符集的大小就锁定在了256（2^8）。采用BBPE的好处是可以跨语言共用词表，显著压缩词表的大小。而坏处就是，对于类似中文这样的语言，一段文字的序列长度会显著增长。因此，BBPE based模型可能比BPE based模型表现的更好。然而，BBPE sequence比起BPE来说略长，这也导致了更长的训练/推理时间。BBPE其实与BPE在实现上并无大的不同，只不过基础词表使用256的字节集。
- WordPiece：WordPiece算法可以看作是BPE的变种。不同的是，WordPiece基于概率生成新的subword而不是下一最高频字节对。WordPiece算法也是每次从词表中选出两个子词合并成新的子词。BPE选择频数最高的相邻子词合并，而WordPiece选择使得语言模型概率最大的相邻子词加入词表。
- Unigram：它和 BPE 以及 WordPiece 从表面上看一个大的不同是，前两者都是初始化一个小词表，然后一个个增加到限定的词汇量，而 Unigram Language Model 却是先初始一个大词表，接着通过语言模型评估不断减少词表，直到限定词汇量。
- SentencePiece：SentencePiece它是谷歌推出的子词开源工具包，它是把一个句子看作一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把空格也当作一种特殊字符来处理，再用BPE或者Unigram算法来构造词汇表。SentencePiece除了集成了BPE、ULM子词算法之外，SentencePiece还能支持字符和词级别的分词。

下图是一些主流模型使用的分词算法，比如：GPT-1 使用的BPE实现分词，LLaMA/BLOOM/GPT2/ChatGLM使用BBPE实现分词。BERT/DistilBERT/Electra使用WordPiece进行分词，XLNet则采用了SentencePiece进行分词。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c7c54cee78754cda9c4cebaf4f82dc43~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp?)

从上面的表格中我们也可以看到当前主流的一些开源大模型有很多基于 BBPE 算法使用 SentencePiece 实现



## Appendix

