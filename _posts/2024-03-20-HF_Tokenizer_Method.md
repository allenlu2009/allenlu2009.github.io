---
title: HuggingFace Tokenizer Function
date: 2024-03-20 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
description: LLM Tokenizer
typora-root-url: ../../allenlu2009.github.io



---





## Source

* https://www.cnblogs.com/carolsun/p/16903276.html




## 開場

HuggingFace 的 transformers 大概是所有做大語言模型都會使用 API。大語言模型分成兩個部分：tokenizer 和  LLM (transformer network).  兩者基本是完全獨立的模組。





![image-20240221222453811](/../../../../OneDrive/allenlu2009.github.io/media/image-20240221222453811-0919699.png)

## [transformer 中 tokenizer 的那些事](https://www.cnblogs.com/carolsun/p/16903276.html)

Tokenizer 基本做兩件事：分詞和編碼。 Huggingface中的[tokenizers](https://huggingface.co/docs/tokenizers/index) 进行文本分词，其中有很多函数，`tokenizer.tokenize`、`tokenizer,convert_tokens_to_ids`、`tokenizer.encode`、`tokenizer`、`tokenizer.encode_plus`、`tokenizer.pad` 在使用的时候经常会傻傻分不清楚，希望在这里对常用到的函数进行说明。

```python
# 导入
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
```

我们使用tokenizer的目的分为两种，一个是分词，一个是编码。输入的内容分为两种，一种是纯文本 ("this is a dog")，一种是已经分好词的token id list [123, 23, 34].

## 如果你传入的是纯文本

1. 分词

- 两步，转为token id list（不带cls 和 sep）

```objectivec
# 不带 cls 和sep 
token_list=tokenizer.tokenize("你好！中国科学院。")  # 只分词 ['你', '好', '！', '中', '国', '科', '学', '院', '。'] 
tokenizer.convert_tokens_to_ids(token_list)  # 转为token id list [872, 1962, 8013, 704, 1744, 4906, 2110, 7368, 511] 
```

- 一步直接转为token id list （带cls 和sep，带truncation，带padding）

```python
tokenizer.encode(text="你好！中国科学院。", max_length=15, pad_to_max_length=True, truncation=True, return_special_tokens_mask=True)
# [101, 872, 1962, 8013, 704, 1744, 4906, 2110, 7368, 511, 102, 0, 0, 0, 0] ，101是cls，102是sep
```

2. 编码

```python
tokenizer(text="你好！中国科学院。", max_length=15, pad_to_max_length=True, truncation=True, return_special_tokens_mask=True)
"""
生成的文本可以直接送入bert
{'input_ids': [101, 872, 1962, 8013, 704, 1744, 4906, 2110, 7368, 511, 102, 0, 0, 0, 0], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'special_tokens_mask': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]}
"""
```



## 如果你传入的是已经分好词的token list

1. 编码：和对文本`tokenizer`的结果是一致的

```python
token_list=tokenizer.tokenize("你好！中国科学院。")
token_ids = tokenizer.convert_tokens_to_ids(token_list)  # 输入id
b=tokenizer.encode_plus(text=token_list, max_length=15, pad_to_max_length=True, truncation=True, return_special_tokens_mask=True)
b=tokenizer.encode_plus(text=token_ids, max_length=15, pad_to_max_length=True, truncation=True, return_special_tokens_mask=True)
"""
生成的文本可以直接送入bert
{'input_ids': [101, 872, 1962, 8013, 704, 1744, 4906, 2110, 7368, 511, 102, 0, 0, 0, 0], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'special_tokens_mask': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]}
"""


e=tokenizer.prepare_for_model(
            token_ids,
            truncation=True,
            max_length=15,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
```

1. 转换为tensor

```ini
d=tokenizer.pad({'input_ids':b['input_ids'],'token_type_ids':b['token_type_ids'], 'attention_mask':b['attention_mask']}, return_tensors="pt", padding=False)
```





## Appendix
