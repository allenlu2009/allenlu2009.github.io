---
title: HuggingFace Dataset and Pytorch Dataset I
date: 2024-04-03 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

* [pytorch-sentiment-analysis/1 - Neural Bag of Words.ipynb at main · bentrevett/pytorch-sentiment-analysis (github.com)](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/1 - Neural Bag of Words.ipynb)  Excellent example!



### 使用 Hugging Face Datasets (不要用 torchtext Datasets!!)

Hugging Face 資料集提供了一个 Python 庫，可以簡化加载、探索和准备机器学习项目所需的数据集。 以下是它的主要优点：

- **統一的界面:** 有效地加载各种格式的数据集，例如文本文件（CSV、JSON、TXT）、网络档案或 Apache Arrow。
- **自動拆分:** 轻松地将数据集拆分训练集、验证集和测试集，以便进行模型评估。
- **数据处理:** 直接在库中使用分词、清理和特征工程功能来预处理文本数据。
- **緩存:** 下载的数据集會缓存在本地，以便更快地访问和减少带宽使用量。

### 自動拆分

数据集通常分为两个或多个拆分，即不重叠的数据示例，最常见的是训练拆分（我们在模型上进行训练）和测试拆分（我们在训练后评估我们的模型）。还有一个验证拆分，我们稍后会详细讨论。训练、测试和验证拆分通常也称为训练、测试和验证集 —— 在这些教程中我们会交替使用拆分和集两个词 —— 数据集通常指的是这三个集合的组合。IMDb数据集实际上还带有第三个拆分，称为未标记拆分，其中包含一堆没有标签的示例。我们不需要这些，所以我们在split参数中不包括它们。请注意，如果我们没有传递split参数，则会加载数据的所有可用拆分。

我们如何知道我们必须使用“imdb”作为IMDb数据集的名称，并且有一个“unsupervised”拆分呢？datasets库有一个很棒的网站，用于浏览可用的数据集，参见：https://huggingface.co/datasets/。通过导航到IMDb数据集页面，我们可以看到关于IMDb数据集的更多信息。

加载数据集时收到的输出告诉我们它正在使用本地缓存的版本，而不是从在线下载数据集。

```python
import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

train_data, test_data
############################################################
(Dataset({
     features: ['text', 'label'],
     num_rows: 25000
 }),
 Dataset({
     features: ['text', 'label'],
     num_rows: 25000
 }))
```

我们可以打印出拆分，其中显示了数据集的特征和num_rows。numrows是拆分中示例的数量，正如我们所看到的，每个拆分中有25,000个示例。datasets库提供的数据集中的每个示例都是一个字典，而特征则是出现在每个字典/示例中的键。因此，IMDb数据集中的每个示例都有一个_text和一个label键。

如果只有 train split, 就只有一半。

```python
train_data = datasets.load_dataset("imdb", split=["train"])
train_data
###############################################################
[Dataset({
     features: ['text', 'label'],
     num_rows: 25000
 })]
```



我们可以检查拆分的features属性以获取有关特征的更多信息。我们可以看到text是dtype=string的值 —— 换句话说，它是一个字符串 —— 而label是一个ClassLabel。ClassLabel意味着该特征是示例所属类别的整数表示。num_classes=2表示我们的标签是两个值之一，0或1，names=['neg', 'pos']给出了这些值的人类可读版本。因此，标签为0表示示例是一个负面评价，标签为1表示示例是一个正面评价。

```python
train_data.features
————————————————————————————————————————————
{'text': Value(dtype='string', id=None),
 'label': ClassLabel(names=['neg', 'pos'], id=None)}
```

训练集来查看一个示例。正如我们所看到的，文本非常杂乱，也有很多废话。

```
train_data[0], train_data[1:10]
```



Create Validation Datasets

```python
test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]
```



## Token (Scalar) :  對應 vocab_size!

在处理数据之前，我们需要做的第一件事就是对其进行 tokenization。机器学习模型并不设计处理字符串，它们设计用来处理数字。因此，我们需要将 strings 拆分为单个 token，然后将这些 token 转换为数字 (scalar)。我们将在稍后进行转换，但首先我们将看一下 tokenization。

Tokenization 涉及使用 tokenizer 处理数据集中的 strings。 Tokenizer 是一个将 strings 转换为 list 的函数。有许多类型的 tokenizer 可用，有幾種不同的方法和用途。

以英文爲例：

* 基於單字母  (character, 也就是 1-byte) 的 tokenizer.  除了教學之外很少用，例如 Karpathy 的 Makemore 的 toy example。除非之後無限長語言模型出現，例如 Mamba-byte，目前可以忽略。
* 基於單詞的 tokenizer.  可以使用 **torchtext.data.utilis.get_tokenizer('basic_english')**。主要用於文本分類，例如情感分析。
* **BPE每一步都将最常见的一对\*相邻数据单位 \*，也就是 Byte-Pair, 替换 (Encode) 为该数据中没有出现过的一个\*新单位\*，反复迭代直到满足停止条件。**  BPE  (Byte-Pair-Encode) 是目前所有 **LLM** 使用的 tokenizer!

以中文爲例

* 基於單字的 tokenizer,  有 UTF-8 (3-bytes), GBK (2-bytes), BIG5 (2-bytes)
* 基於單詞的 tokenizer, 例如 Jieba 結巴 



|                    | 英文 tokenizer (scalar)                                      | 中文 tokenizer (scalar)                                      |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 單字               | 1-byte (vocab size = 256),  only for toy                     | 3-bytes (UTF-8, vocab size: 20-50K); 2-bytes (GBK, BIG5: 20K?) |
| 單詞               | torchtext tokenizer (vocab size > 1M?,  需要設定 min-frequency 限制) | jeba (結巴) (vocab size > 1M?)                               |
| Byte-Pair Encode   | tiktoken (OpenAI), others (HuggingFace)                      | X                                                            |
| Embedding (vector) | Word2Vec,  GloVe                                             | Any vector?                                                  |



但我们将使用torchtext提供的相对简单的 basic_english tokenizer。我们可以这样加载我们的 tokenizer：

```python
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
tokenizer("Hello world! How are you doing today? I'm doing fantastic!")
['hello',
 'world',
 '!',
 'how',
 'are',
 'you',
 'doing',
 'today',
 '?',
 'i',
 "'",
 'm',
 'doing',
 'fantastic',
 '!']
```



## 數據處理

```python
def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens}

max_length = 256

train_data = train_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)
test_data = test_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)
   
train_data
Dataset({
    features: ['text', 'label', 'tokens'],
    num_rows: 25000
})
train_data.features
{'text': Value(dtype='string', id=None),
 'label': ClassLabel(names=['neg', 'pos'], id=None),
 'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)}
 

```



### Creating a Vocabulary

接下来，我们需要构建一个词汇表。这是一个查找表，其中数据集中的每个唯一  token 都有一个相应的索引 index（整数, scalar）。

我们这样做是因为机器学习模型无法处理字符串，只能处理数字值。每个索引用于构建每个标记的独热向量。独热向量是一个所有元素都是0的向量，只有一个元素是1，维度是词汇表中所有唯一标记的总数，通常用来表示。

使用数据集中的每个单词来创建词汇表的一个问题是通常有大量的唯一 token。解决这个问题的方法之一是只使用出现最频繁的 tokens 来构建词汇表，或者只使用数据集中出现超過最少次数的 tokens。在这个 notebook 中，我们选择后者，只保留出现至少 5 次的 tokens。

对于出现少于 5 次的 tokens 会发生什么呢？我们用一个特殊的未知标记来替换它们，表示为 <unk>。例如，如果句子是 "This film is great and I love it"，但是单词 "love" 不在词汇表中，它会变成 "This film is great and I <unk> it"。

我们使用 torchtext.vocab 中的 build_vocab_from_iterator 函数来创建我们的词汇表，指定 min_freq（标记应该出现的最小次数以添加到词汇表）和 special_tokens（即使在数据集中不到 min_freq 次数也应该添加到词汇表开头的标记）。

第一个 special token 是我们的未知标记，另一个 <pad> 是我们用于填充句子的 special token。

当我们将句子输入模型时，我们一次传入一个句子的批次，即一个批次包含多个句子。一次传入一个句子的批次比一次传入一个句子更受欢迎，因为它允许我们的模型在一个批次中并行处理所有句子的计算，从而加快了训练和评估模型所花费的时间。批次中的所有句子都需要具有相同的长度（以 tokens 的数量为准）。因此，为了确保每个句子的长度相同，任何比最长句子短的句子都需要在末尾添加填充标记。

```python
min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)
len(vocab)
21635
```

處理其他的 tokens

```python
unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]
vocab.set_default_index(unk_index)
vocab.lookup_indices(["hello", "world", "some_token", "<pad>"])
[5516, 184, 0, 1]
```



### Numericalizing Data

现在我们有了词汇表，我们可以将数据数字化。这涉及将数据集中的标记转换为索引。与使用 Dataset.map 方法对数据进行标记化类似，我们将定义一个函数，该函数接受一个示例和我们的词汇表，获取每个示例中每个标记的索引，然后创建一个 ids 字段，其中包含数字化的标记。最後再轉成 tensors.

```
def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}

train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])
```



## 最後一步：Create Data Loaders 使用 torch 的 DataLoader!!!!!

最后一步是创建数据加载器。我们可以迭代数据加载器以获取批量的示例。这也是我们将执行任何必要填充的地方。

首先，我们需要定义一个函数来将一批示例（包含在列表中）整理成我们希望数据加载器输出的形式。

在这里，我们希望数据加载器输出的是一个包含键 "ids" 和 "label" 的字典。

batch["ids"] 的值应该是一个形状为 [batch, length] 的张量，其中长度是批中最长句子（按标记计算）的长度，所有比这个长度更短的句子都应该填充到这个长度。

batch["label"] 的值应该是一个形状为 [batch] 的张量，其中包含批中每个句子的标签。

我们定义一个函数 get_collate_fn，它接受填充标记索引并返回实际的整理函数。在实际的整理函数 collate_fn 中，我们获取批中每个示例的 "ids" 张量列表，然后使用 pad_sequence 函数将这些张量列表转换为所需的 [batch, length] 形状的张量，并使用指定的 pad_index 进行填充。默认情况下，pad_sequence 会返回一个形状为 [length, batch] 的张量，但通过设置 batch_first=True，这两个维度将被交换。我们获取 "label" 张量列表，并将这些张量列表转换为单个形状为 [batch] 的张量。

```python
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn
    
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader    
```

接下来，我们定义一个函数来返回我们的实际数据加载器。它接受一个数据集、期望的批大小（我们希望在一个批中有多少句子）、我们的填充标记索引，以及是否应该对数据集进行洗牌。

最后，我们获取训练、验证和测试数据的数据加载器。

我们将批大小设置为 512。我们的批大小应该尽量设置得更高，因为更大的批量意味着更多的并行计算、更少的计算时间，因此训练和评估速度更快。

只有训练数据加载器需要进行洗牌，因为它是用来调整模型参数的唯一数据加载器，而且训练数据应该始终进行洗牌。

```python
batch_size = 512

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)
```





## Embedding (Vector):  對應 embedding dimension

所謂 embeding 就是從把 scalar token 轉換成 vector embedding!   Embedding 可以是 LLM 的一部分，也就網路的第一層。或是已經訓練好 (pre-trained) 的 vector (例如 Word2Vec 或是 GloVE)。在傳統的 NLP.

傳統 NLP:  單詞 tokenizer +  pre-trained embedding +  簡單網路 + downstream tasks.   使用 supervised learning.   例如文本分類 (spam, sentiment, etc.)

LLM:   BPE tokenizer +  LLM (embedding + positional encoding + transformer layers).  使用 self-supervised learning.  最後再 fine-tune 到 downstream tasks!

 













## Appendix

