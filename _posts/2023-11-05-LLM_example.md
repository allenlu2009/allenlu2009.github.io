---
title: LLM Toy Example
date: 2023-10-14 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

* ShakespeareGPT  https://www.kaggle.com/code/shreydan/shakespearegpt

* [(56) Let's build GPT: from scratch, in code, spelled out. - YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)



## Takeaway

Karpathy 有幾個 key concept 非常好!   

* LLM weights 基本是 information (lossy) compression,  從 entropy/information 的角度 make sense.
* Attention 是 **token to token communication**,   FFC/MLP 是每個 **token 内部的 computation.**  



<img src="/media/image-20231029165342034.png" alt="image-20231029165342034" style="zoom: 67%;" />



### Dataset

莎士比亞的文章。總共約 10,000 words? or characters.



### Tokenizer

1. character level encoder:  codebook 65
2. BPE (byte-pair encoder)
   * GPT2-3:   codebook 50541?
   * Tiktoken (OpenAI)

基本是 trade-off of the codebook vs. the token length!

[Hii, hello world]:  character tokenizer: 12 tokens;  BPE:  3 tokens



### Batch Training 

1. context length: block size
2. variable context: (1, 1), (2, 1), (3, 1), .... (block_size, 1)





### Language Model



#### Bigram Lanuage Model

* Simplest embedding model (vocab_size x vocab_size: 65x65)

* Traing (forward and backward) and generate





##### 關鍵是 next token prediction from T token to T+1 token

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```





#### Transformer Model

第一個 transformer example: 210K parameters

第二個 transformer example: 888K parameters

第三個 transformer example: 11.5K parameters







## Appendix

