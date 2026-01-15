---
title: LLM KV Cache Code
date: 2023-12-16 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2023-10-29-KV_Cache]]"
categories:
  - GenAI
tags:
  - Attention
  - KV-Cache
  - LLM
  - LLM-Inference
  - Transformer
---


## Source

* [【手撕LLM-KVCache】顯存刺客的前世今生--文末含代碼 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/667763542)
* [大模型推理性能優化之KV Cache解讀 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/630832593)
* The KV Cache: Memory Usage in Transformers  https://www.youtube.com/watch?v=80bIUggRJf4&ab_channel=EfficientNLP
* [[LLM\]KV cache詳解 圖示，顯存，計算量分析，代碼 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/646577898)



##  KV Cache 原理

Without KV Cache:
X: input[token_len, D:embed]
Wq, Wk, Wv: [D:embed, D:embed]
Q, K, V: [token_len, D:embed]

With KV Cache:

Attention layer:
第一次是 pre-fill phase, 和 without KV Cache 一樣。儲存 Cache_K and Cache_V.
接下來每次都只有一個 input token (from previous output), 和 KV caches 一起產生新的一個 output token.
X: input[1, D:embed]
Wq, Wk, Wv: [D:embed, D:embed]
Q, K, V: [1, D:embed],  K+Cache_K, V+Cache_V: [K+1, D:embed]
softmax(Q . K+Cache_K) (V+Cache_V): [1, D] [D, K+1] [K+1, D] = [1, D]

FF layer:  和 KV cache 無關。

<img src="/media/image-20231216214404615.png" alt="image-20231216214404615" style="zoom:67%;" />



#### 無 KV-Cache 的示意

```python
import torch
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM

# 加載模型
config = LlamaConfig(vocab_size = 100,
                    hidden_size = 256,
                    intermediate_size = 512, # FF layer hidden state
                    num_hidden_layers = 2,
                    num_attention_heads = 4,
                    num_key_value_heads = 4,
                    )
model = LlamaForCausalLM(config)

# 創建數據、不使用tokenizer
X = torch.randint(0, 100, (1,10))
print(X.shape)

# 
idx={}
idx['input_ids'] = X
for i in range(4):
    print(f"\nGeneration第{i}個時的輸入{idx['input_ids'].shape}：")
    print(f"Generation第{i}個時的輸入{idx['input_ids']}：")
    output = model(**idx)  # output dimension = (batch, seq_len, num_head, head_dim)?
    logits = output['logits'][:,-1,:]
    idx_next = torch.argmax(logits , dim=1)[0]
    # 沒有 kv-cache, 需要 cat input 和 output
    idx['input_ids'] = torch.cat((idx['input_ids'], idx_next.unsqueeze(0).unsqueeze(1)), dim=-1) 
```

結果是

```
torch.Size([1, 10])

Generation第0個時的輸入torch.Size([1, 10])：
Generation第0個時的輸入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25]])：

Generation第1個時的輸入torch.Size([1, 11])：
Generation第1個時的輸入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25,  1]])：

Generation第2個時的輸入torch.Size([1, 12])：
Generation第2個時的輸入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25,  1, 66]])：

Generation第3個時的輸入torch.Size([1, 13])：
Generation第3個時的輸入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25,  1, 66,  3]])：
```

#### 有 KV-Cache 的示意

1. **KV Cache Initialization**:
    - `kv_cache = None`: Before the loop, initialize the cache to `None`. It will store the key-value pairs after the first pass through the model.
2. **Model Call with KV Cache**:
    - `logits, kv_cache = model(x, max_seq_length, input_pos, kv_cache=kv_cache)`: Pass the `kv_cache` into the model. On the first forward pass, `kv_cache` is `None`, but the model returns the computed keys and values and updates `kv_cache` with these values for future iterations.
    - On subsequent passes, the `kv_cache` contains the keys and values for previously seen tokens, so the model only needs to compute new keys and values for the new token.

#### How KV Cache Works:

- **Initial Pass**: When generating the first token, the model calculates the attention for all tokens in the current sequence and stores the key-value pairs in `kv_cache`.
- **Subsequent Passes**: For each newly generated token, the model only computes the key and value tensors for that specific token. The cached key-value pairs are used for all previous tokens, thus reducing the amount of computation needed for the model's attention mechanism.


**To fully integrate KV caching, you’d need to modify the model itself to support this mechanism, ensuring it can accept and return cached keys and values for reuse during token generation.**

```python
    # this code generate With KV Cache
    kv_cache = None
    i = 0
    T = idx.size(0)
    T_new = T+max_new_tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)
    max_new_tokens = 10
    
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)
        print(f"input_t{i}: ", x.int())
        i += 1
        # forward
        logits, kv_cache = model(x, max_seq_length, input_pos, kv_cache=kv_cache)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)
        # advance
        input_pos = input_pos[-1:] + 1
        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

    return idx
```


```
input_t0:  tensor([[ 1,  3, 13,  9, 20,  4,  3,  9,  5]], dtype=torch.int32)
input_t1:  tensor([[3]], dtype=torch.int32)
input_t2:  tensor([[8]], dtype=torch.int32)
input_t3:  tensor([[4]], dtype=torch.int32)
input_t4:  tensor([[3]], dtype=torch.int32)
input_t5:  tensor([[6]], dtype=torch.int32)
input_t6:  tensor([[13]], dtype=torch.int32)
input_t7:  tensor([[6]], dtype=torch.int32)
input_t8:  tensor([[15]], dtype=torch.int32)
input_t9:  tensor([[23]], dtype=torch.int32)
```



### 有 KV-Cache 的實施例  (from NanoGPTplus)
- 此處 enable kv cache code 都放在 generate.  但需要 model 本身支持 kv_cache as input and output kv_cache.  下例會揭開黑盒子說明如何實現。
- 另外只用一個 token 也是放在 generate, 而不是 model 內部。
- 此處使用 Top-K Sampling instead of Greedy Sampling

檢查 idx, kv_cache, context 的 shape (dimension)
* idx : (batch, seq_len): batch =1  而且 seq_len 每次加 1.
* **kv_cache: (k/v, batch, num_head , seq_len, head_dim) = (2, 1, 12, s, 64)**


```python
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        use_kv_cache: bool,
        temperature: float = 1.0,
        top_k_logits: Optional[int] = None,
    ) -> Tensor:
        """Generate one new token after the current one.
        Parameters
        ----------
        idx : Tensor
            index of the current character
        max_new_tokens : int
            number of characters to be generated
        use_kv_cache: bool
            use key-value cache for speed up token generation; if true the number of generated tokens
            should not be larger than context size of the model
        temperature : float, optional
            If the temperature is low, the probabilities to sample with the highest log probability
        top_k_logits : Optional[int], optional
            only top K logits (with the highest value) will be kept, by default None

        Returns
        -------
        Tensor
            tensor containing indices of the provided characters and newly generated
        """
        if use_kv_cache and (max_new_tokens + idx.shape[-1] - 1) > self.context_size:
            msg = (
                "With kv-cache the number of new tokens should not be greater than context size"
            )
            logger.error(msg)
            raise ValueError(msg)
        # in the beginning initialize kv-cache either as None values if kv-cache is disabled,
        # or as empty tensors if enabled, kv cache 是每一層都要！
        kv_cache = (
            [torch.empty(2, 0, device=idx.device, dtype=idx.dtype) for _ in range(self.num_layers)]
            if use_kv_cache
            else None
        )
        for iteration in trange(max_new_tokens, ascii=True):
            # with kv-cache - use only last token, without - crop to the last block_size
            # also crop to the last block if idx provided with more than 1 token in the
            # beginning of token generation (start words)
            if not use_kv_cache or (iteration == 0 and idx.shape[-1] > 1):
                context = idx[:, -self.context_size :]
            else:
                context = idx[:, -1:]   # with kv-cache, 只用最新的一個 token
            # get the predictions
            logits, kv_cache = self(
                context,
                inference=True,
                kv_cache=kv_cache if use_kv_cache else None,
            )  # (B, T, C), with inference=True -> (1, 1, C)
            # focus only on the last time step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # becomes (B, C)
            if top_k_logits:
                # topk returns rearranged tensor where the first column contains the highest values,
                # the last column - the smallest values from top K logits ...
                values, _ = torch.topk(logits, min(top_k_logits, logits.shape[-1]))
                # ... that's why we need to compare with the last column
                logits[logits < values[:, -1]] = float("-inf")  # `-1:` is to preserve dimensionality
            # apply softmax on the predictions to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx
```



### PicoGPT 例子

A very simple picogpt sample code shows:
with kv cache / without kv cache = 0.44 / 0.14 = 3X faster!
(input, output) = (29, 100) tokens
/ml_code/Cursor/nanogpt/nanogpt.py

/ml_code/Cursor/nanogpt/nanogpt.py

重點：
- 此處 enable kv cache code 放在 generate.  但需要 model 本身支持 kv_cache as input and output kv_cache.  下例會揭開黑盒子說明如何實現。
- 另外只選一個 token 是放在 forward, 而不是 generate。
- 此處使用 Top-K Greedy Sampling


```python
class CachedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = nn.MultiheadAttention(config.n_embed, config.n_heads, batch_first=True, bias=config.qkv_bias)
        self.register_buffer("mask", self.create_mask(config.block_size))  # Create mask using triu

    def create_mask(self, block_size):
        """Create a causal mask using the upper triangular matrix."""
        return torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()

    def forward(self, x, attn_mask=None, past_key_value=None):
        batch_size, seq_len, _ = x.shape
        if past_key_value is None:
            key = x
            value = x
            mask = self.mask[:seq_len, :seq_len]
            attn_output, attn_weights = self.mha(x, key, value, attn_mask=mask)
        else:  # post_key_value is NOT None, x is only 1 token
            key, value = past_key_value
            key = torch.cat([key, x], dim=1)
            value = torch.cat([value, x], dim=1)
            #mask = self.mask[:seq_len, -key.shape[1]:]
            attn_output, attn_weights = self.mha(x, key, value)
            
        return attn_output, (key, value)


class CursorGPTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = CachedMultiheadAttention(config)
        self.norm1 = nn.LayerNorm(config.n_embed)
        self.norm2 = nn.LayerNorm(config.n_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embed, config.n_embed * 4),
            nn.ReLU(),
            nn.Linear(config.n_embed * 4, config.n_embed)
        )
        self.drop_shortcut = nn.Dropout(config.dropout)

    def forward(self, x, mask, past_key_value=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, present_key_value = self.att(x, attn_mask=mask, past_key_value=past_key_value)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x, present_key_value


class CursorGPT(nn.Module):
    def __init__(self, config: DecoderConfig):  # Updated to accept DecoderConfig
        super().__init__()
        self.n_embed = config.n_embed
        self.n_layers = config.n_layers
        self.block_size = config.block_size
        self.normx = nn.LayerNorm(config.n_embed)
        self.embedding = nn.Embedding(config.vocab_size, self.n_embed)
        self.position_embedding = nn.Embedding(config.block_size, self.n_embed)
        self.layers = nn.ModuleList([CursorGPTLayer(config) for _ in range(self.n_layers)])
        self.fc_out = nn.Linear(self.n_embed, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        self.kv_cache = config.kv_cache 

    def forward(self, x, mask=None, past_key_values=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        # Apply dropout only during training
        x = self.dropout(self.embedding(x) + self.position_embedding(positions))

        present_key_values = () if self.kv_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            # Now, if past_key_value is not None, we'll only process the last token
            if past_key_value is not None:
                # Adjusting x to only use the last token
                x, present_key_value = layer(x[:, -1:, :], mask, past_key_value)  # Just the last token
            else:
                x, present_key_value = layer(x, mask, past_key_value)
            #x, present_key_value = layer(x, mask, past_key_value)
            #x = layer(x, mask, past_key_value)
            if self.kv_cache:
                present_key_values += (present_key_value,)

        output = self.fc_out(self.normx(x))

        if self.kv_cache:
            return output, present_key_values
        else:
            return output


    def generate(self, idx, new_token, greedy=False):
        # Initialize present_key_values as a tuple of None for each layer
        present_key_values = (None,) * self.n_layers
        for _ in range(new_token):
            idx_cond = idx[:, -self.block_size:]  # Truncates the input sequence to only include the last block_size tokens.

            # Call the model
            if self.kv_cache:
                logits, present_key_values = self(idx_cond, past_key_values=present_key_values)
            else:
                logits = self(idx_cond, past_key_values=None)  # If not using cache, just pass None
            
            next_token_logits = logits[:, -1, :]

            if greedy:  # Check if greedy sampling is enabled
                # Greedy sampling, deterministic
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            else:
                # Probabilistic sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
        return idx[0].tolist()

```