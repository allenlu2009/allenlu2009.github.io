---
title: LLM KV Cache Code
date: 2023-12-16 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

* [【手撕LLM-KVCache】显存刺客的前世今生--文末含代码 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/667763542)
* [大模型推理性能優化之KV Cache解讀 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/630832593)
* The KV Cache: Memory Usage in Transformers  https://www.youtube.com/watch?v=80bIUggRJf4&ab_channel=EfficientNLP
* [[LLM\]KV cache詳解 圖示，顯存，計算量分析，代碼 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/646577898)





<img src="/media/image-20231029165342034.png" alt="image-20231029165342034" style="zoom: 67%;" />









## 1. KV Cache是啥？

大模型推理性能優化的一個常用技術是KV Cache，該技術可以在不影響任何計算精度的前提下，**通過減少計算提高推理性能。**但是沒有白吃的午餐，需要更多的内存空間做爲 KV cache.  

## 2. 背景

生成式generative模型的推理過程很有特點，我們給一個輸入文本 (長度為 $s$)，模型會輸出一個回答（長度爲 $n$），其實該過程中執行了$n$ 次推理 (inference) 過程。**即GPT類模型一次推理只輸出一個token，輸出token會與輸入tokens 拼接在一起，然後作爲下一次推理的輸入，這樣不斷反覆直到遇到終止符。**

如上描述是我們通常認知的GPT推理過程。代碼描述如下：

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model = GPT2LMHeadModel.from_pretrained("/WORK/Test/gpt", torchscript=True).eval()

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/WORK/Test/gpt")
in_text = "Lionel Messi is a"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198]) # line break symbol
out_token = None
i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, _ = model(in_tokens)
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = torch.cat((in_tokens, out_token), 0)
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1

out_text = tokenizer.decode(in_tokens)
print(f' Input: {in_text}')
print(f'Output: {out_text}')
```

輸出

```
step 0 input: Lionel Messi is a player
step 1 input: Lionel Messi is a player who
step 2 input: Lionel Messi is a player who has
step 3 input: Lionel Messi is a player who has been
step 4 input: Lionel Messi is a player who has been a
step 5 input: Lionel Messi is a player who has been a key
step 6 input: Lionel Messi is a player who has been a key part
step 7 input: Lionel Messi is a player who has been a key part of
step 8 input: Lionel Messi is a player who has been a key part of the
step 9 input: Lionel Messi is a player who has been a key part of the team
step 10 input: Lionel Messi is a player who has been a key part of the team's
step 11 input: Lionel Messi is a player who has been a key part of the team's success
step 12 input: Lionel Messi is a player who has been a key part of the team's success.
step 13 input: Lionel Messi is a player who has been a key part of the team's success.

 Input: Lionel Messi is a
Output: Lionel Messi is a player who has been a key part of the team's success.
```

可以看出如上計算的問題嗎？每次推理過程的輸入tokens都變長了 ($n_{ctx}$)，導致推理FLOPs隨之增大。有方法實現推理過程的FLOPs基本恆定不變或變小嗎？（*埋個伏筆，注意是基本恆定*）。



## 3. 原理

b: batch

s: sequence length

h: model input dimension

在上面的推理過程中，每 step 內，輸入一個 token序列，經過Embedding層將輸入token序列變爲一個三維張量[b, s, h]，經過一通計算，最後經logits層將計算結果映射至詞表空間，輸出張量維度爲[b, s, vocab_size]。

**以上 GPT2 code 爲例： b = 1;  s 會每次加 1, 最大到 $n_{ctx}$ ;  h = 768;  vocab_size = 50257**

當前輪輸出token與輸入tokens拼接，並作爲下一輪的輸入tokens，反覆多次。可以看出第 $i+1$ 輪輸入數據只比第 $i$ 輪輸入數據新增了一個token，其他全部相同！因此第 $i+1$輪推理時必然包含了第 $i$ 輪的部分計算。

從 attention block 的角度來看，就是下面的 $x_i : [b, s, h]$  到下次 $x_{i+1} : [b, s+1, h]$  每次 token 長度都會加一。計算量也會變大。

step $i$:

* 計算 Q, K, V:   input 和 output shape at i step:  $[b, s, h] \times [h, h] \to  [b, s, h]$  
* 計算 $Q K^T$  矩陣的 input 和 output $ [b, head_{num}, s, d_{head}] \times [b, head_{num}, s, d_{head}] \to [b, head_{num}, s, s]$

step $i+1$:

* 計算 Q, K, V:   input 和 output shape at i+1 step:  $[b, s+1, h] \times [h, h] \to  [b, s+1, h]$  
* 計算 $Q K^T$  矩陣的 input 和 output $ [b, head_{num}, s+1, d_{head}] \times [b, head_{num}, s+1, d_{head}] \to [b, head_{num}, s+1, s+1]$



<img src="/media/image-20231029190338478.png" alt="image-20231029190338478" style="zoom: 33%;" />

開始是 $s=1, 2, ...$, 直到最後 $s = n_{ctx}$ (maximum context length, GPT2 = 1024).   此時已到達 sequence lengthh 的上限 .  接下來每次進來的 token 都會 shift 掉一個最前面的 token.   也就是 $x_{i+1}$ 是 shifted $x_i$.

<img src="/media/image-20231029192355602.png" alt="image-20231029192355602" style="zoom: 33%;" />

最暴力的方法是每次都計算大的矩陣乘法。但是如果我們可以緩存前一次的 (key, value) 值。是否可以減少重算下一次的 (key, value)?

<img src="/media/image-20231029192501455.png" alt="image-20231029192501455" style="zoom: 33%;" />



**KV Cache的出發點就在這裏，緩存當前輪可重複利用的計算結果，下一輪計算時直接讀取緩存結果，就是這麼簡單，不存在什麼Cache miss問題。** 

SM stands for SoftMax.

<img src="/media/image-20231029202752642.png" alt="image-20231029202752642" style="zoom:33%;" />

例如在輸入新的 token “chill"，之前的 "cold" 對應的 K vector 和 attention score (V) 其實都不用重算。只需要計算新的 "chill" 對應的 vector 和 attention score (K, V)   



<img src="/media/image-20231029203045625.png" alt="image-20231029203045625" style="zoom:33%;" />

<img src="/media/image-20231029204503470.png" alt="image-20231029204503470" style="zoom:33%;" />

問題：緩存的做法是否可以用在 FFN (feedforward block)?  好像不行?  因爲 FFN input vector shift 之後對應的 weights 就會完全不同？可是 attention 對應的 score 是 position independent?  **只有 Attention 有 context and KV cache gain!**



#### KV Cache Memory Usage:

KV parameter count: $2 b s h l$;   Memory size: $2 bshl  \text{ *precision}$

<img src="/media/image-20231029205351222.png" alt="image-20231029205351222" style="zoom: 50%;" />









## 實現細節

目前各大模型推理都實現了KV Cache，下面就看如何使用了。我們可以在上面代碼基礎上修改，主要改動：

- 在推理時新增了 past_key_values 參數，該參數就會以追加方式保存每一輪的K V值。kv cache變量內容爲((k,v), (k,v), ..., (k,v))，即有 $n_{layers}$ 個 k,v 組成的一個元組，其中 k 和 v 的維度均爲 $[b, n_{head}, s, d_{head}]$。這裏可以順帶計算出每輪推理對應的 cache 數據量爲 2∗b∗s∗ℎ∗$n_{layers}$ ，這裏 s 值等於當前輪次值。以GPT3-175B爲例，假設以 float16 來保存 KV cache，senquence長度爲100，batchsize=1，則 KV cache佔用顯存爲 2×100×12288×96×2 Byte= 472MB。
- 推理輸出的token直接作爲下一輪的輸入，不再拼接，因爲上文信息已經在 kvcache 中。

代碼示例：

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


model = GPT2LMHeadModel.from_pretrained("/WORK/Test/gpt", torchscript=True).eval()

# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/WORK/Test/gpt")
in_text = "Lionel Messi is a"
in_tokens = torch.tensor(tokenizer.encode(in_text))

# inference
token_eos = torch.tensor([198]) # line break symbol
out_token = None
kvcache = None
out_text = in_text
i = 0
with torch.no_grad():
    while out_token != token_eos:
        logits, kvcache = model(in_tokens, past_key_values=kvcache) # 增加了一個 past_key_values 的參數
        out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
        in_tokens = out_token # 輸出 token 直接作爲下一輪的輸入，不再拼接
        text = tokenizer.decode(in_tokens)
        print(f'step {i} input: {text}', flush=True)
        i += 1
        out_text += text

print(f' Input: {in_text}')
print(f'Output: {out_text}')
```



通過上面代碼只能看到調用層面的變化，實現細節還需看各框架的底層實現，例如Hugging Face的transformers庫代碼實現就比較清爽，在[modeling_gpt2.py](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py%23L319)中Attention部分相關代碼如下：



```python
   query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None: # 當輸出第一個token後，layer_past就是非None了
            past_key, past_value = layer_past # 取出之前計算好的 key, value
            key = torch.cat((past_key, key), dim=-2) # past_key 與當前 token 對應的 key 拼接
            value = torch.cat((past_value, value), dim=-2) # past_value 與當前 token 對應的 value 拼接

        if use_cache is True:
            present = (key, value)
        else:
            present = None
```



其實，KV Cache 配置開啓後，推理過程可以分爲2個階段：

1. 預填充階段 ($s = 1,2, .., n_{ctx}$)：發生在計算第一個輸出token過程中，這時Cache是空的，計算時需要爲每個 transformer layer 計算並保存key cache和value cache，在輸出token時Cache完成填充；FLOPs同KV Cache關閉一致，存在大量 GEMM 操作，推理速度慢。 **正常推理，預存 key-value cache；compute-bound 計算**
2. 使用KV Cache階段：發生在計算第二個輸出token至最後一個token過程中，這時Cache是有值的，每輪推理只需讀取Cache，同時將當前輪計算出的新的Key、Value追加寫入至Cache；FLOPs降低，GEMM 變爲 GEMV 操作，推理速度相對第一階段變快，這時屬於Memory-bound類型計算。**memory-bound 計算**

這裏用圖可能更有助理解，下圖是一個Decoder Block，含有Self-Attention和MLP，標紅部分爲KV Cache影響到的內容，即KV Cache開啓後，標紅的序列長度 s 變爲 1，當batch_size=1時，Self-Attention中的2個dense全都變爲gemv操作，MLP中的dense也全都變爲gemv操作。看懂這個圖就可以答對上面的3個問題啦。

![ ](https://pic2.zhimg.com/80/v2-6f6b895d6d37154654ffcc13bd23bf9d_720w.webp)





### 總結

KV Cache是Transformer推理性能優化的一項重要工程化技術，各大推理框架都已實現並將其進行了封裝（例如 transformers庫 generate 函數已經將其封裝，用戶不需要手動傳入past_key_values）並默認開啓（config.json文件中use_cache=True）。



### NanoGPTplus 例子

<img src="/media/image-20231216214404615.png" alt="image-20231216214404615" style="zoom:67%;" />



#### 無 KV-Cache 的例子

```python
import torch
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM

# 加载模型
config = LlamaConfig(vocab_size = 100,
                    hidden_size = 256,
                    intermediate_size = 512,
                    num_hidden_layers = 2,
                    num_attention_heads = 4,
                    num_key_value_heads = 4,
                    )
model = LlamaForCausalLM(config)

# 创建数据、不使用tokenizer
X = torch.randint(0, 100, (1,10))
print(X.shape)

# 
idx={}
idx['input_ids'] = X
for i in range(4):
    print(f"\nGeneration第{i}个时的输入{idx['input_ids'].shape}：")
    print(f"Generation第{i}个时的输入{idx['input_ids']}：")
    output = model(**idx) 
    logits = output['logits'][:,-1,:]
    idx_next = torch.argmax(logits , dim=1)[0]
    
    idx['input_ids'] = torch.cat((idx['input_ids'], idx_next.unsqueeze(0).unsqueeze(1)), dim=-1) 
```

結果是

```
torch.Size([1, 10])

Generation第0个时的输入torch.Size([1, 10])：
Generation第0个时的输入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25]])：

Generation第1个时的输入torch.Size([1, 11])：
Generation第1个时的输入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25,  1]])：

Generation第2个时的输入torch.Size([1, 12])：
Generation第2个时的输入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25,  1, 66]])：

Generation第3个时的输入torch.Size([1, 13])：
Generation第3个时的输入tensor([[48,  8, 96,  3,  1,  3, 65, 85, 18, 25,  1, 66,  3]])：
```





#### 有 KV-Cache 的例子

```python
    # this code generate With KV Cache
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
        logits = model(x, max_seq_length, input_pos)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)
        # advance
        input_pos = input_pos[-1:] + 1
        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

    return idx
```





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





#### KV Cache for Inference (主要用於推理)

**KV Cache 的主要工作是減少 computation!  不是 DRAM BW reduction!  剛好相反，KV cache 會增加 DRAM bandwidth.**

**本質上是“空間換時間”。**

1. 0-cache  每次都要從 DRAM 讀 parameter 計算所有 output token:
   * DRAM BW = parameter size x output token/sec
   * Computation = 2 x parameter size TOPS

2. KV cache 假設 internal SRAM = parameter size + KV cache:  理論上 DRAM access 只需要一次?
   * DRAM BW = parameter size x output token/sec + KV cache size x 6 x output token/sec?
   * **DRAM BW / token = parameter size + KV cache size x 6 (讀幾次? 寫幾次?)**
   * **Computation = ??  TOPS (減少 s 倍)  見下文**




是否可能 “時間換空間"?  On-die 7GB 或是 3.5GB SRAM，不可能！





在推斷階段，transformer模型加速推斷的一個常用策略就是使用 KV cache。一個典型的大模型生成式推斷包含了兩個階段：

1. **預填充階段**：輸入一個prompt序列，爲每個transformer層生成 key cache和value cache（KV cache）。
2. **解碼階段**：使用並更新KV cache，一個接一個地生成詞，當前生成的詞依賴於之前已經生成的詞。



第 $i$個transformer層的權重矩陣爲 $W_Q^i, W_K^i, W_V^i, W_O^i, W_1^i, W_2^i$。

其中 self-attention 的 4 個權重矩陣  $W_Q^i, W_K^i, W_V^i, W_O^i \in R^{h \times h}$。

並且MLP塊的2個權重矩陣 $W_1^i \in R^{h \times 4h}, W_2^i \in R^{4h \times h}$。



**預填充階段**

假設第 $i$個transformer層的輸入爲 $x^i$ ，self-attention塊的key、value、query和output表示爲 $x_K^i, x_V^i, x_Q^i, x_{out}^i$ 其中 $x_K^i, x_V^i, x_Q^i, x_{out}^i \in R^{b\times s\times h}$。

Key cache 和 value cache 的計算過程為

$x_K^i = x^i \cdot W_K^i$

$x_V^i = x^i \cdot W_V^i$

第 $i$ 個 transformer 層剩餘的計算過程為

<img src="/media/image-20231022220017738.png" alt="image-20231022220017738" style="zoom: 67%;" />

**解碼階段**

給定當前生成詞在第 $i$ 個transformer層的向量表示爲 $t^i \in R^{b \times 1 \times h}$.  推理計算分兩部分：更新 KV cache 和計算第 $i$ 個 transformer 層的輸出。

更新 key cache 和 value cache 的計算過程如下：

<img src="/media/image-20231022220801197.png" alt="image-20231022220801197" style="zoom: 67%;" />

## 計算量減少分析：

輸入數據的形狀爲 [b,1,ℎ]，kv cache中含有kv_length個past word。我們**先分析self-attention塊的計算**，

1. 計算 Q, K, V ：矩陣乘法的輸入和輸出形狀爲 [b,1,ℎ]×[ℎ,ℎ]→[b,1,ℎ] 。計算量爲 3∗2bℎ2=6bℎ2 。

2. QK^T 矩陣乘法的輸入和輸出形狀爲[b, head_num, 1, per_head_hidden_size]×[b, head_num, per_head_hidden_size, kv_length+s]→[b,head_num,1,kv_length+1] 。計算量爲 2bs (kv_length +1 )ℎ 。

1. 計算在V上的加權 score . V ，矩陣乘法的輸入和輸出形狀爲 [b, head_num, 1, kv_length+1]×[b,head_num,kv_length+1,per_head_hidden_size]→[b,head_num,1,per_head_hidden_size] 。計算量爲 2bs(kv_length +1)ℎ 。

2. attention後的線性映射，矩陣乘法的輸入和輸出形狀爲 [b,1,ℎ]×[ℎ,ℎ]→[b,1,ℎ] 。計算量爲 2bℎ^2 。

   

**接下來分析MLP塊的計算，計算公式如下**：

1. 第一個線性層，矩陣乘法的輸入和輸出形狀爲 [b,1,ℎ]×[ℎ,4ℎ]→[b,1,4ℎ] 。計算量爲 8bℎ2 。
2. 第二個線性層，矩陣乘法的輸入和輸出形狀爲 [b,1,4ℎ]×[4ℎ,ℎ]→[b,1,ℎ] 。計算量爲 8bℎ2 。

將上述計算量相加，得到**每個transformer層的計算量大約爲** 24 b h2 + 4bℎ+4b(kv_length )ℎ 。

不採用kv cache時爲： 24 b sh ℎ2+4 b s2 ℎ

此外，另一個計算量的大頭是logits的計算，將隱藏向量映射爲詞表大小。矩陣乘法的輸入和輸出形狀爲 [b,1,ℎ]×[ℎ,V]→[b,1,V] ，計算量爲 2bℎV 。

不採用kv cache時爲： 2bsℎV



**Attention 的計算量可以節省 s 倍！  Really??**



### KV cache 額外的顯存佔用分析

<img src="/media/image-20231029093628970.png" alt="image-20231029093628970" style="zoom:33%;" />

* 存儲 kvlength 個K｜V value，形狀爲 [b, head_num, kv_seq_len, head_dim]，

* 顯存佔用爲： 4blh(kv_length)

假設輸入序列的長度爲 $s$，輸出序列的長度爲  $n$，以float16來保存KV cache，那麼**KV cache的峯值顯存佔用大小爲** $b(s+n)h*l*2*2 = 4blh(s+n)$。**這裏第一個2表示K/V cache，第二個2表示float16佔2個bytes。**

* Training 的中間激活時 : $34 blsh + 11 b l s^2 a$,  KV cache 只存了 attention 中的 K and V 部分，有包含 score?
* Model 參數量是 $12 l h^2$ (和 b, s 無關！),  假設是 16-bit,  Model 內存是  $24 l h^2$
* 假設 inference $b=1$ (這不一定是對的，在 speculative decode, 大 model 的 $b > 1$):   KV cache : $4 blh (s+n)$.   KV cache / model parameter ~ $b (s+n) / 6 h$!   對於 long context,  $s + n$ 可能會大於 $h$!!  $s$ 就是 $n_{ctx}$,  $h$ 就是 $d_{model}$
* 以 Llama2-7B 爲例,  $h = 4096$,  但是 $n_{ctx} 最大也有 4096$!

##### 

#### Example Llama2 (4A16W)

以 Llama2-7B 爲例。

| 模型名     | 參數量 | 層數, l | 隱藏維度, h | 注意力頭數 a | Context s |
| ---------- | ------ | ------- | ----------- | ------------ | --------- |
| Llama2-7B  | 7B     | 32      | 4096        | 32           | 4096      |
| Llama2-13B | 13B    | 40      | 5120        | 40           | 4096      |
| Llama2-33B | 33B    | 60      | 6656        | 52           | 4096      |
| Llama2-70B | 70B    | 80      | 8192        | 64           | 4096      |

Llama2 的模型參數量爲7B，佔用的顯存大小爲 **(INT8**) 7Bx2 = 7GB 。假設 activation 是 FP16.

假設 Llama2 的序列長度 $s$ 爲 2048 。對比不同的批次大小 $b$ 佔用的中間激活：

當 b=1 時，KV cache 佔用顯存爲 $(4bsh)*l$ byte ≈1GB ，大約是模型參數顯存的15%。

假設 Llama2 的序列長度 $s$ 爲 4096 。對比不同的批次大小 $b$ 佔用的中間激活：

當 b=1 時，KV cache 佔用顯存爲 $(4bsh)*l$ byte ≈2.1GB ，大約是模型參數顯存的31%。

如果 model 是 4-bit (4W16A)  7Bx0.5 = 3.5GB, 更糟糕:  KV cache 佔的比例 double.   



#### Example GPT3-175B (8A16W)

以GPT3-175B爲例，我們來直觀地對比下模型參數與中間激活的顯存大小。GPT3的模型配置如下。我們假設採用混合精度訓練，模型參數和中間激活都採用float16數據類型，每個元素佔2個bytes。

| 模型名 | 參數量 | 層數, l | 隱藏維度, h | 注意力頭數 a |
| ------ | ------ | ------- | ----------- | ------------ |
| GPT3   | 175B   | 96      | 12288       | 96           |

GPT3的模型參數量爲175B，佔用的顯存大小爲 1×175B = 175GB for inference。

GPT3的序列長度 $s$ 爲 2048 。對比不同的批次大小 $b$ 佔用的中間激活：

b=1 ，輸入序列長度 s=2048,  中間激活佔用顯存爲 $(4bsh)*l$ byte ≈9.7GB ，大約是模型參數顯存的 5.6%。

 b=64 ，輸入序列長度 s=512 ，輸出序列長度 n=32 ，則KV cache佔用顯存爲 $4blh(s+n) = 164 GB$，大約是模型參數顯存的 1 倍。

<img src="/media/image-20231029092759285.png" alt="image-20231029092759285" style="zoom: 50%;" />




