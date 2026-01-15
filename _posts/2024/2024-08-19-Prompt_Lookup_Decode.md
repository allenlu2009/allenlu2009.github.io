---
title: "LLM - 加速 : Prompt Lookup Decode"
date: 2024-08-19 23:10:08
description: Prompt Lookup Decode
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - HuggingFace
  - LLM
  - LLM-Accelerate
  - Retrieval
  - Speculative
---




LLM 最大的速度瓶頸是的 memory footprint 以及 generation mode 遇到的 memory bandwidth 瓶頸。 加速有多種方式，但是萬變不離其宗都是把 autoregressive 的 sequential decode 變成 parallel decode (verification).  目前常見的做法：

* Speculative decode:  利用小 (draft model) 和大 (native model) 模型達成加速 [[2023-12-04-Speculative_Decode|blog]]。
* Medusa decode:  利用在 draft model 的 multi-heads 的 information 預測達成加速 [[2023-12-10-Medusa_Memory|blog]]。
* Lookahead decode:  利用數學的解聯立方程式的迭代法 (Jacob or GS-Jacob) 達成加速 [[2023-12-04-Lookahead_Decode|blog]]。
* Retrieval decode: 利用靜態或動態的 “word bank" 的 n-gram 特性，不自己產生 token，達成加速。例如 Prompt-Lookup-Decode,  REST. 
* Encode (prompt, parallel) + decode (generative): 利用 prompt parallel mode 的 hint 給 decode. 



本文在 Colab L4 GPU with 22GB DRAM 的 GPU 跑 Mistral-7B Model


## Prompt Lookup Decode

不論是觀念還是做法上，這是最簡單但是有效的一種方法。
因為非常簡單，很多 model hub 都已經整合 prompt lookup decode 在 model 中。

### HuggingFace Transformer

HuggingFace 已經把 prompt lookup decode 整合到 transformer 的 option.
就是 prompt_lookup_num_tokens.  Default 是 0 (off).

> generation_output = model.generate(**input_ids, do_sample=False, max_new_tokens=512, streamer=streamer, prompt_lookup_num_tokens=10)


```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
 
model_name_or_path = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"
 
tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

chat = [
    {
        "role": "system",
        "content": "You are Hermes 2, an unbiased AI assistant, and you always answer to the best of your ability."
    },
    {
        "role": "user",
        "content": (
            "You are given a partial and unparsed scientific article, please read it carefully and complete the "
            f"request below.{article}Please summarize the article in 5 sentences."
        )
    },
]
processed_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(processed_chat, return_tensors='pt').to(model.device)
 
streamer = TextStreamer(tokenizer)

# baseline without prompt lookup decode
# generation_output = model.generate(**input_ids, do_sample=False, max_new_tokens=512, streamer=streamer)

# with prompt lookup decode
generation_output = model.generate(**input_ids, do_sample=False, max_new_tokens=512, streamer=streamer, prompt_lookup_num_tokens=10)

max_memory = torch.cuda.max_memory_allocated(model.device)
print("Max memory (MB): ", max_memory * 1e-6)
new_tokens = generation_output.shape[1] - input_ids.input_ids.shape[1]
print("Throughput (tokens/sec): ", new_tokens / (start_event.elapsed_time(end_event) * 1.0e-3)) ```


Model Size and GPU Configuration
* Mixtrial-7B:  4.15 GB (INT4 precision)
* GPU L4,  22.5GB.

|                         | w/o PLD | w/ PLD | Comment  |
| ----------------------- | ------- | ------ | -------- |
| Max Memory (MB)         | 6386    | 6389   | the same |
| Throughput (tokens/sec) | 26.44   | 10.56  | 2.5X     |

### Llama.cpp

https://github.com/ggerganov/llama.cpp/tree/master/examples/lookup



## 原理和 implementation







## Reference

X:  Prompt Lookup Decode demo:  https://twitter.com/joao_gante/status/1747322413006643259
HuggingFace: https://huggingface.co/docs/transformers/generation_strategies






