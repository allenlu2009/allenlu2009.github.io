---
title: LLM Perplexity Benchmark
date: 2025-07-18 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1231-Perplexity]]"
categories:
  - LLM
tags:
  - AI
  - Algorithm
  - GPT
  - LLM
  - Perplexity
  - Transformer
---




## Source
- [(22) The Key Equation Behind Probability - YouTube](https://www.youtube.com/watch?v=KHVR587oW8I&t=810s)  a good youtube video of cross-entropy
- Huggingface perplexity code example
- Code:  wikitext2 perplexity_llm_claude.py


### Perplexity evaluation 參數設定


1. **Block size 大於等於 2048 with stride ratio 0.5 基本得到穩定的結果.**  **建議使用 block size = 4096 with stride ratio 0.5!**  - Stride = block_size × stride_ratio.


## Perplexity Takeaway

1. **Scaling law : bigger model has smaller perplexity**。最好是用 3B-4B SLM 為了實用的目的。
2. 4. Use 3 different datasets (Wiki2, PTB, Shakespeare) for SLM benchmark.  Dataset 難度:  Wiki2 < PTB < Shakespeare。 **Wikitext2 的表現在各 model 最一致穩定**。
	1. Shakespeare: GPT2-XL 表現比小模型差!  其中 Romeo and Juliet 的故事有 data contaimination.
	2. PTB: Gemma3 1B/4B 表現非常差。特別 4B 違反所有的結論。

### Evaluation 參數設定

1. **Batch=1 比較不會 OOM (Out of memory)** 
2. **Stride overlap 愈多愈好，但是計算量大增加。**
	1. 絕對不要用 stride ratio = 1，i.e. 即沒有 overlap!  非常差！
	2. **設定 Stride-ratio = 0.5 非常接近最佳！**
3. Block size 愈大愈好，但是計算量大增加。
	1. **設定 4K block size 實用應該足夠。**



### SMB 1B 組

#### Wikitext2 Dataset

![[Pasted image 20250720201357.png]]

#### PTB Dataset

![[Pasted image 20250720201659.png]]

#### Shakespeare Dataset
![[Pasted image 20250720201825.png]]


### SMB 3/4B 組


#### Wikitext2 Dataset
![[Pasted image 20250720202459.png]]


#### PTB Dataset

![[Pasted image 20250720202532.png]]

**Zoom-in 圖:**
![[Pasted image 20250720202858.png]]

#### Shakespeare Dataset

![[Pasted image 20250720202552.png]]

Zoom-in 圖

![[Pasted image 20250720203029.png]]



### **Lower stride ratio (more overlap) and larger block sizes tend to improve perplexity.**

| stride_ratio | stride | Wikitext2 Loss | Wikitext2 PPL | PTB Loss | PTB PPL | Shakespeare Loss | Shakespeare PPL |
| ------------ | ------ | -------------- | ------------- | -------- | ------- | ---------------- | --------------- |
| 0.1          | 204    | 2.4360         | 11.4278       | 3.1431   | 23.1757 | 3.4120           | 30.3251         |
| 0.2          | 409    | 2.4392         | 11.4642       | 3.1443   | 23.2039 | 3.4139           | 30.3842         |
| 0.3          | 614    | 2.4433         | 11.5111       | 3.1460   | 23.2434 | 3.4162           | 30.4522         |
| 0.4          | 819    | 2.4497         | 11.5848       | 3.1470   | 23.2655 | 3.4183           | 30.5170         |
| 0.5          | 1024   | 2.4538         | 11.6328       | 3.1500   | 23.3357 | 3.4228           | 30.6553         |
| 0.6          | 1228   | 2.4596         | 11.6998       | 3.1539   | 23.4273 | 3.4249           | 30.7195         |
| 0.7          | 1433   | 2.4693         | 11.8144       | 3.1596   | 23.5611 | 3.4318           | 30.9312         |
| 0.8          | 1638   | 2.4866         | 12.0198       | 3.1644   | 23.6734 | 3.4370           | 31.0948         |
| 0.9          | 1843   | 2.5084         | 12.2850       | 3.1801   | 24.0480 | 3.4451           | 31.3462         |
| 1.0          | 2048   | 2.5776         | 13.1659       | 3.2449   | 25.6594 | 3.4924           | **32.8659**     |

### **SLM 1B 組 (GPT2, Llama, Gemma)**

- Wiki2 / PTB / Shakespeare perplexity ~ 16 / 20 / 30.
- GPT2 120M, GPT2-large  744M,  GPT2-XL 1.5B。不過 GPT2 是早期 model, context length = 1K, 應該比較差。但是 GPT2-XL 在 Shakespeare 的 perplexity 比 GPT2-large 差，有點奇怪。
- Gemma 很差，特別是 PTB，不知道爲什麽。

| Model       | Block Size | Stride Ratio | Stride | Dataset     | Avg. Loss | Perplexity |
| ----------- | ---------- | ------------ | ------ | ----------- | --------- | ---------- |
| gpt2        | 1024       | 0.25         | 256    | Wikitext2   | 3.2014    | 24.5658    |
| gpt2        | 1024       | 0.5          | 512    | Wikitext2   | 3.2257    | 25.1704    |
| gpt2        | 1024       | 0.75         | 768    | Wikitext2   | 3.2622    | 26.1061    |
| gpt2        | 1024       | 1.0          | 1024   | Wikitext2   | 3.3992    | 29.9405    |
| gpt2        | 1024       | 0.25         | 256    | PTB         | 3.5072    | 33.3561    |
| gpt2        | 1024       | 0.5          | 512    | PTB         | 3.5267    | 34.0120    |
| gpt2        | 1024       | 0.75         | 768    | PTB         | 3.5639    | 35.2992    |
| gpt2        | 1024       | 1.0          | 1024   | PTB         | 3.7268    | 41.5463    |
| gpt2        | 1024       | 0.25         | 256    | Shakespeare | 4.0258    | 56.0266    |
| gpt2        | 1024       | 0.5          | 512    | Shakespeare | 4.0350    | 56.5412    |
| gpt2        | 1024       | 0.75         | 768    | Shakespeare | 4.0570    | 57.7984    |
| gpt2        | 1024       | 1.0          | 1024   | Shakespeare | 4.1500    | 63.4317    |
| gpt2-large  | 1024       | 0.25         | 256    | Wikitext2   | 2.7803    | 16.1233    |
| gpt2-large  | 1024       | 0.5          | 512    | Wikitext2   | 2.8000    | 16.4443    |
| gpt2-large  | 1024       | 0.75         | 768    | Wikitext2   | 2.8319    | 16.9783    |
| gpt2-large  | 1024       | 1.0          | 1024   | Wikitext2   | 2.9671    | 19.4360    |
| gpt2-large  | 1024       | 0.25         | 256    | PTB         | 3.0156    | 20.4014    |
| gpt2-large  | 1024       | 0.5          | 512    | PTB         | 3.0297    | 20.6905    |
| gpt2-large  | 1024       | 0.75         | 768    | PTB         | 3.0600    | 21.3279    |
| gpt2-large  | 1024       | 1.0          | 1024   | PTB         | 3.2195    | 25.0159    |
| gpt2-large  | 1024       | 0.25         | 256    | Shakespeare | 3.5160    | 33.6506    |
| gpt2-large  | 1024       | 0.5          | 512    | Shakespeare | 3.5317    | 34.1804    |
| gpt2-large  | 1024       | 0.75         | 768    | Shakespeare | 3.5558    | 35.0167    |
| gpt2-large  | 1024       | 1.0          | 1024   | Shakespeare | 3.6629    | 38.9724    |
| gpt2-xl     | 1024       | 0.25         | 256    | Wikitext2   | 2.6737    | 14.4940    |
| gpt2-xl     | 1024       | 0.5          | 512    | Wikitext2   | 2.6938    | 14.7878    |
| gpt2-xl     | 1024       | 0.75         | 768    | Wikitext2   | 2.7243    | 15.2461    |
| gpt2-xl     | 1024       | 1.0          | 1024   | Wikitext2   | 2.8565    | 17.3997    |
| gpt2-xl     | 1024       | 0.25         | 256    | PTB         | 2.9238    | 18.6115    |
| gpt2-xl     | 1024       | 0.5          | 512    | PTB         | 2.9395    | 18.9063    |
| gpt2-xl     | 1024       | 0.75         | 768    | PTB         | 2.9688    | 19.4680    |
| gpt2-xl     | 1024       | 1.0          | 1024   | PTB         | 3.1337    | 22.9577    |
| gpt2-xl     | 1024       | 0.25         | 256    | Shakespeare | 3.7458    | 42.3414    |
| gpt2-xl     | 1024       | 0.5          | 512    | Shakespeare | 3.7567    | 42.8064    |
| gpt2-xl     | 1024       | 0.75         | 768    | Shakespeare | 3.7705    | 43.4018    |
| gpt2-xl     | 1024       | 1.0          | 1024   | Shakespeare | 3.8218    | 45.6877    |
| Llama3.2-1B | 2048       | 0.25         | 512    | Wikitext2   | 2.4417    | 11.4930    |
| Llama3.2-1B | 2048       | 0.5          | 1024   | Wikitext2   | 2.4538    | 11.6328    |
| Llama3.2-1B | 2048       | 0.75         | 1536   | Wikitext2   | 2.4788    | 11.9275    |
| Llama3.2-1B | 2048       | 1.0          | 2048   | Wikitext2   | 2.5776    | 13.1659    |
| Llama3.2-1B | 4096       | 0.25         | 1024   | Wikitext2   | 2.4108    | 11.1432    |
| Llama3.2-1B | 4096       | 0.5          | 2048   | Wikitext2   | 2.4174    | 11.2166    |
| Llama3.2-1B | 4096       | 0.75         | 3072   | Wikitext2   | 2.4310    | 11.3702    |
| Llama3.2-1B | 4096       | 1.0          | 4096   | Wikitext2   | 2.4989    | 12.1688    |
| Llama3.2-1B | 2048       | 0.25         | 512    | PTB         | 3.0441    | 20.9920    |
| Llama3.2-1B | 2048       | 0.5          | 1024   | PTB         | 3.0510    | 21.1357    |
| Llama3.2-1B | 2048       | 0.75         | 1536   | PTB         | 3.0615    | 21.3600    |
| Llama3.2-1B | 2048       | 1.0          | 2048   | PTB         | 3.1478    | 23.2859    |
| Llama3.2-1B | 4096       | 0.25         | 1024   | PTB         | 3.0234    | 20.5601    |
| Llama3.2-1B | 4096       | 0.5          | 2048   | PTB         | 3.0281    | 20.6582    |
| Llama3.2-1B | 4096       | 0.75         | 3072   | PTB         | 3.0340    | 20.7810    |
| Llama3.2-1B | 4096       | 1.0          | 4096   | PTB         | 3.0914    | 22.0071    |
| Llama3.2-1B | 2048       | 0.25         | 512    | Shakespeare | 3.3989    | 29.9325    |
| Llama3.2-1B | 2048       | 0.5          | 1024   | Shakespeare | 3.4064    | 30.1558    |
| Llama3.2-1B | 2048       | 0.75         | 1536   | Shakespeare | 3.4175    | 30.4927    |
| Llama3.2-1B | 2048       | 1.0          | 2048   | Shakespeare | 3.4764    | 32.3436    |
| Llama3.2-1B | 4096       | 0.25         | 1024   | Shakespeare | 3.3793    | 29.3515    |
| Llama3.2-1B | 4096       | 0.5          | 2048   | Shakespeare | 3.3829    | 29.4567    |
| Llama3.2-1B | 4096       | 0.75         | 3072   | Shakespeare | 3.3912    | 29.7030    |
| Llama3.2-1B | 4096       | 1.0          | 4096   | Shakespeare | 3.4291    | 30.8492    |
| gemma3-1B   | 2048       | 0.25         | 512    | Wikitext2   | 3.0110    | 20.3076    |
| gemma3-1B   | 2048       | 0.5          | 1024   | Wikitext2   | 3.0371    | 20.8442    |
| gemma3-1B   | 2048       | 0.75         | 1536   | Wikitext2   | 3.0857    | 21.8827    |
| gemma3-1B   | 2048       | 1.0          | 2048   | Wikitext2   | 3.3237    | 27.7621    |
| gemma3-1B   | 4096       | 0.25         | 1024   | Wikitext2   | 2.9472    | 19.0527    |
| gemma3-1B   | 4096       | 0.5          | 2048   | Wikitext2   | 2.9597    | 19.2915    |
| gemma3-1B   | 4096       | 0.75         | 3072   | Wikitext2   | 2.9830    | 19.7479    |
| gemma3-1B   | 4096       | 1.0          | 4096   | Wikitext2   | 3.1439    | 23.1936    |
| gemma3-1B   | 2048       | 0.25         | 512    | PTB         | 4.6951    | 109.4108   |
| gemma3-1B   | 2048       | 0.5          | 1024   | PTB         | 4.7198    | 112.1477   |
| gemma3-1B   | 2048       | 0.75         | 1536   | PTB         | 4.7643    | 117.2445   |
| gemma3-1B   | 2048       | 1.0          | 2048   | PTB         | 4.9686    | 143.8289   |
| gemma3-1B   | 4096       | 0.25         | 1024   | PTB         | 4.6413    | 103.6824   |
| gemma3-1B   | 4096       | 0.5          | 2048   | PTB         | 4.6524    | 104.8413   |
| gemma3-1B   | 4096       | 0.75         | 3072   | PTB         | 4.6651    | 106.1757   |
| gemma3-1B   | 4096       | 1.0          | 4096   | PTB         | 4.8058    | 122.2155   |
| gemma3-1B   | 2048       | 0.25         | 512    | Shakespeare | 3.7388    | 42.0463    |
| gemma3-1B   | 2048       | 0.5          | 1024   | Shakespeare | 3.7544    | 42.7088    |
| gemma3-1B   | 2048       | 0.75         | 1536   | Shakespeare | 3.7827    | 43.9366    |
| gemma3-1B   | 2048       | 1.0          | 2048   | Shakespeare | 3.9603    | 52.4706    |
| gemma3-1B   | 4096       | 0.25         | 1024   | Shakespeare | 3.6929    | 40.1608    |
| gemma3-1B   | 4096       | 0.5          | 2048   | Shakespeare | 3.7046    | 40.6352    |
| gemma3-1B   | 4096       | 0.75         | 3072   | Shakespeare | 3.7196    | 41.2470    |
| gemma3-1B   | 4096       | 1.0          | 4096   | Shakespeare | 3.8318    | 46.1474    |


### **SLM 3-4B 組 (Llama, Phi, Qwen)**

- Model size:
- Wiki2 / PTB /  Shakespeare perplexity 大約在 8 / 14 / 14 左右。
- Gemma3 表現有點爛，特別在 PTB。而且 4B 比 1B 還差，非常奇怪。
    
- The Qwen2.5-3B model has 3.09 billion parameters (2.77 billion non-embedding parameters) and a context length of 32,768 tokens [2](https://huggingface.co/Qwen/Qwen2.5-3B).
    
- The provided results indicate that for Qwen2.5-3B, using a block size of 8192 generally leads to the best perplexity compared to smaller block sizes for a given stride ratio.

| Model       | Block Size | Stride Ratio | Stride | Dataset     | Avg. Loss | Perplexity |
| ----------- | ---------- | ------------ | ------ | ----------- | --------- | ---------- |
| Llama3.2-3B | 2048       | 0.25         | 512    | Wikitext2   | 1.9370    | 6.9377     |
| Llama3.2-3B | 2048       | 0.5          | 1024   | Wikitext2   | 1.9473    | 7.0097     |
| Llama3.2-3B | 2048       | 0.75         | 1536   | Wikitext2   | 1.9674    | 7.1519     |
| Llama3.2-3B | 2048       | 1.0          | 2048   | Wikitext2   | 2.0560    | 7.8144     |
| Llama3.2-3B | 4096       | 0.25         | 1024   | Wikitext2   | 1.9099    | 6.7523     |
| Llama3.2-3B | 4096       | 0.5          | 2048   | Wikitext2   | 1.9159    | 6.7931     |
| Llama3.2-3B | 4096       | 0.75         | 3072   | Wikitext2   | 1.9275    | 6.8721     |
| Llama3.2-3B | 4096       | 1.0          | 4096   | Wikitext2   | 1.9869    | 7.2926     |
| Llama3.2-3B | 2048       | 0.25         | 512    | PTB         | 2.3731    | 10.7304    |
| Llama3.2-3B | 2048       | 0.5          | 1024   | PTB         | 2.3793    | 10.7968    |
| Llama3.2-3B | 2048       | 0.75         | 1536   | PTB         | 2.3923    | 10.9384    |
| Llama3.2-3B | 2048       | 1.0          | 2048   | PTB         | 2.4849    | 12.0005    |
| Llama3.2-3B | 4096       | 0.25         | 1024   | PTB         | 2.3510    | 10.4960    |
| Llama3.2-3B | 4096       | 0.5          | 2048   | PTB         | 2.3559    | 10.5474    |
| Llama3.2-3B | 4096       | 0.75         | 3072   | PTB         | 2.3626    | 10.6188    |
| Llama3.2-3B | 4096       | 1.0          | 4096   | PTB         | 2.4225    | 11.2740    |
| Llama3.2-3B | 2048       | 0.25         | 512    | Shakespeare | 2.1525    | 8.6066     |
| Llama3.2-3B | 2048       | 0.5          | 1024   | Shakespeare | 2.1582    | 8.6555     |
| Llama3.2-3B | 2048       | 0.75         | 1536   | Shakespeare | 2.1667    | 8.7292     |
| Llama3.2-3B | 2048       | 1.0          | 2048   | Shakespeare | 2.2319    | 9.3180     |
| Llama3.2-3B | 4096       | 0.25         | 1024   | Shakespeare | 2.1371    | 8.4751     |
| Llama3.2-3B | 4096       | 0.5          | 2048   | Shakespeare | 2.1402    | 8.5015     |
| Llama3.2-3B | 4096       | 0.75         | 3072   | Shakespeare | 2.1471    | 8.5602     |
| Llama3.2-3B | 4096       | 1.0          | 4096   | Shakespeare | 2.1851    | 8.8914     |
| Phi4-mini   | 2048       | 0.25         | 512    | Wikitext2   | 2.1263    | 8.3834     |
| Phi4-mini   | 2048       | 0.5          | 1024   | Wikitext2   | 2.1360    | 8.4658     |
| Phi4-mini   | 2048       | 0.75         | 1536   | Wikitext2   | 2.1576    | 8.6504     |
| Phi4-mini   | 2048       | 1.0          | 2048   | Wikitext2   | 2.2523    | 9.5096     |
| Phi4-mini   | 4096       | 0.25         | 1024   | Wikitext2   | 2.1025    | 8.1862     |
| Phi4-mini   | 4096       | 0.5          | 2048   | Wikitext2   | 2.1059    | 8.2147     |
| Phi4-mini   | 4096       | 0.75         | 3072   | Wikitext2   | 2.1142    | 8.2830     |
| Phi4-mini   | 4096       | 1.0          | 4096   | Wikitext2   | 2.1809    | 8.8542     |
| Phi4-mini   | 2048       | 0.25         | 512    | PTB         | 2.6570    | 14.2536    |
| Phi4-mini   | 2048       | 0.5          | 1024   | PTB         | 2.6640    | 14.3539    |
| Phi4-mini   | 2048       | 0.75         | 1536   | PTB         | 2.6750    | 14.5120    |
| Phi4-mini   | 2048       | 1.0          | 2048   | PTB         | 2.7733    | 16.0114    |
| Phi4-mini   | 4096       | 0.25         | 1024   | PTB         | 2.6444    | 14.0746    |
| Phi4-mini   | 4096       | 0.5          | 2048   | PTB         | 2.6464    | 14.1038    |
| Phi4-mini   | 4096       | 0.75         | 3072   | PTB         | 2.6509    | 14.1672    |
| Phi4-mini   | 4096       | 1.0          | 4096   | PTB         | 2.7078    | 14.9960    |
| Phi4-mini   | 2048       | 0.25         | 512    | Shakespeare | 2.9523    | 19.1509    |
| Phi4-mini   | 2048       | 0.5          | 1024   | Shakespeare | 2.9561    | 19.2232    |
| Phi4-mini   | 2048       | 0.75         | 1536   | Shakespeare | 2.9633    | 19.3611    |
| Phi4-mini   | 2048       | 1.0          | 2048   | Shakespeare | 3.0186    | 20.4623    |
| Phi4-mini   | 4096       | 0.25         | 1024   | Shakespeare | 2.9470    | 19.0493    |
| Phi4-mini   | 4096       | 0.5          | 2048   | Shakespeare | 2.9474    | 19.0561    |
| Phi4-mini   | 4096       | 0.75         | 3072   | Shakespeare | 2.9511    | 19.1272    |
| Phi4-mini   | 4096       | 1.0          | 4096   | Shakespeare | 2.9838    | 19.7629    |
| Qwen2.5-3B  | 2048       | 0.25         | 512    | Wikitext2   | 2.0188    | 7.5296     |
| Qwen2.5-3B  | 2048       | 0.5          | 1024   | Wikitext2   | 2.0313    | 7.6238     |
| Qwen2.5-3B  | 2048       | 0.75         | 1536   | Wikitext2   | 2.0534    | 7.7941     |
| Qwen2.5-3B  | 2048       | 1.0          | 2048   | Wikitext2   | 2.1472    | 8.5608     |
| Qwen2.5-3B  | 4096       | 0.25         | 1024   | Wikitext2   | 1.9869    | 7.2928     |
| Qwen2.5-3B  | 4096       | 0.5          | 2048   | Wikitext2   | 1.9938    | 7.3436     |
| Qwen2.5-3B  | 4096       | 0.75         | 3072   | Wikitext2   | 2.0039    | 7.4183     |
| Qwen2.5-3B  | 4096       | 1.0          | 4096   | Wikitext2   | 2.0687    | 7.9145     |
| Qwen2.5-3B  | 8192       | 0.25         | 2048   | Wikitext2   | 1.9659    | 7.1415     |
| Qwen2.5-3B  | 8192       | 0.5          | 4096   | Wikitext2   | 1.9708    | 7.1763     |
| Qwen2.5-3B  | 8192       | 0.75         | 6144   | Wikitext2   | 1.9766    | 7.2179     |
| Qwen2.5-3B  | 8192       | 1.0          | 8192   | Wikitext2   | 2.0228    | 7.5592     |
| Qwen2.5-3B  | 2048       | 0.25         | 512    | PTB         | 2.5680    | 13.0401    |
| Qwen2.5-3B  | 2048       | 0.5          | 1024   | PTB         | 2.5762    | 13.1474    |
| Qwen2.5-3B  | 2048       | 0.75         | 1536   | PTB         | 2.5896    | 13.3243    |
| Qwen2.5-3B  | 2048       | 1.0          | 2048   | PTB         | 2.6893    | 14.7217    |
| Qwen2.5-3B  | 4096       | 0.25         | 1024   | PTB         | 2.5414    | 12.6972    |
| Qwen2.5-3B  | 4096       | 0.5          | 2048   | PTB         | 2.5484    | 12.7868    |
| Qwen2.5-3B  | 4096       | 0.75         | 3072   | PTB         | 2.5552    | 12.8742    |
| Qwen2.5-3B  | 4096       | 1.0          | 4096   | PTB         | 2.6211    | 13.7503    |
| Qwen2.5-3B  | 8192       | 0.25         | 2048   | PTB         | 2.5209    | 12.4399    |
| Qwen2.5-3B  | 8192       | 0.5          | 4096   | PTB         | 2.5266    | 12.5107    |
| Qwen2.5-3B  | 8192       | 0.75         | 6144   | PTB         | 2.5312    | 12.5690    |
| Qwen2.5-3B  | 8192       | 1.0          | 8192   | PTB         | 2.5755    | 13.1377    |
| Qwen2.5-3B  | 2048       | 0.25         | 512    | Shakespeare | 2.5388    | 12.6644    |
| Qwen2.5-3B  | 2048       | 0.5          | 1024   | Shakespeare | 2.5449    | 12.7418    |
| Qwen2.5-3B  | 2048       | 0.75         | 1536   | Shakespeare | 2.5533    | 12.8495    |
| Qwen2.5-3B  | 2048       | 1.0          | 2048   | Shakespeare | 2.6189    | 13.7207    |
| Qwen2.5-3B  | 4096       | 0.25         | 1024   | Shakespeare | 2.5241    | 12.4791    |
| Qwen2.5-3B  | 4096       | 0.5          | 2048   | Shakespeare | 2.5270    | 12.5162    |
| Qwen2.5-3B  | 4096       | 0.75         | 3072   | Shakespeare | 2.5339    | 12.6029    |
| Qwen2.5-3B  | 4096       | 1.0          | 4096   | Shakespeare | 2.5734    | 13.1107    |
| Qwen2.5-3B  | 8192       | 0.25         | 2048   | Shakespeare | 2.5157    | 12.3757    |
| Qwen2.5-3B  | 8192       | 0.5          | 4096   | Shakespeare | 2.5174    | 12.3966    |
| Qwen2.5-3B  | 8192       | 0.75         | 6144   | Shakespeare | 2.5209    | 12.4403    |
| Qwen2.5-3B  | 8192       | 1.0          | 8192   | Shakespeare | 2.5438    | 12.7283    |
| gemma3-4B   | 2048       | 0.25         | 512    | Wikitext2   | 2.3640    | 10.6331    |
| gemma3-4B   | 2048       | 0.5          | 1024   | Wikitext2   | 2.3911    | 10.9254    |
| gemma3-4B   | 2048       | 0.75         | 1536   | Wikitext2   | 2.4596    | 11.6997    |
| gemma3-4B   | 2048       | 1.0          | 2048   | Wikitext2   | 2.8554    | 17.3807    |
| gemma3-4B   | 4096       | 0.25         | 1024   | Wikitext2   | 2.3234    | 10.2104    |
| gemma3-4B   | 4096       | 0.5          | 2048   | Wikitext2   | 2.3298    | 10.2764    |
| gemma3-4B   | 4096       | 0.75         | 3072   | Wikitext2   | 2.3500    | 10.4853    |
| gemma3-4B   | 4096       | 1.0          | 4096   | Wikitext2   | 2.6017    | 13.4865    |
| gemma3-4B   | 2048       | 0.25         | 512    | PTB         | 6.0464    | 422.6029   |
| gemma3-4B   | 2048       | 0.5          | 1024   | PTB         | 5.9746    | 393.3032   |
| gemma3-4B   | 2048       | 0.75         | 1536   | PTB         | 5.6783    | 292.4377   |
| gemma3-4B   | 2048       | 1.0          | 2048   | PTB         | 5.8874    | 360.4721   |
| gemma3-4B   | 4096       | 0.25         | 1024   | PTB         | 6.2515    | 518.8036   |
| gemma3-4B   | 4096       | 0.5          | 2048   | PTB         | 6.2290    | 507.2570   |
| gemma3-4B   | 4096       | 0.75         | 3072   | PTB         | 6.1217    | 455.6433   |
| gemma3-4B   | 4096       | 1.0          | 4096   | PTB         | 6.0715    | 433.3349   |
| gemma3-4B   | 2048       | 0.25         | 512    | Shakespeare | 3.1077    | 22.3704    |
| gemma3-4B   | 2048       | 0.5          | 1024   | Shakespeare | 3.1265    | 22.7944    |
| gemma3-4B   | 2048       | 0.75         | 1536   | Shakespeare | 3.2014    | 24.5672    |
| gemma3-4B   | 2048       | 1.0          | 2048   | Shakespeare | 3.5761    | 35.7343    |
| gemma3-4B   | 4096       | 0.25         | 1024   | Shakespeare | 3.0754    | 21.6576    |
| gemma3-4B   | 4096       | 0.5          | 2048   | Shakespeare | 3.0821    | 21.8031    |
| gemma3-4B   | 4096       | 0.75         | 3072   | Shakespeare | 3.0992    | 22.1791    |
| gemma3-4B   | 4096       | 1.0          | 4096   | Shakespeare | 3.3238    | 27.7651    |



### **Llama3.2-1B/3B**
- 3B 比 1B 好，sure
- **Llama3.2 的 benchmark 非常好, overfitting dataset?**

| Model       | Block Size | Stride Ratio | Stride | Dataset     | Avg. Loss | Perplexity |
| ----------- | ---------- | ------------ | ------ | ----------- | --------- | ---------- |
| Llama3.2-1B | 2048       | 0.25         | 512    | Wikitext2   | 2.4417    | 11.4930    |
| Llama3.2-1B | 2048       | 0.5          | 1024   | Wikitext2   | 2.4538    | 11.6328    |
| Llama3.2-1B | 2048       | 0.75         | 1536   | Wikitext2   | 2.4788    | 11.9275    |
| Llama3.2-1B | 2048       | 1.0          | 2048   | Wikitext2   | 2.5776    | 13.1659    |
| Llama3.2-1B | 4096       | 0.25         | 1024   | Wikitext2   | 2.4108    | 11.1432    |
| Llama3.2-1B | 4096       | 0.5          | 2048   | Wikitext2   | 2.4174    | 11.2166    |
| Llama3.2-1B | 4096       | 0.75         | 3072   | Wikitext2   | 2.4310    | 11.3702    |
| Llama3.2-1B | 4096       | 1.0          | 4096   | Wikitext2   | 2.4989    | 12.1688    |
| Llama3.2-1B | 2048       | 0.25         | 512    | PTB         | 3.0441    | 20.9920    |
| Llama3.2-1B | 2048       | 0.5          | 1024   | PTB         | 3.0510    | 21.1357    |
| Llama3.2-1B | 2048       | 0.75         | 1536   | PTB         | 3.0615    | 21.3600    |
| Llama3.2-1B | 2048       | 1.0          | 2048   | PTB         | 3.1478    | 23.2859    |
| Llama3.2-1B | 4096       | 0.25         | 1024   | PTB         | 3.0234    | 20.5601    |
| Llama3.2-1B | 4096       | 0.5          | 2048   | PTB         | 3.0281    | 20.6582    |
| Llama3.2-1B | 4096       | 0.75         | 3072   | PTB         | 3.0340    | 20.7810    |
| Llama3.2-1B | 4096       | 1.0          | 4096   | PTB         | 3.0914    | 22.0071    |
| Llama3.2-1B | 2048       | 0.25         | 512    | Shakespeare | 3.3989    | 29.9325    |
| Llama3.2-1B | 2048       | 0.5          | 1024   | Shakespeare | 3.4064    | 30.1558    |
| Llama3.2-1B | 2048       | 0.75         | 1536   | Shakespeare | 3.4175    | 30.4927    |
| Llama3.2-1B | 2048       | 1.0          | 2048   | Shakespeare | 3.4764    | 32.3436    |
| Llama3.2-1B | 4096       | 0.25         | 1024   | Shakespeare | 3.3793    | 29.3515    |
| Llama3.2-1B | 4096       | 0.5          | 2048   | Shakespeare | 3.3829    | 29.4567    |
| Llama3.2-1B | 4096       | 0.75         | 3072   | Shakespeare | 3.3912    | 29.7030    |
| Llama3.2-1B | 4096       | 1.0          | 4096   | Shakespeare | 3.4291    | 30.8492    |
| Llama3.2-3B | 2048       | 0.25         | 512    | Wikitext2   | 1.9370    | 6.9377     |
| Llama3.2-3B | 2048       | 0.5          | 1024   | Wikitext2   | 1.9473    | 7.0097     |
| Llama3.2-3B | 2048       | 0.75         | 1536   | Wikitext2   | 1.9674    | 7.1519     |
| Llama3.2-3B | 2048       | 1.0          | 2048   | Wikitext2   | 2.0560    | 7.8144     |
| Llama3.2-3B | 4096       | 0.25         | 1024   | Wikitext2   | 1.9099    | 6.7523     |
| Llama3.2-3B | 4096       | 0.5          | 2048   | Wikitext2   | 1.9159    | 6.7931     |
| Llama3.2-3B | 4096       | 0.75         | 3072   | Wikitext2   | 1.9275    | 6.8721     |
| Llama3.2-3B | 4096       | 1.0          | 4096   | Wikitext2   | 1.9869    | 7.2926     |
| Llama3.2-3B | 2048       | 0.25         | 512    | PTB         | 2.3731    | 10.7304    |
| Llama3.2-3B | 2048       | 0.5          | 1024   | PTB         | 2.3793    | 10.7968    |
| Llama3.2-3B | 2048       | 0.75         | 1536   | PTB         | 2.3923    | 10.9384    |
| Llama3.2-3B | 2048       | 1.0          | 2048   | PTB         | 2.4849    | 12.0005    |
| Llama3.2-3B | 4096       | 0.25         | 1024   | PTB         | 2.3510    | 10.4960    |
| Llama3.2-3B | 4096       | 0.5          | 2048   | PTB         | 2.3559    | 10.5474    |
| Llama3.2-3B | 4096       | 0.75         | 3072   | PTB         | 2.3626    | 10.6188    |
| Llama3.2-3B | 4096       | 1.0          | 4096   | PTB         | 2.4225    | 11.2740    |
| Llama3.2-3B | 2048       | 0.25         | 512    | Shakespeare | 2.1525    | 8.6066     |
| Llama3.2-3B | 2048       | 0.5          | 1024   | Shakespeare | 2.1582    | 8.6555     |
| Llama3.2-3B | 2048       | 0.75         | 1536   | Shakespeare | 2.1667    | 8.7292     |
| Llama3.2-3B | 2048       | 1.0          | 2048   | Shakespeare | 2.2319    | 9.3180     |
| Llama3.2-3B | 4096       | 0.25         | 1024   | Shakespeare | 2.1371    | 8.4751     |
| Llama3.2-3B | 4096       | 0.5          | 2048   | Shakespeare | 2.1402    | 8.5015     |
| Llama3.2-3B | 4096       | 0.75         | 3072   | Shakespeare | 2.1471    | 8.5602     |
| Llama3.2-3B | 4096       | 1.0          | 4096   | Shakespeare | 2.1851    | 8.8914     |



### **Gemma3-1B/4B**
- 4B 比 1B 好，sure
- **4B PTB 比 1B 差?  而且 block size 大反而更差？overlap 多反而差？非常怪！！！**

| Model     | Block Size | Stride Ratio | Stride | Dataset     | Avg. Loss | Perplexity   |
| --------- | ---------- | ------------ | ------ | ----------- | --------- | ------------ |
| gemma3-1B | 2048       | 0.25         | 512    | Wikitext2   | 3.0110    | 20.3076      |
| gemma3-1B | 2048       | 0.5          | 1024   | Wikitext2   | 3.0371    | 20.8442      |
| gemma3-1B | 2048       | 0.75         | 1536   | Wikitext2   | 3.0857    | 21.8827      |
| gemma3-1B | 2048       | 1.0          | 2048   | Wikitext2   | 3.3237    | 27.7621      |
| gemma3-1B | 4096       | 0.25         | 1024   | Wikitext2   | 2.9472    | 19.0527      |
| gemma3-1B | 4096       | 0.5          | 2048   | Wikitext2   | 2.9597    | 19.2915      |
| gemma3-1B | 4096       | 0.75         | 3072   | Wikitext2   | 2.9830    | 19.7479      |
| gemma3-1B | 4096       | 1.0          | 4096   | Wikitext2   | 3.1439    | 23.1936      |
| gemma3-1B | 1024       | 0.25         | 256    | PTB         | 4.8153    | 123.3813     |
| gemma3-1B | 1024       | 0.5          | 512    | PTB         | 4.8549    | 128.3633     |
| gemma3-1B | 1024       | 0.75         | 768    | PTB         | 4.9333    | 138.8411     |
| gemma3-1B | 1024       | 1.0          | 1024   | PTB         | 5.2248    | 185.8153     |
| gemma3-1B | 2048       | 0.25         | 512    | PTB         | 4.6951    | 109.4108     |
| gemma3-1B | 2048       | 0.5          | 1024   | PTB         | 4.7198    | 112.1477     |
| gemma3-1B | 2048       | 0.75         | 1536   | PTB         | 4.7643    | 117.2445     |
| gemma3-1B | 2048       | 1.0          | 2048   | PTB         | 4.9686    | 143.8289     |
| gemma3-1B | 4096       | 0.25         | 1024   | PTB         | 4.6413    | 103.6824     |
| gemma3-1B | 4096       | 0.5          | 2048   | PTB         | 4.6524    | 104.8413     |
| gemma3-1B | 4096       | 0.75         | 3072   | PTB         | 4.6651    | 106.1757     |
| gemma3-1B | 4096       | 1.0          | 4096   | PTB         | 4.8058    | 122.2155     |
| gemma3-1B | 2048       | 0.25         | 512    | Shakespeare | 3.7388    | 42.0463      |
| gemma3-1B | 2048       | 0.5          | 1024   | Shakespeare | 3.7544    | 42.7088      |
| gemma3-1B | 2048       | 0.75         | 1536   | Shakespeare | 3.7827    | 43.9366      |
| gemma3-1B | 2048       | 1.0          | 2048   | Shakespeare | 3.9603    | 52.4706      |
| gemma3-1B | 4096       | 0.25         | 1024   | Shakespeare | 3.6929    | 40.1608      |
| gemma3-1B | 4096       | 0.5          | 2048   | Shakespeare | 3.7046    | 40.6352      |
| gemma3-1B | 4096       | 0.75         | 3072   | Shakespeare | 3.7196    | 41.2470      |
| gemma3-1B | 4096       | 1.0          | 4096   | Shakespeare | 3.8318    | 46.1474      |
| gemma3-4B | 2048       | 0.25         | 512    | Wikitext2   | 2.3640    | 10.6331      |
| gemma3-4B | 2048       | 0.5          | 1024   | Wikitext2   | 2.3911    | 10.9254      |
| gemma3-4B | 2048       | 0.75         | 1536   | Wikitext2   | 2.4596    | 11.6997      |
| gemma3-4B | 2048       | 1.0          | 2048   | Wikitext2   | 2.8554    | 17.3807      |
| gemma3-4B | 4096       | 0.25         | 1024   | Wikitext2   | 2.3234    | 10.2104      |
| gemma3-4B | 4096       | 0.5          | 2048   | Wikitext2   | 2.3298    | 10.2764      |
| gemma3-4B | 4096       | 0.75         | 3072   | Wikitext2   | 2.3500    | 10.4853      |
| gemma3-4B | 4096       | 1.0          | 4096   | Wikitext2   | 2.6017    | 13.4865      |
| gemma3-4B | 1024       | 0.25         | 256    | PTB         | 5.4340    | 229.0644     |
| gemma3-4B | 1024       | 0.5          | 512    | PTB         | 5.2521    | 190.9586     |
| gemma3-4B | 1024       | 0.75         | 768    | PTB         | 5.2035    | 181.9126     |
| gemma3-4B | 1024       | 1.0          | 1024   | PTB         | 5.5823    | 265.6782     |
| gemma3-4B | 2048       | 0.25         | 512    | PTB         | 6.0464    | **422.6029** |
| gemma3-4B | 2048       | 0.5          | 1024   | PTB         | 5.9746    | **393.3032** |
| gemma3-4B | 2048       | 0.75         | 1536   | PTB         | 5.6783    | **292.4377** |
| gemma3-4B | 2048       | 1.0          | 2048   | PTB         | 5.8874    | **360.4721** |
| gemma3-4B | 4096       | 0.25         | 1024   | PTB         | 6.2515    | **518.8036** |
| gemma3-4B | 4096       | 0.5          | 2048   | PTB         | 6.2290    | **507.2570** |
| gemma3-4B | 4096       | 0.75         | 3072   | PTB         | 6.1217    | **455.6433** |
| gemma3-4B | 4096       | 1.0          | 4096   | PTB         | 6.0715    | **433.3349** |
| gemma3-4B | 2048       | 0.25         | 512    | Shakespeare | 3.1077    | 22.3704      |
| gemma3-4B | 2048       | 0.5          | 1024   | Shakespeare | 3.1265    | 22.7944      |
| gemma3-4B | 2048       | 0.75         | 1536   | Shakespeare | 3.2014    | 24.5672      |
| gemma3-4B | 2048       | 1.0          | 2048   | Shakespeare | 3.5761    | 35.7343      |
| gemma3-4B | 4096       | 0.25         | 1024   | Shakespeare | 3.0754    | 21.6576      |
| gemma3-4B | 4096       | 0.5          | 2048   | Shakespeare | 3.0821    | 21.8031      |
| gemma3-4B | 4096       | 0.75         | 3072   | Shakespeare | 3.0992    | 22.1791      |
| gemma3-4B | 4096       | 1.0          | 4096   | Shakespeare | 3.3238    | 27.7651      |


### **Phi3 vs. Ph4 3.8B**
- Phi3 比 Phi4 好，Wiki2/PTB/Shakespeare: 5/9/11 vs. 8/14/19.  **Phi3 overfitting dataset?**

| Model        | Block Size | Stride Ratio | Stride | Dataset     | Avg. Loss | Perplexity |
| ------------ | ---------- | ------------ | ------ | ----------- | --------- | ---------- |
| Phi3-mini-4k | 2048       | 0.25         | 512    | Wikitext2   | 1.6759    | 5.3435     |
| Phi3-mini-4k | 2048       | 0.5          | 1024   | Wikitext2   | 1.6848    | 5.3913     |
| Phi3-mini-4k | 2048       | 0.75         | 1536   | Wikitext2   | 1.7045    | 5.4987     |
| Phi3-mini-4k | 2048       | 1.0          | 2048   | Wikitext2   | 1.7927    | 6.0059     |
| Phi3-mini-4k | 4096       | 0.25         | 1024   | Wikitext2   | 1.6682    | 5.3024     |
| Phi3-mini-4k | 4096       | 0.5          | 2048   | Wikitext2   | 1.6682    | 5.3028     |
| Phi3-mini-4k | 4096       | 0.75         | 3072   | Wikitext2   | 1.6746    | 5.3365     |
| Phi3-mini-4k | 4096       | 1.0          | 4096   | Wikitext2   | 1.7290    | 5.6351     |
| Phi3-mini-4k | 2048       | 0.25         | 512    | PTB         | 2.2739    | 9.7176     |
| Phi3-mini-4k | 2048       | 0.5          | 1024   | PTB         | 2.2808    | 9.7847     |
| Phi3-mini-4k | 2048       | 0.75         | 1536   | PTB         | 2.2948    | 9.9227     |
| Phi3-mini-4k | 2048       | 1.0          | 2048   | PTB         | 2.3957    | 10.9758    |
| Phi3-mini-4k | 4096       | 0.25         | 1024   | PTB         | 2.2695    | 9.6746     |
| Phi3-mini-4k | 4096       | 0.5          | 2048   | PTB         | 2.2692    | 9.6716     |
| Phi3-mini-4k | 4096       | 0.75         | 3072   | PTB         | 2.2741    | 9.7188     |
| Phi3-mini-4k | 4096       | 1.0          | 4096   | PTB         | 2.3326    | 10.3046    |
| Phi3-mini-4k | 2048       | 0.25         | 512    | Shakespeare | 2.4457    | 11.5387    |
| Phi3-mini-4k | 2048       | 0.5          | 1024   | Shakespeare | 2.4497    | 11.5845    |
| Phi3-mini-4k | 2048       | 0.75         | 1536   | Shakespeare | 2.4555    | 11.6520    |
| Phi3-mini-4k | 2048       | 1.0          | 2048   | Shakespeare | 2.5045    | 12.2370    |
| Phi3-mini-4k | 4096       | 0.25         | 1024   | Shakespeare | 2.4427    | 11.5041    |
| Phi3-mini-4k | 4096       | 0.5          | 2048   | Shakespeare | 2.4428    | 11.5051    |
| Phi3-mini-4k | 4096       | 0.75         | 3072   | Shakespeare | 2.4457    | 11.5382    |
| Phi3-mini-4k | 4096       | 1.0          | 4096   | Shakespeare | 2.4737    | 11.8662    |
| Phi4-mini    | 2048       | 0.25         | 512    | Wikitext2   | 2.1263    | 8.3834     |
| Phi4-mini    | 2048       | 0.5          | 1024   | Wikitext2   | 2.1360    | 8.4658     |
| Phi4-mini    | 2048       | 0.75         | 1536   | Wikitext2   | 2.1576    | 8.6504     |
| Phi4-mini    | 2048       | 1.0          | 2048   | Wikitext2   | 2.2523    | 9.5096     |
| Phi4-mini    | 4096       | 0.25         | 1024   | Wikitext2   | 2.1025    | 8.1862     |
| Phi4-mini    | 4096       | 0.5          | 2048   | Wikitext2   | 2.1059    | 8.2147     |
| Phi4-mini    | 4096       | 0.75         | 3072   | Wikitext2   | 2.1142    | 8.2830     |
| Phi4-mini    | 4096       | 1.0          | 4096   | Wikitext2   | 2.1809    | 8.8542     |
| Phi4-mini    | 2048       | 0.25         | 512    | PTB         | 2.6570    | 14.2536    |
| Phi4-mini    | 2048       | 0.5          | 1024   | PTB         | 2.6640    | 14.3539    |
| Phi4-mini    | 2048       | 0.75         | 1536   | PTB         | 2.6750    | 14.5120    |
| Phi4-mini    | 2048       | 1.0          | 2048   | PTB         | 2.7733    | 16.0114    |
| Phi4-mini    | 4096       | 0.25         | 1024   | PTB         | 2.6444    | 14.0746    |
| Phi4-mini    | 4096       | 0.5          | 2048   | PTB         | 2.6464    | 14.1038    |
| Phi4-mini    | 4096       | 0.75         | 3072   | PTB         | 2.6509    | 14.1672    |
| Phi4-mini    | 4096       | 1.0          | 4096   | PTB         | 2.7078    | 14.9960    |
| Phi4-mini    | 2048       | 0.25         | 512    | Shakespeare | 2.9523    | 19.1509    |
| Phi4-mini    | 2048       | 0.5          | 1024   | Shakespeare | 2.9561    | 19.2232    |
| Phi4-mini    | 2048       | 0.75         | 1536   | Shakespeare | 2.9633    | 19.3611    |
| Phi4-mini    | 2048       | 1.0          | 2048   | Shakespeare | 3.0186    | 20.4623    |
| Phi4-mini    | 4096       | 0.25         | 1024   | Shakespeare | 2.9470    | 19.0493    |
| Phi4-mini    | 4096       | 0.5          | 2048   | Shakespeare | 2.9474    | 19.0561    |
| Phi4-mini    | 4096       | 0.75         | 3072   | Shakespeare | 2.9511    | 19.1272    |
| Phi4-mini    | 4096       | 1.0          | 4096   | Shakespeare | 2.9838    | 19.7629    |


    
## Appendix

#### **LLaMa-1B of different block size (with 0.5 stride ratio)**

| Dataset     | block_size | stride | Average Loss | Perplexity |
| ----------- | ---------- | ------ | ------------ | ---------- |
| WikiText2   | 1024       | 512    | 2.5270       | 12.5164    |
| WikiText2   | 2048       | 1024   | 2.4538       | 11.6328    |
| WikiText2   | 4096       | 2048   | 2.4174       | 11.2166    |
| PTB         | 1024       | 512    | 3.1850       | 24.1678    |
| PTB         | 2048       | 1024   | 3.1500       | 23.3357    |
| PTB         | 4096       | 2048   | 3.1290       | 22.8516    |
| Shakespeare | 1024       | 512    | 3.4556       | 31.6784    |
| Shakespeare | 2048       | 1024   | 3.4228       | 30.6553    |
| Shakespeare | 4096       | 2048   | 3.3991       | 29.9363    |


| **Model**            | ---                                | Pre-  | Train  | ---  | ---              | Fine- | Tune | ---   |
| -------------------- | ---------------------------------- | ----- | ------ | ---- | ---------------- | ----- | ---- | ----- |
|                      | Length                             | Batch | Loss   | PPL  | Length           | Batch | Loss | PPL   |
| GPT-2 (124M)         | 1024                               | 4/70  | 3.36   | 28.7 | 1024             | 8/31  | 3.34 | 28.3  |
|                      |                                    |       |        |      | variable         |       | 0.23 | *1.3* |
|                      |                                    |       |        |      | pad to 1024      | 8/470 | 0.96 | 2.6   |
|                      |                                    |       |        |      | pad to batch max | 8/470 | 1.82 | 6.3   |
| -- use hugging       |                                    |       | 3.23   | 25.2 |                  |       |      |       |
| GPT2-large (774M)    | 1024/st=0.5                        |       | 2.8    | 16.4 |                  |       |      |       |
| LLaMA-1B             | 1024/st=0                          | 4/71  | 2.69   | 14.7 |                  |       |      |       |
|                      | st_ratio=0.5                       | 1/?   | 2.57   | 12.5 |                  |       |      |       |
| --                   | 2048                               | 2/71  | 2.56   | 13.0 |                  |       |      |       |
|                      | st_ratio=0.5                       | 1/?   | 2.45   | 11.6 |                  |       |      |       |
| --                   | 4096                               | 1/71  | 2.5    | 12.1 |                  |       |      |       |
|                      | st_ratio=0.5                       | 1/?   | 2.41   | 11.2 |                  |       |      |       |
| LLaMA-3B             | 1024                               | 4/71  | 2.1    | 8.5  |                  |       |      |       |
| --                   | 2048                               | 2/71  | 2.0    | 7.6  |                  |       |      |       |
| --                   | 4096                               | 1/71  | 1.96   | 7.1  |                  |       |      |       |
| Phi3-mini-3.8B       | 1024                               | 4/82  | 1.93   | 6.9  |                  |       |      |       |
| --                   | 2048                               | 2/82  | 1.82   | 6.2  |                  |       |      |       |
| --                   | 4096                               | 1/82  | 1.75   | 5.8  |                  |       |      |       |
| Gemma-7B             | 1024                               | 4/72  | 5.88?? |      |                  |       |      |       |
| Gemma7B              | 2048                               | 4/36  | --     |      |                  |       |      |       |
| Gemma7B              | 2048                               | 2/72  | 4.6    |      |                  |       |      |       |
|                      |                                    |       |        |      |                  |       |      |       |
| 2025/7/9 use hugging |                                    |       |        |      |                  |       |      |       |
| GPT2-large (774M)    | 1024<br>/stride=512                |       | 2.8    | 16.4 | gpt2perp.ipynb   |       |      |       |
|                      | 1024<br>/stride=1024<br>no overlap |       |        | 19.4 |                  |       |      |       |
| GPT2 (124M)          | 1024<br>/stride=512                |       | 3.22   | 25.2 | gpt2perp.py      |       |      |       |
| GPT2 (124M)          | 1024<br>/stride=1024<br>no overlap |       | 3.4    | 29.9 | gpt2perp.py      |       |      |       |
|                      |                                    |       |        |      |                  |       |      |       |




## Source ChatGPT

計算 perplexity, 一般用 WikiText-2 因為比較小，品質比較好。

### **Comparison Table**

|**Feature**|**WikiText-2**|**WikiText-103**|**enwik8**|
|---|---|---|---|
|**Size**|~2M tokens|~103M tokens|~100M characters|
|**Vocabulary Size**|~33,000 tokens|~267,000 tokens|N/A (raw character-level)|
|**Preprocessing**|Minimal|Minimal|None (includes raw text)|
|**Task Focus**|Word-level modeling|Word-level modeling|Character-level modeling|
|**Use Cases**|Small-scale experiments|Large-scale pretraining|Byte/character-level tasks|
|**Computational Cost**|Low|High|Moderate|

Here is a summary table for **gpt2-xl** including both model metadata and your recent evaluation metrics:

## GPT-2 XL Model Specification

|Property|Value|
|---|---|
|Hidden Size|1600|
|Layers|48|
|Context Window|1024|
|Vocab Size|50,257|
|Parameters|1.5 billion|
|File Size (safetensors)|~6.4 GB|

## gpt2-xl Evaluation Results

| Model   | Block Size | Stride Ratio | Stride | Dataset     | Avg. Loss | Perplexity |
| ------- | ---------- | ------------ | ------ | ----------- | --------- | ---------- |
| gpt2-xl | 1024       | 0.25         | 256    | Wikitext2   | 2.6737    | 14.4940    |
| gpt2-xl | 1024       | 0.5          | 512    | Wikitext2   | 2.6938    | 14.7878    |
| gpt2-xl | 1024       | 0.75         | 768    | Wikitext2   | 2.7243    | 15.2461    |
| gpt2-xl | 1024       | 1.0          | 1024   | Wikitext2   | 2.8565    | 17.3997    |
| gpt2-xl | 1024       | 0.25         | 256    | PTB         | 2.9238    | 18.6115    |
| gpt2-xl | 1024       | 0.5          | 512    | PTB         | 2.9395    | 18.9063    |
| gpt2-xl | 1024       | 0.75         | 768    | PTB         | 2.9688    | 19.4680    |
| gpt2-xl | 1024       | 1.0          | 1024   | PTB         | 3.1337    | 22.9577    |
| gpt2-xl | 1024       | 0.25         | 256    | Shakespeare | 3.7458    | 42.3414    |
| gpt2-xl | 1024       | 0.5          | 512    | Shakespeare | 3.7567    | 42.8064    |
| gpt2-xl | 1024       | 0.75         | 768    | Shakespeare | 3.7705    | 43.4018    |
| gpt2-xl | 1024       | 1.0          | 1024   | Shakespeare | 3.8218    | 45.6877    |

**Notes:**

- For gpt2-xl, block size cannot exceed model max_length (1024).
    
- Stride is calculated as `block_size * stride_ratio`.
    
- Settings with block_size greater than 1024 are skipped due to model constraints.
    

Let me know if you want to see comparison with other models!
