---
title: Transformer Layer Normalization Placement
date: 2024-10-04 14:10:08
description: LLM Tokenizer
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-02-14-ML_Normalization]]"
categories:
  - Science
tags:
  - Normalization
  - Transformer
---


## Takeaway

- Pre-LN does not rely on the learning rate warm-up stage and can be trained much faster than the Post-LN.
- Pre-LN更容易训练好理解，因为它的恒等路径更突出，但为什么它效果反而没那么好呢?
- 因爲 Pre-LN 的深度有“水分”！也就是说，一个 L 层的 Pre-LN 模型，其实际等效层数不如 L 层的Post-LN 模型，而层数少了导致效果变差了。

## Background

自然語言處理任務中，經常使用的是Layer Normalization(LN)而非Batch Normalization(BN)，關於兩者具體介紹可以參考 [[2024-1003-Xformer_Normalization]].

在隨機優化理論中，學習率往往設置爲**常數**或者**逐漸衰減** (decay)，從而保證算法收斂，這種學習率的設置方法也與機器學習裏很多任務上的實際經驗類似。然而，不管是設置學習率爲常數還是使學習率逐漸衰減都不能讓Transformer很好地收斂。

在優化Transformer結構時，除了設置初始學習率與它的衰減策略，往往還需要在訓練的初始階段設置一個非常小（接近0）的學習率，讓它經過一定的迭代輪數後逐漸增長到初始的學習率，這個過程稱作**warm-up階段（學習率預熱）**。

Warm-up是原始Transformer結構優化時的一個必備學習率調整策略。Transformer結構對於warm-up的超參數（**持續輪數、增長方式、初始學習率**等）非常敏感，若調整不慎，往往會使得模型無法正常收斂。

<img src="/media/image-20241003222646.png" alt="20241003222646" style="zoom:60%;" />
Transformer結構的優化非常困難，其具體表現在：

- **Warm-up階段超參數敏感；**
- **優化過程收斂速度慢。**

## Layer Normalization Placement

"Attention is All You Need"  採用的 layer normalization 稱爲 post-LN.  Post-LN approach has been widely adopted in models like BERT and the original Transformer model proposed by Vaswani et al. (2017).

<img src="/media/image-20241003194932.png" alt="20241003194932" style="zoom:60%;" />  


因爲是在 Add residue shortcut 的後面，因此稱爲 Post-LN，如下圖左。 
相反的稱爲 Pre-LN, 如下圖右。

<img src="/media/image-20241003194622.png" alt="20241003194622" style="zoom:60%;" />

### Post-LN 和 Pre-LN 的比較

我們先用數學表示：

Pre Norm: $\quad \boldsymbol{x}_{t+1}=\boldsymbol{x}_t+F_t\left(\operatorname{Norm}\left(\boldsymbol{x}_t\right)\right)$
Post Norm: $\quad \boldsymbol{x}_{t+1}=\operatorname{Norm}\left(\boldsymbol{x}_t+F_t\left(\boldsymbol{x}_t\right)\right)$

注意
- 這裏的 Norm 可以是 Layer normalization 或是其他的 Normalization. 
- 這裏的 $F_t$  包含:  attention net (ATTN) 以及 feed-forward net (FFN) 


落實到 code:

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    # Pre Norm implementation
    def forward_preLN(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    # Post Norm implementation
    def forward_postLN(self,x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x
```


#### Post-LN

Advantage 
- Save the bias of the Multi-Head Attention and FFN because of the Layer Norm reset the bias.
- Better accuracy, check the reference.

Disadvantage
- Just look at the diagram, Layer Normalizations block the way.   It suffers from gradient stability issue at initialization. 
- Post-LN also needs to do learning rate warm-up, which can initially slow down convergence but helps stabilize the training process by gradually increasing the learning rate from a small value to a target value over several iterations.  Not sure if this is related to the Layer Normalizations block the way.

#### Pre-LN

Advantage
- The gradients are better behaved at initialization, reducing issues related to vanishing or exploding gradients
- This architecture can eliminate the need for a learning rate warm-up stage, potentially speeding up convergence during training

Disadvantage
- The performance is worse compared to Post-LN


**Pre Norm更容易训练好理解，因为它的恒等路径更突出，但为什么它效果反而没那么好呢？**
**Pre Norm的深度有“水分”！也就是说，一个LL层的Pre Norm模型，其实际等效层数不如LL层的Post Norm模型，而层数少了导致效果变差了。**

Pre Norm结构无形地增加了模型的宽度而降低了模型的深度，而我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。而Post Norm刚刚相反，它每Norm一次就削弱一次恒等分支的权重，所以Post Norm反而是更突出残差分支的，因此Post Norm中的层数更加“足秤”，一旦训练好之后效果更优。

#### Summary of Differences

| Feature                          | Post-LN                                       | Pre-LN                                      |
| -------------------------------- | --------------------------------------------- | ------------------------------------------- |
| **Layer Normalization Position** | After residual connection                     | Before residual connection                  |
| **Gradient Behavior**            | May lead to large gradients at initialization | Better gradient stability at initialization |
| **Learning Rate Warm-Up**        | Often required for stable training            | Can be omitted, leading to faster training  |
| **Linear Layer Bias**            | Can save some biases redundent with LN        | May need more bias                          |
| **Model accuracy**               | Better                                        | Worse, more close to identity network       |
| **Overall Assessment**           | Difficult to train, but better accuracy       | Easy to train, but worse in accuracy        |

In conclusion, while both Post-Layer Normalization and Pre-Layer Normalization have their respective advantages, recent research suggests that Pre-LN may offer improved stability and efficiency during training, making it a compelling choice for modern transformer architectures[


## DeepNorm

000层的Transformer庞然大物来了。

当然，这篇文章充分利用了Post-LN和Pre-LN的各自优点，改造出了**DeepNorm**的归一化方案，用于支撑深层模型的训练。

Nguyen和Salazar(2019)发现相对于Post-LN，Pre-LN能够提升Transformer的稳定性。然而，Pre-LN在底层的梯度往往大于顶层，导致其性能不及 Post-LN。为了缓解这一问题，研究人员一直努力通过更好的初始化方式或更好的模型架构来改进 transformer 的深度。这些方法可以使多达数百层的Transformer 模型实现稳定化，然而以往的方法没有能够成功地扩展至1000层。

**DeepNorm** 结合了 Post-LN 的良好性能以及 Pre-LN 的训练稳定性。与Post-LN 相比，DeepNorm 在执行层归一化之前 Up-Scale了残差连接。

樣子看起來像 Post-LN

```python
def deepnorm(x, alpha):
    return nn.LayerNorm(x * alpha + f(x))

def deepnorm_init(w, beta):
    if w in ['ffn', 'v_proj', 'out_proj']:
        nn.init.xavier_normal_(w, gain=beta)
    elif w in ['q_proj', 'k_proj']:
        nn.init.xavier_normal_(w, gain=1)
```



## Reference

- Layer Normalization Placement in the Transformer Architecture https://arxiv.org/abs/2002.04745
- 科學空間：[为什么Pre Norm的效果不如Post Norm？](https://kexue.fm/archives/9009) 
- https://zhuanlan.zhihu.com/p/480783670

