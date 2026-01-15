---
title: Perplexity of LLM
date: 2024-12-16 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2023-03-26-Transformer_LLM]]"
categories:
  - LLM
tags:
  - Attention
  - GPT
  - LLM
  - Perplexity
  - Transformer
---


## Source
- [(22) The Key Equation Behind Probability - YouTube](https://www.youtube.com/watch?v=KHVR587oW8I&t=810s)  a good youtube video of cross-entropy

Two things
1. check other pdf cascading!
2. check liger to increase model size
3. do scatter plot
4. check fine-tune result using Shakespeare
5. (done) check Shakespeare dataset batch 32-38 text -> 檢查是 Romeo and Juliet 的故事。應該是太有名被 training 到 3B model 中。
6. print the peak memory and throughput information

## Introduction

本文討論 perplexity，直接翻譯就是困惑度，名符其實。不過我認為 perplexity 是一個寶庫！

首先從一個 sample generator (with a pdf) 開始
- perplexity 就是 self-entropy 的 exponential.  物理意義是平均每個 token 有多少條“同樣機率”的路徑。
- Perplexity 和 information theory 的 AEP 概念一樣。不過 perplexity 是 normalized to 一個 token.   AEP 則是所有的 path, 隨著 sample exponential 增加。
- Perplexity 包含 theoretical view 和 empirical (sample) view.  Theoretical view 是 self-entropy 的 exponential.  Empirical view 則是用 samples 計算 self-entropy and perplexity.

再來是用一個 sample generator 去近似一個 dataset distribution (e.g. Wikitext2)。此時 perplexity 衡量的不是 sample generator 的 self-entropy exponential 或是 dataset 的 self-entropy exponential，**而是 cross-entropy 的 exponential.**  

- Theoretical view: model generator pdf: q(x1, x2, ..., xn). H(P, Q) = \int p (-log q).  p(x) 是 s1, s2, .. sn.  real probability.  x1, x2, .., xn 是 model generated probability.  
- Empirical view:  \sum 1 -log(q)_p_i

- - log p(x1, x2, ..., xn) = - log p(x1) p(x2 | x1) p(x3 | x1, x2) .... p(xn | x1, ... x(n-1)) = - sum log(x | x.)


所以 perplexity depends on (1) dataset self-entropy; (2) model capability to approach dataset; (3) 如何得到 joint pdf?  LLM decoder 目前是用 autoregression, 就是上公式展開的方式。


## Dataset self-entropy

這容易理解，以下從簡單到複雜

Text or sequence
- biased coin 加上 state diagram: 0, 0, .., 0, 1, 0, 0, ....
- Shakespeare dataset
- Wikitext2 dataset
- Pure random data: 0, 1, 1, 0, 0, 1, 0, ....

Figure/graph
-  Low details patch: 例如雲，水
- High details patch: 人, tree

Sentence encoding (BERT)
- dfddf




I find the issue!!!! of why definition of perplexity 2^H(P).   b Perplexity is a measure often used in information theory and natural language processing to evaluate the uncertainty or complexity of a probability distribution.

用一個例子，例如 Shakespear dataset.  e 的幾率已經是 > 1/26.  例如是 12422/23444.
我們在算 self-entropy, 可以用 :  (1) sum p1 log p1  這個 sum 是所有 {a, ..., z};  (2)  直接把 Shakespeare 的 dataset 的  (log num_a/all + log num_b / all  ) / all,  因爲 e 自動已經把比例算入了。





因爲實際的 sample 本身就已經包含 p(x)  所以 1/N sample 其實就是對應 p(x).   
因爲 sample 的本身就是 p(x),   所以不是 cross-entropy, 而是 self-entropy

## 瞭解 Perplexity

困惑度（PP）是一種用於評估語言模型的指標，表示模型對樣本的預測能力。它被定義為一個序列的指數化平均負對數似然。較低的困惑度值表示更好的預測性能。 PPL 永遠大於等於 1 以及小於等於 vocab_size.  可以視爲對下一個 token 的可能選擇預估。如果是 1 代表非常確定。越大代表把握度越低。


## Perplexity 的理論定義 = exponential entropy

**Distribution View**:  非常簡單，就是 exponential entropy.   
注意 entropy 和 base=2 或是 base=e 有關，但是 perplexity 和 base 無關。不過一般用 base=e.
  $$
  \text{PP}(p)
  \;=\;
  2^{H(p)}.
  $$
 
$$
  H(p)
  \;=\;
  -\sum_{x}p(x)\,\log_2 p(x)
  \;=\;
  \mathbb{E}_{x\sim p}\bigl[-\log_2 p(x)\bigr].
  $$
  
白話文就是有多少個“相等機率”的結果。
- Fair coin (p=0.5):  $H(p=0.5) = 1$, $PP = 2^{H} = 2$.    有兩個“相等機率”的結果。
- Bias coin (p=0 or 1): $H(p=0 \text{ or } p=1) = 0$,  $PP = 2^{H} = 1$.   只有一個“相等機率”的結果。    
- Uniform distribution of $N$ outcomes:  $H = \log_2 N$,   $PP = 2^H = N$.   N 個“相等機率”的結果。

## Perplexity 的 Empirical 定義

理論定義的最大問題是我們並不知道 $p(x)$.   這其實也不是問題，因爲我們可以從 "samples" 推估 distribution 和 entropy!   我們用一個簡單的例子説明。

假如一個 biased coin (p=0.25)，產生 100 samples, 其中 24 個 1, 76 個 0.  
1. **純理論:**  p = 0.25,  H(p) =   0.25 * log(0.25) + 0.75 * log(0.75)
2. **純 samples:**  p = 25/(25+75) = 0.25,   H(p) = 0.25 * log(0.25) + 0.75 * log(0.75) = ...
3. **半 samples, 半理論:** $\mathbb{E}_{x\sim p}\bigl[-\log_2 p(x)\bigr] = \frac{1}{N} \Sigma^N f(x_n)$   where $f(x) = -\log_2 p(x)$  這裏用到一個觀念:  probability space 的 distribution 會直接反應在 time domain 的 sample distribution.   因此對時間做平均等於對機率分佈的期望值！ 一個問題是爲什麽要四不像？因爲在實務上，例如 LLM token distribution 剛好會有 p(x)！另外在很多機率分佈和 sample 轉換都會用到這個特性。



### Perplexity 在 LLM 的定義是:
$$
\operatorname{PP}(X)=\exp \left\{-\frac{1}{t} \sum_i^t \log p\left(x_i \mid x_{<i}\right)\right\}
$$

這裏 Perplexity 的定義就比較抽象： 對於一個 tokenized sequence $X = (x_0, x_1, ..., x_t)$ , perplexity is the exponentiated average negative log-likelihood of a sequence.  



### Formula
The formula for perplexity $PP$ is given by:

$$
PP = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(w_i | w_1, w_2, \ldots, w_{i-1})\right)
$$

Where:
- $N$ is the number of tokens in the sequence.
- $p(w_i | w_1, w_2, \ldots, w_{i-1})$ is the probability assigned by the model to the $i^{th}$ token given all previous tokens.


## Perplexity 和 Entropy, AEP 的關係

先把上式中的 $\Sigma \log p(w_i \vert w_1, \cdot, w_{i-1})$ 交換成爲 $\log \Pi \, p(w_i \vert w_1, \cdot, w_{i-1})$ 

$$
\begin{aligned}
\Pi \, p(w_i \vert w_1, \cdot, w_{i-1}) &= p(x_1) p(x_2 \| x_1) p(x_3 \| x_2, x_1) p(x_4 \| x_3, x_2, x_1) \cdots p(x_t \| x_{t-1}, \cdots, x_1) \\
&= p(x_1) \frac{p(x_1, x_2)}{p(x_1)} \frac{p(x_3, x_2, x_1)}{p(x_1, x_2)} \cdots 
= p(x_1, x_2, \ldots, x_t)
\end{aligned}
$$


* 接下來把 $-1/N$ 放入 $\log$  以及 $\exp$ 和 $\log$ 互相抵銷，PPL 是條件機率的幾何平均值的倒數 = $\sqrt[t]{p(x_1,\cdots,x_n)^{-1}}$.   因此 PPL 的最小值是 1.   對應所有條件機率 = 1.  當然實務上機率必然小於 1, 所以倒數大於 1.  除非 joint distribution 是 delta function，不然 PPL 一定大於 1.   如果 joint distribution 的 entropy 越大,  PPL 就越大。注意這裏的 joint distribution 是指 sequence sample 的機率。
* 上述是用 natural log 和 exponential function 而不是用 2 為底。不過這完全沒有問題，應爲 log 的 exp 互相抵消。所以底是 2 或是 e 沒有差別。


Below is a clarification that distinguishes between the **common, empirical definition** of perplexity (as used in practice on a finite test set) and the **theoretical definition** of perplexity in terms of the true distribution’s entropy. The key point is that, in practice, we almost always talk about perplexity on a **finite sample** (test set), not on the entire distribution directly.

---

## 1. The Common (Empirical) Definition of Perplexity

Suppose you have a probability model $p$ and a **test set** (or evaluation set) of $N$ observations:
$$
x_1, x_2, \dots, x_N.
$$

In language modeling, for instance, each $x_i$ might be a word or a token in the test corpus. The **common, empirical definition** of perplexity is

$$
\text{Perplexity}(p; x_{1:N})
\;=\;
2^{
-\frac{1}{N}\,\sum_{i=1}^N \log_{2}\bigl[p(x_i)\bigr]
}.
$$

Equivalently, using natural logs,

$$
\text{Perplexity}(p; x_{1:N})
\;=\;
\exp\!\Bigl(-\frac{1}{N}\,\sum_{i=1}^N \ln\bigl[p(x_i)\bigr]\Bigr).
$$

### Why this definition?
1. **Average negative log-likelihood:**  
   $-\log_2 p(x_i)$ measures how “surprising” it is for the model $p$ to see $x_i$. Averaging over the test set gives a measure of overall predictive quality.
2. **Exponentiate that average:**  
   Exponentiating the **mean** of these $-\log_2$ values puts it back on a more interpretable scale, akin to an “effective number of equally likely outcomes.”

---

## 2. The Relationship to Entropy

### 2.1. Population (Theoretical) View vs. Sample (Empirical) View

- **Sample (Empirical) View**:  
  In practice, we do not have access to the entire distribution; we just have a **finite test set** $\{x_1, \ldots, x_N\}$. We replace the true expectation $\mathbb{E}[-\log_2 p(x)]$ by the **average** $\frac{1}{N}\sum_{i=1}^N -\log_2 p(x_i)$. That is how we get
  $$
  \text{Perplexity}(p; x_{1:N}) 
  \;=\;
  2^{\,-\frac{1}{N}\sum_{i=1}^N \log_2 p(x_i)}.
  $$

### 2.2. As $N \to \infty$

If the sample is large and representative, then

$$
-\frac{1}{N}\sum_{i=1}^N \log_2 p(x_i)
\;\approx\;
\mathbb{E}_{x\sim p}\bigl[-\log_2 p(x)\bigr]
\;=\;
H(p),
$$

which implies

$$
\text{Perplexity}(p; x_{1:N})
\;\approx\;
2^{H(p)}.
$$

So in the **limit of large $N$**, the **empirical perplexity** converges to the **theoretical** $2^{H(p)}$. That is the precise sense in which people say “Perplexity is $2^H$.”

---

## 3. Reconciling the Two Definitions

1. **Most common usage (empirical)**:  
   $$
   \boxed{
   \text{Perplexity on a test set}
   \;=\;
   2^{\;\displaystyle -\frac{1}{N}\,\sum_{i=1}^N \log_{2}[\,p(x_i)\,]}
   \quad
   (\text{common in practice}).
   }
   $$

2. **Theoretical limit (distribution-based)**:  
   $$
   \boxed{
   \text{Perplexity of a distribution}
   \;=\;
   2^{H(p)}
   \quad
   (\text{if you knew }p\text{ exactly}).
   }
   $$

They are consistent because the first becomes an estimator for the second when the sample is large enough and drawn from $p$.

---

## 4. Bottom Line

- **In textbooks and research papers**, when someone says “the perplexity of a language model is $2^{H}$,” they typically have in mind the **theoretical** expression that arises if the model were the true distribution.
- **In actual experiments**, we compute **empirical perplexity** on a finite test set using
  $$
  2^{
  \displaystyle -\frac{1}{N}\sum_{i=1}^N \log_2 p(x_i)
  }.
  $$
  This **estimates** $2^{H(p)}$ if $p$ is a good model of the data and $N$ is large.

That is why you sometimes see two slightly different formulas—one in terms of sums over a dataset (common in practice) and one in terms of an expectation and the entropy (common in theory). They match in the limit.

