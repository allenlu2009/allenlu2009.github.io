---
title: Perplexity of LLM
date: 2023-12-19 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

* X/Twitter: Akshay  https://twitter.com/akshay_pachaar/status/1728028273010188483

* [Perplexity of fixed-length models (huggingface.co)](https://huggingface.co/docs/transformers/perplexity)

* https://en.m.wikipedia.org/wiki/Perplexity

* https://www.youtube.com/watch?v=XfpMkf4rD6E&t=1395s&ab_channel=StanfordOnline



### Perplexity 定義 [Wiki]

在信息理论中，困惑度 (perplexity) 是对来自离散概率分布的样本值不确定性的度量。困惑度越大，观察者猜测从分布中抽取的值的可能性就越小。困惑度最早是在1977年由Frederick Jelinek、Robert Leroy Mercer、Lalit R. Bahl和James K. Bake在语音识别的背景下引入的。

因為熵 (entropy) 也是對於离散概率分布的样本值不确定性的度量。所以兩者之間應該有關係？答案是肯定的。



#### Perplexity of a (1 random variable) probability distribution

定義：Perplexity (PP) of a discrete probability distribution $p$ Is:
$$
PP(p):=2^{H(p)}=2^{-\sum_x p(x) \log _2 p(x)}=\prod_x p(x)^{-p(x)}
$$

* 其中 $H(p)$ 是 $p$ 的熵。 Perplexity 就是 2 的熵次方。**因爲離散分佈的熵 $H \ge 0$, 所以 perplexity $\ge$ 1.**
* Perplexity = 1 代表 deterministic, 沒有任何不確定性。
* 此處的 $\log$ 的 base 是 2.  改變 base 不影響 Perplexity.
* Entropy 越大，perplexity 越大。
* 一個 random variable $X$ 的 perplexity 是其 probability distribution 的 perplexity.
* Binary distribution (Bernoulli) 的 perplexity 最大值是 2.  $k$-value uniform distribution 有最大的 entropy 和 perplexity:  $H(p) = \log_2k$ and $PP(p) = 2^{H(p)} = k$.



* 以英文爲例
  * English text (26 letters and a space):  $H(X) = -(\log_2 1/27) = 4.75$ bits,   $PP(X) = 27$
  * 實際每個 letter 出現頻率/機率不同:  $H(X) = 4.219$ bits,   $PP(X) = 18.6$
  * 實際每個 letter 之間不是獨立。考慮 digrams, 或是 trigrams:  $H(X) = 2.77$ bits,   $PP(X) = 6.8$



#### Perplexity of n independent random variable probability distribution

一個 random variable 的 perplexity 顯然沒有什麼有趣的點。再來是多個 random variables or random sequence.

最簡單的 case 是 independent random sequence, $\{x_1, x_2, \cdots, x_n\}$.  

$H(X_1, X_2, ..X_n) = H(X_1) + H(X_2) + ... + H(Xn)$



**直觀的定義：** $PP(p(x_1, ...x_n)) = 2 ^ {H(X_1, X_2, .. X_n)} = 2 ^{n H(X)}$

Perplexity 有什麼物理意義？可以參考 typical set and AEP principle in Appendix A.

$N$ 個 binary variables 最多有 $2^N$ 個 possible outcomes.  或是 $2^N$ 路徑。

以上的 $N$ variable Perplexity $2^{n H(X)}$定義就是 equal probability outcome/path 的數目。 

* 如果 $H(X) = 0, PP(p(x_1, ...x_n)) = 1$,  只有唯一 outcome/path,  機率 = 1.
*  如果 $H(X) = 1, PP(p(x_1, ...x_n)) = 2^N$,  所有 outcome/path 都有同樣機率 $2^{-N}$.
* $H(X)$ 越接近 0,  $PP(p(x_1, ...x_n))$  約接近 1 outcome/path, 都有同樣機率。



直觀的定義有個問題就是 $N$,  一般還是會 normalized to N.   例如在 data compression 就是 $log_2 2^{H(X_1,...X_n)}/n = H(x) $

**修正的 (平均) perplexity 定義:**  $PP(p(x_1, ...x_n)) = 2 ^ {H(X_1, X_2, .. X_n)/n} = 2 ^{H(X)}$

好像太單純了?



#### Perplexity of n general random variable probability distribution, fixed-length language model

LLM 定義 perplexity 的問題是: 我們並不知道語言模型的 distribution, 稱爲 $p(x)$.   我們只有 sample sequence.   

Perplexity 在 LLM 的定義是:
$$
\operatorname{PP}(X)=\exp \left\{-\frac{1}{t} \sum_i^t \log p\left(x_i \mid x_{<i}\right)\right\}
$$

這裏 Perplexity 的定義就比較抽象： 對於一個 tokenized sequence $X = (x_0, x_1, ..., x_t)$ , perplexity is the exponentiated average negative log-likelihood of a sequence.  

以上式來看，log 的 summation and average 對應是**機率倒數的幾何平均值！**不是 entropy 的單位。

* PPL 内部的條件機率:
  $$p(x_1) p(x_2\|x_1) p(x_3\|x_2,x_1) p(x_4\|x_3,x_2,x_1)\cdots p(x_t \| x_{t-1},\cdots x_1) =  p(x_1) \frac{p(x_1,x_2)}{p(x_1)} \frac{p(x_3,x_2,x_1)}{p(x_1,x_2)} \cdots = p(x_1, x_2, ... x_t)$$

* 接下來 exp 和 log 互相抵銷，PPL 是條件機率的幾何平均值的倒數 = $\sqrt[t]{p(x_1,\cdots,x_t)^{-1}}$.   因此 PPL 的最小值是 1.   對應所有條件機率 = 1.  當然實務上機率必然小於 1, 所以倒數大於 1.  除非 joint distribution 是 delta function，不然 PPL 一定大於 1.   如果 joint distribution 的 entropy 越大,  PPL 就越大。注意這裏的 joint distribution 是指 sequence sample 的機率。
* 上述是用 natural log 和 exponential function 而不是用 2 為底。不過這完全沒有問題，應爲 log 的 exp 互相抵消。所以底是 2 或是 e 沒有差別。


* 上式指數部分:  $-\frac{1}{N} \sum_{i=1}^N \log  p\left(x_i\right)$  既不是和 N 成比例，也不是平均。但可以用 cross-entropy 解釋:

$$
H(\tilde{p}, p)=-\sum_x \tilde{p}(x) \log  p(x)
$$
​			where $\tilde{p}$ denotes the empirical distribution of the test sample (i.e., $\tilde{p}(x)=n / N$ if $x$ appeared $n$ times in the test sample of size $N$ ).

* Cross-entropy $H(\tilde{p},p)$ 的定義是 $\tilde{p}$ 和 $p$ 的接近程度。如果 $\tilde{p} = p$,  $H(\tilde{p},p) = H(p)$
* By the definition of $\mathrm{KL}$ divergence, it is also equal to

$$
H(\tilde{p})+D_{K L}(\tilde{p} \| p)
$$
​					which is $\geq H(\tilde{p})$. Consequently, the perplexity is minimized when $q=\tilde{p}$. 

* LLM perplexity 假設 $\tilde{p}(x) = 1/t$, **可以視爲 lower bound!**



* 如果 sequence 的每個 token 都是獨立:  $PP(X) = p(x)^{-1} \ne 2^{H(X)}$?  注意這和 uniform distribution 不同。待會我們看到。
* 如果所有的 token 都完全 depends on previous token:  $PP(X) = 1$ 



#### Appendix A: Typical Set and AEP Principle

這和 Information theory 的內容 : typical set $A_{\varepsilon}^n$ 的定羲如下:
$$
A_{\varepsilon}^n=\left\{\left(x_1, \cdots, x_n\right):\left|-\frac{1}{n} \log p\left(x_1, \cdots, x_n\right)-H(X)\right|<\varepsilon\right\} .
$$

- AEP 的 typical set 有 $\left|\mathcal{A}^n\right|=2^{n H(X)}$ sequences, 每一 sequence 都是 equal probability, i.e. $2^{-n H(x)}$, 
- 檢查一些特例:
  。 $p(H)=0$ (or 1) $\rightarrow H(X)=0, n H(X)=0$. 所以 $p(H H H \ldots H)=1$ or $p(T T \ldots T)=1$ ，其他 $2^n-1$ sequences 機率都是 0 . 因此 AEP 只有一條 sequence, $\left|\mathcal{A}^n\right|=2^{n H(x)}=1$.
  。 $p(H)=0.5 \rightarrow H(X)=1, n H(X)=n$. 所有 $2^n$ sequences 機率都是 $2^{-n} .\left|\mathcal{A}^n\right|=2^{n H(x)}=2^n$.
- AEP 所佔的機率, 當 $n \rightarrow \infty \mathrm{P}(\mathrm{AEP}) \rightarrow 1$.
- AEP 的䚐念是從統計力學而來, 在熱平衡態, 每一個分子的平均能量都是 $\frac{3}{2} k T$.

<img src="/media/image-20231219223300389.png" alt="image-20231219223300389" style="zoom:80%;" />
