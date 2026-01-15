#### Next

Q: Score function 的幾何或是物理意義是什麽?  D-log-likelihood function. 

Q: 爲什麽 Fisher information = var = -1 * 2nd order derivative?   

Q: 到底 variance 大比較好 estimate? (yes), 似乎 counter intuitive!!  Variance 小比較好 estimate?



#### Fisher Information vs. Shannon  Information 

簡單的想法就是做 n 次實驗看 (joint) distribution.  不過又會牽扯 n 要選多少。在理論推導是最好只使用原始的 distribution (n=1).

Fisher 引入 score function (1st derivative of log-likelihood function)

我們定義

假設 $\sigma$ 已知，score function 是 1 dimension linear function with negative slope as below:
$$
s(\mu;x_1, \ldots, x_n) =  l'\left(\mu, \sigma^2 ; x_1, \ldots, x_n\right)=\frac{1}{\sigma^2} \sum_{j=1}^n\left(x_j-\mu\right)
$$




Parameter estimation 的考量:

先用 observed 的 (X1, X2, X3, …, Xn) 用一個公式，估計出 θest (e.g. θest = A/n, or any other θest = g(X1, X2, .., Xn) )

再來用 θest 同時做無窮個平行的想像實驗的 parameter，找出對應的 θest'  "distribution" based on the above 公式。

(1) 如何估計 θ: θest ?  用估計的 θest  做無窮平行想像實驗所估計的 θest' 平均後應該要等於 (unbiased) 或趨近 θest,  就是 **constentency**.  (ref: Wifi[ statistics consistency](https://en.wikipedia.org/wiki/Consistency_(statistics)) or [Fisher consistency](https://en.wikipedia.org/wiki/Fisher_consistency))

(2) 這個估計的 θest 有多準確? 或是 θest 的 variance (or 信心區間) 有多大? 用 Fisher 的話來說，是否 **efficient.**  

(3) 這個 estimation θest 是否是 **optimal?**  如何定義 optimal?  可以從 (1) and (2) 來看，**optimal 就是必須 consistency 而且最 efficient (最小 variance).**

這 3 個觀念是環環相扣。如果 consistency 不成立。只要讓 θest = 0.  就會有最小的 variance. 也是最 optimla estimation. 但這是無用的 estimator.

因此 parameter estimation 最基本的觀念是 consistency.  就是用 estimated parameter 來作實驗，應該要得到一致性的結果。再加上有最好的 efficiency, 就是 optimal estimator.  

想像如果我們估計的 θest 比真實的 θo 小 (e.g. θest = A/2n instead of A/n = θo/2)。雖然機率小，還是有機會丟出 A 次 head.  

如果用這個 θest 同時做無窮個平行的 Bernoulli process with size n.  出現 head 次數的 distribution 就是 Bi-nomial distribution.  這個無窮平行想像實驗的 head 平均次數是 n*θest = A/2 次。從這些想像實驗估出的 θest' = (A/2)/2n = A/4n!

明顯 θest <> θest'  也就是不 consistent!

 

### Fisher 1922 論証 (under some assumption) maximum likelihood estimator (MLE) 具有 consistency, sufficiency 的 optimal parameter estimator.  (Fisher 認為 MLE 得出的 statistics are always sufficient statistics. 這個說法是有問題。請參考下一節的結論 "sufficiency …")

### Rao (1963) 証明了 MLE estimator is efficient (minimum variance) in the class of consistent and uniformly asymptotically normal (CUAN) estimator.

幾個常有的誤區:

\* MLE 並非唯一的 consistent estimator.  MLE 也並非 unbiased estimator.  無法從 (1) 推導出 MLE.  但可以証明 MLE 是 optimal estimator.

\* Fisher 定義的 efficiency 是基於 minimum variance, 也就是 least mean square error/loss.  這並非唯一的 loss function.

例如也可以用 L1-norm loss function.  不同的 loss function 對應不同的 optimal efficient parameter estimator.

 

### 如何証明 MLE 是 consist and optimal parameter estimator? 

Consistency 請參考 [reference](http://www.econ.uiuc.edu/~wsosa/econ507/MLE.pdf). 

Optimality 就不是那麼容易。

 

### Fisher 天才之處在1922 引入了 sufficient statistic 的觀念。1925 又引入了 Score and Efficiency 觀念

先說結論: (請參考 [Stigler](https://arxiv.org/pdf/0804.2996.pdf) 的 The Epic Story of MLE)

Sufficiency implies optimality (efficiency, or minimum variance), at least when combined with consistency and asymptotic normality.

Fisher 給了一個簡潔有力的論証關於以上的結論。請參考 Stigler article.  

In general 我們需要完整的 samples (X1, X2, …, Xn) 才能用於 parameter θ estimation.

Fisher 考慮是否能 "精練" 出 S(X1, X2, …, Xn) 針對 θ parameter, 就可以丟掉原來的 X1, X2, …, Xn 而不失去任何的統計資訊。切記這個 S(X1, X2, ..Xn), 稱為 sufficient statistic 必須是針對特定 θ (θ 可以是 vector, 包含多個參數); 而不是 sufficient statistic 可以任意取代原始的 samples 或 distribution.  如果真的如此，也不用 big data。只要存壓縮版的 sufficient statistic 就夠了。但我們永遠不知道日後會用原始資料來 estimate 什麼東西。

Fisher 的論証如下:

Maximum likelihood estimation -->  Function of sufficient statistic --> Optimality

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage49.png)

其中第一步是有問題的。Maximum likelihood estimation 並不 always 得出 sufficient statistic.  

Fisher 當初計算幾個常用的 distribution 而得出 sufficient statistic 存在並且可由 MLE 推出。

實際上很多 distribution with parameter 並不存在 sufficient statistics.  但 MLE 存在且為 optimal! 

之後 Fisher 也不再提出這個論証，而用其他的方法得到 MLE --> Optimality 的結論。

 

### Sufficient Statistic

### 雖然 Fisher 後來沒有再用 sufficient statistics 証明 MLE 的 optimality.

Sufficient statistics 仍然是有用的觀念。常見於一些理論的 framework.  (e.g. exponential family).

數學定義很直接如下。重點是可以推出 Fisher-Neyman factorization theorem.

例子和証明可以參考 wiki.

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage50.png)

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage51.png)

 

### Exponential Family

再來看 Exponential Family, 基本上就是 sufficient statistic 的最佳代言。

Exponential Family 包含大多常見的 PDF/PMF 如下:

the **[normal](https://en.wikipedia.org/wiki/Normal_distribution), [exponential](https://en.wikipedia.org/wiki/Exponential_distribution),[gamma](https://en.wikipedia.org/wiki/Gamma_distribution), [chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution), [beta](https://en.wikipedia.org/wiki/Beta_distribution), [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution), [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution), [categorical](https://en.wikipedia.org/wiki/Categorical_distribution), [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution), [Wishart](https://en.wikipedia.org/wiki/Wishart_distribution), [Inverse Wishart](https://en.wikipedia.org/wiki/Inverse_Wishart_distribution)** and many others. A number of common distributions are exponential families only when certain parameters are considered fixed and known, e.g**.** **[binomial](https://en.wikipedia.org/wiki/Binomial_distribution) (with fixed number of trials), [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution) (with fixed number of trials), and [negative binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution)** **(with fixed number of failures).** Examples of common distributions that are **not exponential families are** **[Student's \*t\*](https://en.wikipedia.org/wiki/Student's_t_distribution), most [mixture distributions](https://en.wikipedia.org/wiki/Mixture_distribution)****,** and even the family of [uniform distributions](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous))with unknown bounds. 

Scalar parameter exponential family 可以表示如下:

![f_{X}(x\mid \theta )=h(x)\exp \left(\eta (\theta )\cdot T(x)-A(\theta )\right)](https://wikimedia.org/api/rest_v1/media/math/render/svg/393d23d5e866e091c87546255856ea9313d8f126)

where *T*(*x*), *h*(*x*), *η*(θ), and *A*(θ) are known functions.

\* T(x) 就是 sufficient statistic.  

\* η 稱為 natural parameter.  

\* A(η) 稱為 log-partition function.  

 

**Exponential families 有一些重要的特性:**

\* Exponential families 的 sufficient statistics T(x) 可以 summarize arbitrary amount of IID data.

\* Exponential families 都有 conjugate priors, 這對 Bayesian statistics 很有用。

\* Exponential families 的 posterior pdf (prior * conditional pdf) 雖然有 close-form.  但大多非 exponential families. E.g. Student t-distribution. 

**Example of Normal distribution with known variance, estimate mean**



This is a single-parameter exponential family, as can be seen by setting



**Example of Normal distribution, estimate mean and variance** 



This is an exponential family which can be written in canonical form by defining



**可以看到不同的 θ 對應不同的 sufficient statistics.  即使 PDF 的形式一樣。**

 

### f(x; Θ) 可以視為 PDF/PMF function of x given Θ.  或是 L(θ; X) likelihood function of θ given X.

### Fisher 厲害之處在 explore f(x;Θ) or L(θ;X) 更多的統計特性

首先定義 score V  (given observed **X**) as the gradient w.r.t. θ of the log likelihood function 

V ≣ V(θ; **X**) = ∂ LL(θ; **X**) / ∂θ = ∂ log L(θ; **X**) / ∂θ = ∑ ∂ log L(θ, Xi)/∂θ   

一般而言, L(θ) 不一定是 concave, e.g. normal distribution.  但 LL(θ) or log L(θ) or ∑ log L(θ) 通常是 concave.

顯然這是 MLE 的延伸。MLE 基本上是設定 gradient wrt θ 為 0. i.e. V = 0 得到最佳的 θmle given oberved **X**.



 

再來就是對 LL(θ; **X) or** V(θ; **X**) 在 θ=θest 作 Taylor series,

LL(θ; **X**) =  C - n A/2 ( θ - θmle )^2  (Log Likelihood function 可以用 parabola 近似, 頂點就是 θmle) 

V(θ; **X**) = ∂ LL(θ; **X**) / ∂θ = - n A ( θ - θmle )   (V 就是 Log Likelihood 的 gradient or 斜率)

E( V(θ; X) ) = 0

I(θ) = Var(V(θ; X)) = E( V(θ; X)^2 ) = - E( ∂V/∂θ ) = - E( ∂^2 log f(x; θ)/∂θ∂θ )  

 

 

以上有點複雜, 有些參數有 n, 有些參數是 X or **X**。可以參考[本文](http://www.econ.uiuc.edu/~wsosa/econ507/MLE.pdf)整理很清楚。 

## 以下是 Score and Information 的解釋和推導

 θ or θo 是 K dimension parameters.  一般 θ 是變數，θo 是 given a value. 

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage55.png)

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage53.png) 

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage59.png)

 l(θ; Z) = log L(θ; Z)  稱為 log likelihood function 

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage54.png) 

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage56.png)

Score 分為 score and sample score (n observed samples).  原則上都是 function of θ with a given Z or Zi.

但 Score s(θ; Z) 的 Z 可視為 random variable.  因此 E[ s(θ;Z) ] 是有意義。當然 Z 也可以是 observed value, 那就是一個觀察值的 score. 可能意義不大。

相形之下 Sample score 中的 Zi 比較是實際 observed values.  比較少用 E(sample score).  當然也可以定義為 Z1, Z2,… Zn 的 joint pdf.  

In summary, 如果要算 Expectation score, 多半是用一個 random variable Z.  如果是算實值的 score, 多半是用 sample score (deterministic).  

同樣的道理也可以 apply to Hessian (RV) and Sample hessian (deterministic); 或是 (Fisher) Information and Sample information? (有 Sample information 嗎?)  嚴格來說 Fisher information 是取 expectation value, 本身已經是 deterministic function of θ.  並非 random variable.  當然也沒有所謂的 sample information.  不過我們就類推 Sample information = -1 * Sample hessian, 因為 expectation of a value = value. 

從大數法則來看:  Information ≈ Sample information / n.  

log likelihood l(θ;Z) 基本上大多是 concave funciton (當然有例外，特別是在 mixture pdf), 所以原則上 Hessian and Sample hessian: H(θ; Z) ≦ 0

反之 information matrix J or Fisher information I(θ) 是 covaraince matrix (or variance for single variable), 都是 positive or postive semi-definite matrix.  後面証明因為 E[Score] = 0, 因此 information matrix 也就是 Score 的 variance.  

最神奇的是 J (or I a KxK matrix) = Cov[ s(θo; Z) ] = E[ s(θo;Z) s(o;Z)' ] =  - E[ H(θo; Z) ]  !!  証明見 Appendix.

 

## Fisher Information 的物理意義是什麼? 

首先 Fisher information 是 Score (function of Z) random variable 的 variance (or covaraince matrix).

Fisher information 只是 function of θ.  Z, as a random variable 在做 expectation 計算中消失。

所以 given θo and Z 的 pdf.  可以很容易算出 Fisher Information 的理論值。

實務上則非如此。因為 θo 是我們要 estimate 的參數!  

###  

### (1) 我們只有 given samples and Z 的 pdf.  如何推算 Sample information?  

### (2) 另外一個問題是 Fisher Information 為什麼重要? 

先說結論 (2):  Sample Information 就是用 Zi 估計 θo 所得到 θest 的 variance (or covaraince matrix) 的倒數, i.e. Cov[θest - θo]^(-1).  所以 Information 愈大，θest 就愈接近 θo.  如何增加 Sample information, (1) 增加 samples n; (2) 增加 Var(Score^2) 

回到第一個問題，沒有 θo, 如何推算 Sample information. 雖然沒有 θo, 但有 n 個 samples, Zi.

顯然解法是先用 Sample Zi 估計 θo as θest.  

Method 1:  用 θest 代入 Fisher Information 可以得到 (unit) Fisher information.  Sample information = n * Fisher information  

Method 2:  用 θest and Zi 計算 Sample score, Sample hessian, and Sample Information.  當然我們假設 Sample information = -1 * Sample hessian.  從另外一個角度想，就是把 Z1, Z2, .. Zn 想成一個 joint PDF or likelihood function.  同樣可以算出 Fisher information.  代入 θest 就是這個 sample (Z1, Z2, …, Zn) 的 Sample information.

直觀來說 Method 1 和 Method 2 在 n 很大時，會趨近相等(by 大數法則)。但在 sample 不多時 (e.g. n = 2) 如何?

結論似乎相同。只要是 Z1, Z2, .. Zn 是 IID.  

可以參考[本文](http://www.econ.uiuc.edu/~wsosa/econ507/MLE.pdf) (p41. variance estimation) 給出三個方法。Method 1 對應 (3); Method 2 對應 (1); 還有 (2) 是用 sample score 平方 (or empirical variance of score) 來算。

 

實際(物理)上的 Information 當然還是有 n 個 samples 的 Sample information.  可以看到 samples 愈多，Sample hessian 就愈負。也就是 Information 就愈大愈多。 (Sample) Information ~ n.

如果我們畫 log likelihood function of n-samples w.r.t θ (e.g. of Normal distribution), 會看到一個 concave function of θ.  

在 Sample score = 0 的 θ,  也稱為 θmle.  對應的是 log likelihood function 的頂點。Sample information 對應的就是頂點 (需要是頂點嗎? 還是每一點都 ok?) 的曲率。顯然 n 愈大，sum of log likelihood function 在頂點的曲率也愈大。曲率的倒數就是 θest 的準確度 or varaince.  這是只有在 MLE 適用?  或是非 MLE estimator 也適用? 

 

## Examples 

我們用 normal distribution with u = θ (mean is the estimated parameter) and known σ=1 (variance) with two sample Z1, Z2 為例。

很直觀 u ≈ θest = (Z1+Z2)/2

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage60.png)

Fisher information of a Normal distribution = 1 assuming 𝜎 = 1.

 

Method 1: Fisher information = 1 => Sample information = 2 不管 θest 是多少。

Method 2: 如果代入 Z1 and Z2 into Sample hessian = hessian(Z1) + hessian(Z2) = -2 (也是和 Z1, Z2 無關).  因此 Sample information = -1 * -2 = 2.  結果和 method 1 相同。

其實只要 Fisher information 和 θ 無關 (如 normal distribution, exponential distribution).  Method 1 and 2 結果會一樣。

 

再來看 Bernoulli distribution.  參考 http://www.ejwagenmakers.com/submitted/LyEtAlTutorial.pdf

 ![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage61.png)

Method 1:  θest = 1/2(Z1+Z2)  =>  I(θ) = 1/[θ(1-θ)]

=>  Sample information = 2 * I(θ) = 2 / [0.5(Z1+Z2) * (1-0.5(Z1+Z2))]

Z1, Z2, θest, Sample information (n=2) 分別如下:

0, 0, 0, ∞

0, 1, 0.5, 8

1, 0, 0.5, 8

1, 1, 1, ∞

 

Method 2:  因為 Sample information = (-1) * Sample hessian = (-1) * [ sum of each hessian ] 

Z1, Z2, θest, Sample information (n=2) 分別如下:

0, 0, 0, ∞

0, 1, 0.5, 8

1, 0, 0.5, 8

1, 1, 1, ∞

似乎只要 θest 相同，method 1 and method 2 的結果一樣。這好像 make sense.  因為只要 I(θ) 是從 pdf/log likelihood 算出且 IID.  ?

什麼時候會 Fisher information*n 和 Sample information 不同?  就是在 Fisher information 是用真的θo 得出 (hidden variable), 但是 sample information 卻是從 Z1, Z2, .., Zn 得出。因為 Fisher information 是 weighted (by pdf) information. 但是 Sample information 是一次的 samples (Z1, …, Zn) 得出的 Fisher information.  

參考本文的[例子](http://www.ejwagenmakers.com/submitted/LyEtAlTutorial.pdf)。 

 

所以 consistent estimator 就是讓 θest 和真實的 θo 趨近: consistency

以及 efficient estimtor 就是 Sample information under θest 趨近 Fisher information (under θo) * n:  efficiency

另外還要加一個 asymptotic normality 條件。

 

### Fisher 在 1925 從 consistency 和 efficiency 角度提出 maximum likelihood estimator (MLE) 是 consistent and efficient estimator.  而不需要用到 sufficient statistic.  因為有些 distribution 的 sufficient statistic 不存在。但是 MLE 仍存在。 Rao 在 1963 証明這點。

 

### Asymptotic Normality of Maximum Likelihood Estimator

回到實務面。一般我們只有 Z1, Z2, …, Zn observed samples 以及 Z random variable 的 PDF.

先從 Sample Score (deterministic function of θ) = 0 得出 maximum likelihood estimation, θn.

###  ![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage63.png)

因為 MLE 是 consistent estimator, 在 n 夠大時 θn → θo.  可以做 1st order Taylor expansion.

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage64.png)

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage66.png)

 

下面重組並引入 n 做為 normalization constant.  在 Sample hessian 部份被 normalized to unit Sample hessian.

對應的 n 被分為 √n * √n.  一個放在 Sample score 的分母，另一個放在 estimated error 的分子。

### ![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage67.png)

接下來就是最精采的部份:

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage69.png)

這應該容易看出。藉著 ergodic 特性，time average 等於 expectation value.

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage70.png)

再來就不是那麼直覺，Normalized sample score (by √n) 會收歛到 zero mean, variance 為 Fisher information 的 normal distribution. 

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage71.png)

Zero mean 很容易了解。因為 E[Score] = E[s(θo;Z)] = 0

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage72.png)

同樣 by ergodic property,  Sample score 的平均值 (/n), 會收歛到 E[s(θo;Z)] = 0.  即使 * √n = 0

另外 Var[Score] = Var[s(θo;Z)] = J.  所以 variance 會收到 √n 的 boost.  變成 n/n^2 = 1/n.  也就是 unit variance 的平均值。

By Central Limit Theory (CLT), 可以 show 最後收歛到 N(0, J).

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage73.png)

因此 estimated error, |θn-θo|, 角度是反比於 √n; 從 θn variance 角度則是反比於 n

## |θmle-θo| 收歛的速度和 √n 正比。或是 θmle 的 variance 和 n 成反比，也和 Fisher information 反比。 

 

## Rao-Cramer Bound or Inequality

## ![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage74.png)

注意 psd 是 positive semi-definite.  V(θ*) = Var(θ*)

前文已提到 MLE 是 Var(θmle) →  (nJ)^(-1).  因此 MLE 比其他的 unbiased estimator 更 efficient!

The proof is in appendix. 

 

## Appendix

上述用到的幾個等式的証明 

![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage56.png)

可以看到是用 random variable Z 的 Score and Hessian (非 Sample score and Sample hessian).

同時用 θo 暗示 give a fixed θ.  注意 θo 不需要是 θmle, for any θo 以上等式都為真。

 

## Appendix A: Proof of Score equality

 

 ![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage57.png)

 ![NewImage](http://allenlu2007.files.wordpress.com/2016/07/mlenewimage58.png)

 

**Proof of E[Score] and Information equality**

![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage5.png)

 

 

![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage6.png)

![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage7.png) 

![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage8.png)


![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage9.png)

![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage10.png)

![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage11.png)

![NewImage](http://allenlu2007.files.wordpress.com/2016/08/mlenewimage12.png)

 

**Proof of Rao-Cramer Inequality**

Use DK divergence >= 0 to prove. 

 

**Proof of MSL Consistency** 

 

### Relationship to KL divergence

 



, sufficient statistics, score, Fisher information, Rao-Cramer bound 等都和此有關。

∂L(θ;X)/∂θ = 0.  為何 pdf 的 maximum point 可以用來做 estimation.  乍看 make sense, 但如果 pdf 有兩個 max, 或是 uniform distribution?  結果如何?

