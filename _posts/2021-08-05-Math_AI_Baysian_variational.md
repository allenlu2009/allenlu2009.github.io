---
title: Math AI - From EM to Variational Bayesian Inference
date: 2021-08-05 08:29:08
categories:
- AI
tags: [softmax, EM, Bayesian, Variational]
typora-root-url: ../../allenlu2009.github.io
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>

## Major Reference
* [@poczosCllusteringEM2015]
* [@matasExpectationMaximization2018] good reference
* [@choyExpectationMaximization2017]
* [@tzikasVariationalApproximation2008] excellent introductory paper

## EM Algorithm
EM 可以視為 MLE 的 extension to hidden state / data.

Let's start with EM algorithm

$$\begin{align}
\ln p(\mathbf{x} ; \boldsymbol{\theta})&=F(q, \boldsymbol{\theta})+K L(q \| p) \\
F(q, \boldsymbol{\theta})&=\int q(\mathbf{z}) \ln \left(\frac{p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})}{q(\mathbf{z})}\right) d \mathbf{z} \\
\mathrm{KL}(q \| p)&=-\int q(\mathbf{z}) \ln \left(\frac{p(\mathbf{z} \mid \mathbf{x} ; \boldsymbol{\theta})}{q(\mathbf{z})}\right) d \mathbf{z}
\end{align}$$

$$\begin{align}
Q\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\mathrm{OLD}}\right) &=\int p\left(\mathbf{z} \mid \mathbf{x} ; \boldsymbol{\theta}^{\text {OLD }}\right) \ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta}) d \mathbf{z} \nonumber\\
&=\langle\ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})\rangle_{p\left(\mathbf{z} \mid \mathbf{x} ; \boldsymbol{\theta}^{0 \mathrm{LD}}\right)} \label{eqQ}
\end{align}$$

此時可以用 $\eqref{eqQ}$ 定義 EM algorithm

$$\begin{align}
\text{E-step : Compute}\quad &p\left(\mathbf{z} \mid \mathbf{x} ; \boldsymbol{\theta}^{\mathrm{OLD}}\right) \label{eqE}\\
\text{M-step : Evaluate}\quad &\boldsymbol{\theta}^{\mathrm{NEW}}=\underset{\boldsymbol{\theta}}{\arg \max } Q\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\mathrm{OLD}}\right) \label{eqM}
\end{align}$$

一般 $\eqref{eqQ}$ 的 joint distribution $p\left(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta}\right)$ 包含完整的 data，容易計算或有 analytical solution.
大多的問題是 $\eqref{eqE}$ conditional or posterior distribution 是否容易計算，是否有 analytical solution.

## Variational EM Framework

最簡單的話就是 hidden variable z = z1, z2, .., zM.  and p(z) = p(z1)...p(zM).
什麼時候會有這種 distribution product?  後面會說明。

$$\begin{equation}
q(\mathbf{z})=\prod_{i=1}^{M} q_{i}\left(z_{i}\right) \label{eqFactor}
\end{equation}$$


$$\begin{align}
F(q, \boldsymbol{\theta})=& \int \prod_{i} q_{i}\left[\ln p (\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})-\sum_{i} \ln q_{i}\right] d \mathbf{z}\nonumber\\
=& \int \prod_{i} q_{i} \ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta}) \prod_{i} d z_{i} - \sum_{i} \int \prod_{j} q_{j} \ln q_{i} d z_{i} \nonumber\\
=& \int q_{j}\left[\int \ln p (\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta}) \prod_{i \neq j}\left(q_{i} d z_{i}\right)\right] d z_{j} -\int q_{j} \ln q_{j} d z_{j}-\sum_{i \neq j} \int q_{i} \ln q_{i} d z_{i} \nonumber\\
=& \int q_{j} \ln \tilde{p} (\mathbf{x}, z_{j} ; \boldsymbol{\theta}) d z_{i}-\int q_{j} \ln q_{j} d z_{j} -\sum_{i \neq j} \int q_{i} \ln q_{i} d z_{i} \nonumber\\
=&-\mathrm{KL}\left(q_{j} \| \tilde{p}\right)-\sum_{i \neq j} \int q_{i} \ln q_{i} d z \label{eqVarELBO}
\end{align}$$

where 

$$\begin{equation}
\ln \tilde{p}\left(\mathbf{x}, z_{j} ; \boldsymbol{\theta}\right)=\langle\ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})\rangle_{i \neq j}=\int \ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta}) \prod_{i \neq j}\left(q_{i} d z_{i}\right) \label{eqVarJ}
\end{equation}$$

$\eqref{eqVarELBO}$ 是 (variational, 因為有 KL divergence) lower bound, KL divergence 必大於 0, 負號後必小於 0.  第二項加上負號是 self-entropy 必大於 0.  
直觀看出讓 KL 為 0，就是 $q_j(z_j) = \tilde{p}(x, z_j; \theta)$, 似乎就是最大值 (how about the self-entropy?).
也就是 optimal distribution $q_j^* (z_j)$ 是

$$
\ln q_j^* \left(z_{j}\right)=\langle\ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})\rangle_{i \neq j} + \text{const.} 
$$

上面的 const 可以由 distribution normalization 得到。所以我們可以得到一組 consistency conditions $\eqref{eqVarJ2}$ for the maximum of variational lower bound subject to $\eqref{eqFactor}$.

$$\begin{equation}
q_{j}^{*}\left(z_{j}\right)=\frac{\exp \left(\langle\ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})\rangle_{i \neq j}\right)}{\int \exp \left(\langle\ln p(\mathbf{x}, \mathbf{z} ; \boldsymbol{\theta})\rangle_{i \neq j}\right) d z_{j}} \quad\text{for}\,\, j=1,\cdots,M \label{eqVarJ2}
\end{equation}$$

$\eqref{eqVarJ2}$ 顯然不會有 explicit solution, 因為 $q_j$ factors 之間是相互 dependent.  A consistent solution 需要 cycling through these factors.  我們定義 Variational EM algorithm

$$\begin{aligned}
\text{Variational E-step : Evaluate}\quad &q^{\mathrm{NEW}}(\mathbf{z})\quad\text{using above equations}\\
\text{Variational M-step : Find}\quad &\boldsymbol{\theta}^{\mathrm{NEW}}=\underset{\boldsymbol{\theta}}{\arg \max } F\left(q^{\mathrm{NEW}}, \boldsymbol{\theta}\right) \label{eqM2}
\end{aligned}$$

## Examples

### 例一： Linear Regression (filter/estimate a noisy signal)
我很喜歡這個例子。從簡單的 least-square error filter 進步到 Kalman filter.  類似的應用：deconvolution/equalization, channel estimation, speech recognition, frequency estimation, time series prediction, and
system identification. 

#### 問題描述
考慮一個未知信號 $y(x) \in R, x \in \Omega ⊆ R^N$, i.e. $R^N \to R$. 
我們想要 predict its value $t_* = y(x_*)$ at an arbitrary location $x_* \in \Omega$.  

我們用 vector 表示 $(t_1, \cdots, t_N)$
 using a vector t = (t1,..., tN)T of N noisy observations tn = y(xn) + εn, at locations x = (x1,..., xN)T, xn ∈ , n = 1,..., N. The additive noise εn is commonly assumed to be independent, zeromean, Gaussian distributed:

$$
y(\mathbf{x})=\sum_{m=1}^{M} \omega_{m} \phi_{m}(\mathbf{x})
$$

注意 $y(x)$ 不是真正的 observables, 而是加上 noise 之後的 t 才是 observations.  我們的目標就是用 $\mathbf{t}$ 來 estimate $\mathbf{w}$.

$$
\mathbf{t}=\boldsymbol{\Phi} \mathbf{w}+\boldsymbol{\varepsilon}
$$

The likelihood function

$$\begin{aligned}
p(\mathbf{t} ; \mathbf{w}, \beta)&=N\left(\mathbf{t} \mid \mathbf{\Phi} \mathbf{w}, \beta^{-1} \mathbf{I}\right)\\
&=(2 \pi)^{-\frac{N}{2}} \beta^{\frac{N}{2}} \exp \left(-\frac{\beta}{2}\|\mathbf{t}-\Phi \mathbf{w}\|^{2}\right)
\end{aligned}$$

#### 三種解法圖式

以下我們用三種 methodologies 用 $\mathbf{t}$ 來 estimate $\mathbf{w}$ (i.e. signal) and $\beta$ (i.e. noise if needed).

*Method 1:* ML Estimation 
如果 number of parameters (w) is the same as the number of observations (t), the ML estimates are very sensitive to the model noise.  我們可以用 DAG (Directed Acyclic Graphic) 說明，如下圖 (a).  雙圓框 t 代表 observed random variable. 方框 (W, beta) 代表 parameter to be estimated.  單圓框（e.g. (b) W）代表 hidden random variable. 

*Method 2:* 假設 weight W 是 random variable with imposed prior. 我們先用 a simple Bayesian model with stationary Gaussian prior on weight, 如下圖 (b).  以這個 model 而言，我們用 EM algorithm performs Bayesian inference.  結果 robust to noise, 類似 Kalman filter? 

<img src="/media/16286850167880.jpg" width="414">

*Method 3:* method 2 的一個缺點是假設 stationary Gaussian noise (i.e. $\beta$, a fixed value to be estimated, 無法 capture the local signal properties.  我們可以引入更複雜 spatially/temporally varying hierarchical model which is based on a non-stationary Gaussian prior for the weight, W and a hyperprior, $\beta$, 如下圖 (c).

這麼複雜的 DAG 顯然無法用 EM algorithm 解，必須用本文的 "Variational EM Framework" infer values of the unknowns. 

<img src="/media/16286850351205.jpg" width="245">

#### Method 1, ML for Vanilla Linear Regression

始於 likelihood function

$$\begin{aligned}
p(\mathbf{t} ; \mathbf{w}, \beta)=(2 \pi)^{-\frac{N}{2}} \beta^{\frac{N}{2}} \exp \left(-\frac{\beta}{2}\|\mathbf{t}-\Phi \mathbf{w}\|^{2}\right)
\end{aligned}$$

假設 $\mathbf{w}, \beta$ 為 constant parameters (to be estimated).  Maximize the likelihood or log-likelihood 等價於 minimize $\|\mathbf{t}-\Phi \mathbf{w}\|^{2}$.  因此**maximal likelihood (ML) estimate of w 等價 least squares (LS) estimate.

$$\begin{equation}
\mathbf{w}_{L S}=\underset{w}{\arg \max } p(\mathbf{t} ; \mathbf{w}, \beta)=\underset{w}{\arg \min } E_{L S}(\mathbf{w})=\left(\boldsymbol{\Phi}^{T} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{T} \mathbf{t} \label{eqLS}
\end{equation}$$

很多情況 $\left(\boldsymbol{\Phi}^{T} \boldsymbol{\Phi}\right)$ 可能是 "ill-conditioned" and difficult to invert.  意味如果 observation t 包含 noise $\varepsilon$, noise 會嚴重干擾 $\mathbf{w}_{L S}$ estimation.  

##### 例 1A：Communication equalization/deconvolution
Assuming a lowpass channel $\Phi = 1 + 0.9 z^{-1}$.  The equalizer $\left(\boldsymbol{\Phi}^{T} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{T}$ 變成 highpass filter; zero-forcing equalizer (ZFE).  如果 noise $\varepsilon$ 是 broadband noise, high frequency noise 會被放大。

In the case of ML, 我們必須小心選 basis functions to ensure matrix $\left(\boldsymbol{\Phi}^{T} \boldsymbol{\Phi}\right)$ can be inverted and avoid "ill-condition".  通常使用 sparse model with few basis functions.

#### Method 2, EM algorithm for Bayesian Linear Regression

Method 2 放寬 $w$ 從定值 fixed value 變成 distribution (random variable). Voila，這就是 Bayesian 精神！

A Bayesian treatment of the linear model begins by assigning a prior distribution to the weights of the model. This introduces bias in the estimation but also greatly reduces its variance, which is a major problem of the ML estimate.

此處我們用 common choice of independent, zero-mean, Gaussian prior distribution for the weights of the linear model:

$$
p(\mathbf{w} ; \alpha)=\prod_{m=1}^{M} N\left(w_{m} \mid 0, \alpha^{-1}\right)
$$

當然假設 zero-mean 聽起來有點奇怪，有可能引入 bias, 但好處是有 regularization 的效果，儘量讓 $w_m$ 不要太大。 

Bayesian inference 接下來是計算 posterior distribution of the hidden variable 

$$\begin{equation}
p(\mathbf{w} \mid \mathbf{t} ; \alpha, \beta)=\frac{p(\mathrm{t} \mid \mathbf{w} ; \beta) p(\mathbf{w} ; \alpha)}{p(\mathbf{t} ; \alpha, \beta)} \label{eqMAP}
\end{equation}$$

$\eqref{eqMAP}$ 分母部分進一步展開：

$$p(\mathbf{t} ; \alpha, \beta)=\int p(\mathbf{t} \mid \mathbf{w} ; \beta) p(\mathbf{w} ; \alpha) d \mathbf{w}=N\left(\mathbf{t} \mid 0, \beta^{-1} \mathbf{I}+\alpha^{-1} \mathbf{\Phi} \boldsymbol{\Phi}^{T}\right)$$

$\eqref{eqMAP}$，posterior of the hidden variable，可以寫成：

$$\begin{equation} 
p(\mathbf{w} \mid \mathbf{t} ; \alpha, \beta)=N(\mathbf{w} \mid \boldsymbol{\mu}, \boldsymbol{\mathbf{\Sigma}}) \label{eqPost}
\end{equation}$$

where 

$$\begin{align}
\boldsymbol{\mu} &=\beta \boldsymbol{\Sigma} \Phi^{T} \mathbf{t} \label{eqMean}\\
\boldsymbol{\Sigma} &=\left(\beta \boldsymbol{\Phi}^{T} \boldsymbol{\Phi}+\alpha \mathbf{I}\right)^{-1} \label{eqVar}
\end{align}$$

可以證明，$\alpha, \beta$ 可以用以下的 maximum likelihood estimate.

$$\begin{align}
\left(\alpha_{\mathrm{ML}}, \beta_{\mathrm{ML}}\right)=& \underset{\alpha, \beta}{\arg \min }\left\{\log \left|\beta^{-1} \mathbf{I}+\alpha^{-1} \boldsymbol{\Phi} \boldsymbol{\Phi}^{T}\right|\right. \nonumber \\
&\left.+\mathbf{t}^{T}\left(\beta^{-1} \mathbf{I}+\alpha^{-1} \boldsymbol{\Phi} \boldsymbol{\Phi}^{T}\right)^{-1} \mathbf{t}\right\} \label{eqab}
\end{align}$$

直接計算 $\eqref{eqab}$ 非常困難。除了 $\eqref{eqab}$ 微分非常複雜。$\alpha, \beta \ge 0$ 是一個 constrained optimization 問題。 EM algorithm 提供一個有效的方法解 $\alpha, \beta$ and infer $\mathbf{w}$ 

**E-step** Compute the Q function

$$\begin{aligned}
Q^{(t)}(\mathbf{t}, \mathbf{w} ; \alpha, \beta) &=\langle\ln p(\mathbf{t}, \mathbf{w} ; \alpha, \beta)\rangle_{p\left(\mathbf{w} \mid \mathbf{t} ; \alpha^{(t)}, \beta^{(t)}\right)} \\
&=\langle\ln p(\mathbf{t} \mid \mathbf{w} ; \alpha, \beta) p(\mathbf{w} ; \alpha, \beta)\rangle_{p\left(\mathbf{w} \mid \mathbf{t} ; \alpha^{(t)}, \beta^{(t)}\right)} \\
&=\left\langle\frac{N}{2} \ln \beta-\frac{\beta}{2}\left(\|\mathbf{t}-\boldsymbol{\Phi} \mathbf{w}\|^{2}\right)\right.\\
&\left.+\frac{M}{2} \ln \alpha-\frac{\alpha}{2}\left(\|\mathbf{w}\|^{2}\right)\right\rangle+\text { const } \\
=& \frac{N}{2} \ln \beta-\frac{\beta}{2}\left\langle\|\mathbf{t}-\boldsymbol{\Phi} \mathbf{w}\|^{2}\right\rangle+\frac{M}{2} \ln \alpha \\
&-\frac{\alpha}{2}\left(\left\langle\|\mathbf{w}\|^{2}\right\rangle\right)+\text { const. }
\end{aligned}$$

三角括號是對 $p(\mathbf{w} \mid \mathbf{t} ; \alpha^{(t)}, \beta^{(t)})$ 的期望值。代入 $\eqref{eqPost}$ 得到

$$
\begin{aligned}
Q^{(t)}(\mathbf{t}, \mathbf{w} ; \alpha, \beta)=& \frac{N}{2} \ln \beta-\frac{\beta}{2}\left(\left\|\mathbf{t}-\boldsymbol{\Phi} \boldsymbol{\mu}^{(t)}\right\|^{2}+\operatorname{tr}\left[\boldsymbol{\Phi}^{T} \boldsymbol{\Sigma}^{(t)} \boldsymbol{\Phi}\right]\right) \\
&+\frac{M}{2} \ln \alpha-\frac{\alpha}{2}\left(\left\|\boldsymbol{\mu}^{(t)}\right\|^{2}+\operatorname{tr}\left[\boldsymbol{\Sigma}^{(t)}\right]\right)+\mathrm{const}
\end{aligned}
$$

where $\boldsymbol{\mu}^{(t)}$ and $\boldsymbol{\Sigma}^{(t)}$ are computed using the current estimates of the parameters $\alpha^{(t)}$ and $\beta^{(t)}$ :

$$
\begin{aligned}
\boldsymbol{\mu}^{(t)} &=\beta^{(t)} \boldsymbol{\Sigma}^{(t)} \boldsymbol{\Phi}^{T} \mathbf{t} \\
\boldsymbol{\Sigma}^{(t)} &=\left(\beta^{(t)} \mathbf{\Phi}^{T} \boldsymbol{\Phi}+\alpha^{(t)} \mathbf{I}\right)^{-1}
\end{aligned}
$$

**M-step** Maximize $Q^{(t)}(\mathbf{t}, \mathbf{w} ; \alpha, \beta)$ with respect to $\alpha, \beta$.

$$
\left(\alpha^{(t+1)}, \beta^{(t+1)}\right)=\underset{(\alpha, \beta)}{\arg \max } Q^{(t)}(\mathbf{t}, \mathbf{w} ; \alpha, \beta)
$$

結果很簡單

$$
\begin{align}
\alpha^{(t+1)} &=\frac{M}{\left\|\boldsymbol{\mu}^{(t)}\right\|^{2}+\operatorname{tr}\left[\boldsymbol{\Sigma}^{(t)}\right]} \label{eqa}\\
\beta^{(t+1)} &=\frac{N}{\left\|\mathbf{t}-\boldsymbol{\Phi} \boldsymbol{\mu}^{(t)}\right\|^{2}+\operatorname{tr}\left[\boldsymbol{\Phi}^{T} \mathbf{\Sigma}^{(t)} \boldsymbol{\Phi}\right]} \label{eqb}
\end{align}
$$

$\eqref{eqa}$ 和 $\eqref{eqb}$ 同時保證 $\alpha, \beta$ 永遠為正值。

幾個重點：
* EM algorithm 有可能收斂到 local minimum; initial condition 很重要
* 注意 $\mathbf{w}$ 不是一個值，而是 distribution.  Inference of $\mathbf{w}$ 就是 posterior distribution $\eqref{eqPost}$.  Posterior distribution 的 mean $\eqref{eqMean}$ 稱為 Bayesian linear minimum mean squire error (LMMSE) inference for $\mathbf{w}$.

#### Method 3, Variational EM-based Bayesian Linear Regression

因為非常複雜，可以直接參考 [@tzikasVariationalApproximation2008].

##### 例 1B：Noisy Signal Estimation/Filtering

如下圖，Original signal 是虛線。實際的 observations 'x' 是 N = 50 samples 包含 signal + Gaussian noise ($\sigma^2 = 4 \times 10^{-2}$), 大約 SNR = 6.6dB.

這裡的 basis functions 使用 Gaussian kernels

$$
\phi_{i}(\mathbf{x})=K\left(\mathbf{x}, \mathbf{x}_{i}\right)=\exp \left(-\frac{1}{2 \sigma_{\phi}^{2}}\left\|\mathbf{x}-\mathbf{x}_{i}\right\|^{2}\right)
$$

接下來用上述三個方法 (1) ML estimation; (2) EM-based Bayesian inference, and (3) variational EM-based Bayesian inference.

(1) ML 基本上完全 follow noisy input, 所以最糟。這也符合期待，因為沒有任何 constraint on the weight. 所以所有的 weights 和 Gaussian kernel 都用來 fit noisy observations.  也就是說 N=50 samples/observations 對應 50 個 Gaussian kernel functions.  這可以從下圖的綠線看出。

(2) Weights are constrained by prior, 此處 prior 假設 zero-mean Gaussian, which regularise the weight to be minimum; otherwise it will incur penalty.

(3) 我們可以通過 $a, b$ 選取控制 non-zero weights, 類似 supporting vectors in SVM.  我們稱為 relevance vectors (RV). 此例只有 5 個  non-zero RV.

<img src="/media/16287874512211.jpg">

### 例二： Bayesian GMM

$$
p(\mathbf{x})=\sum_{j=1}^{M} \pi_{j} N\left(x ; \boldsymbol{\mu}_{j}, \mathbf{T}_{j}\right)
$$

where $\boldsymbol{\pi} = \{ \pi_j \}$ 代表 weights or mixing coefficients.  $\boldsymbol{\mu} = \{ \boldsymbol{\mu}_{j} \}$ 是 means of Gaussian distribution.  $\mathbf{T} = \{ \mathbf{T}_{j} \}$ 是 precision (inverse covariance) matrices.  在 Bayesian GMM 我們更常用 precision matrix.

Bayesian GMM 和一般 GMM 有什麼不同？ 最大的差別就是 $\boldsymbol{\pi}, \boldsymbol{\mu}, \mathbf{T}$ 不再是 parameters for estimation, 而是 random variables. 這有什麼好處？我們可以 impose or embedded our priors on $\boldsymbol{\pi}, \boldsymbol{\mu}, \mathbf{T}$, 通常是 conjugate priors (i.e. no informative priors) [^prior].

[^prior]: Dirichlet for $\boldsymbol{\pi}$.  Gauss-Wishart for ($\boldsymbol{\mu}, \mathbf{T})$  

Bayesian GMM 的 graph model 如下。Hidden random variables 包含 $h = (\mathbf{Z}, \boldsymbol{\pi}, \boldsymbol{\mu}, \mathbf{T})$. Bayesian 的目標是找出 $p(h\mid x)$, 顯然不會有 analytic solution.

<img src="/media/16285137362672.jpg" width="237">

因此我們 divide-and-conquer 利用 $\eqref{eqVarJ2}$
假設 mean-field approximation

$$
\begin{aligned}
q(\mathrm{h}) &= q_{Z}(\mathbf{Z}) q_{\pi}(\boldsymbol{\pi}) q_{\mu T}(\boldsymbol{\mu}, \mathbf{T}) \\
q_{Z}(\mathbf{Z}) &=\prod_{n=1}^{N} \prod_{j=1}^{M} r_{j n}^{z_{j n}} \\
q_{\pi}(\boldsymbol{\pi}) &=\operatorname{Dir}\left(\boldsymbol{\pi} \mid\left\{\lambda_{j}\right\}\right) \\
q_{\mu T}(\boldsymbol{\mu}, \mathbf{T}) &=\prod_{j=1}^{M} q_{\mu}\left(\boldsymbol{\mu}_{j} \mid \mathbf{T}_{j}\right) q_{T}\left(\mathbf{T}_{j}\right) \\
q_{\mu}\left(\boldsymbol{\mu}_{j} \mid \mathbf{T}\right) &=\prod_{j=1}^{M} N\left(\boldsymbol{\mu}_{j} ; \mathbf{m}_{j}, \beta_{j} \mathbf{T}_{j}\right) \\
q_{T}(\mathbf{T}) &=\prod_{j=1}^{M} W\left(\mathbf{T}_{j} ; \eta_{j}, \mathrm{U}_{j}\right)
\end{aligned}$$

看起來還是很複雜，不過 [@tzikasVariationalApproximation2008] 的 reference [27] 有詳細的公式。可以用“簡單” iterative update procedure 得到 optimal approximation $q(h)$ to the true posterior $p(h\mid x)$, 這就是 variational E-step.  下一步就是 variation M-step, 不贅述。

Bayesian-GMM 比起 EM-GMM 到底有什麼好處。前面提到可以 impose priors. 如果沒有 prior information (i.e. use conjugate prior), 還有好處嗎？[@tzikasVariationalApproximation2008] 的說法是 Bayesian-GMM 不會有 singular solution, i.e. single data point Gaussian.  然而在 EM-GMM 常常會發生，如下圖 20 Gaussian components。一般 EM-GMM 解決的方法就是多跑幾次 randomize initial conditions to avoid it.

![](/media/16285679550223.jpg)

另一個好處是可以直接用 Bayesian GMM 決定 Gaussian component number, 而不需要用其他方法 (e.g. cross-validation)。實作如下圖。(a) 初始是 20 component Gaussians; (b), (c) model evolution; (d) 最終解只剩下 5 個 Gaussian components, 其餘 15 個 Gaussian components weight 為 0。注意收斂的過程中都沒有 singularity.  

這聽起來比較 significant, 不過有一個 catch, 就是 Dirichlet prior 不允許 component mixing weight 為 0.  因此如果要用 Bayesian-GMM 決定 Gaussian component number, 必須 remove $\boldsymbol{\pi} = \{ \pi_j \}$ from priors.  也就是把 $\boldsymbol{\pi} = \{ \pi_j \}$ 視為 parameter to be estimated. 
 
![](/media/16285916007272.jpg)

Bayesian GMM 的 graph model 如下。注意此時的 $\pi$ 變成方框，代表 parameter to be estimated.  Hidden random variables 包含 $h = (\mathbf{Z}, \boldsymbol{\mu}, \mathbf{T})$.  

<img src="/media/16286002562443.jpg" width="237">

根據新的 DAG, 我們可以分解如下：

$$\begin{aligned}
&q(\mathrm{h})=q_{Z}(\mathbf{Z}) q_{\mu}(\boldsymbol{\mu}) q_{T}(\mathrm{T})\\
&q_{Z}(\mathbf{Z})=\prod_{n=1}^{N} \prod_{j=1}^{M} r_{j n}^{z_{j n}} \\
&q_{\mu}(\boldsymbol{\mu})=\prod_{j=1}^{M} N\left(\boldsymbol{\mu}_{j} \mid \mathrm{m}_{j}, \mathbf{S}_{j}\right) \\
&q_{T}(\mathbf{T})=\prod_{j=1}^{M} W\left(\mathbf{T}_{j} \mid \eta_{j}, \mathbf{U}_{j}\right)
\end{aligned}
$$

同樣經過一番計算 variational E-step and M-step (此處省略)，可以得到

$$
\pi_{j}=\frac{\sum_{n=1}^{N} r_{j n}}{\sum_{k=1}^{M} \sum_{n=1}^{N} r_{k n}}
$$

在 iteration 過程中，有一些 mixing coefficients $\{\pi_j\}$ 收斂到 0. 定性來說，variational bound 可以視為兩項之和：第一項是 likelihood function, 第二項是 prior 造成的 penalty term to penalizes complex models.

