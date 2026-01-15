---
title: Diffusion Theory for Gaussian Mixture Model
date: 2025-03-23 14:10:08
typora-root-url: ../../allenlu2009.github.io
link: 2023-02-08-GenAI_Diffusion
categories:
  - GenAI
tags:
  - Diffusion
  - GenAI
  - Graph
  - Language
  - Laplacian
  - Math-AI
---






## Main Reference

The Unreasonable Effectiveness of Gaussian Score Approximation for Diffusion Models and its Applications  [2412.09726](https://arxiv.org/pdf/2412.09726)

A unified perspective of [Diffusion Models](https://arxiv.org/pdf/2208.11970)  
https://arxiv.org/pdf/1503.03585.pdf : 2015/03 Stanford Diffusion paper: very good!


## BIG QUESTION!
- 可以把 Gaussian noise 改成 shot noise or other noise (blackout or blank?) 直接用於 denoise?
- 同樣問題，可以把 low resolution 變成 high resolution 直接用於 super resolution?
- 可以 finetune diffusion model for denoise and super resolution? yes, VARformer
- 
## Score Matching 

低維高維

noise corrupted distribution

![[Pasted image 20250412003855.png]]

Training to denoiser, 而不是 noise:  x = y + n
![[Pasted image 20250412004126.png]]

![[Pasted image 20250412004325.png]]

**前文的 neural network 近似 denoiser.**  就是直接 predict clean image, i.e.   $D_{\theta}(\tilde{\mathbf{x}}, \sigma) \approx \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathbf{x}$.  看起來似乎更合理。而且 denoiser 是已經廣泛研究和使用的 building block.  甚至可以直接拿現有 denoiser 用 diffusion process 生成 image.  不過現有的 denoiser 可能需要針對不同的 $\sigma$ 做一些調整。
$$\begin{aligned}
& \min_{\theta}\frac{1}{2}  \mathbb{E}_{p_{\text {data}}(\mathbf{x})} \mathbb{E}_{N({\mathbf{\epsilon}}; 0, I) }\left[\left\|\sigma^2 \mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}+\sigma \mathbf{\epsilon}})+{\mathbf{\epsilon}}{\sigma}\right\|_2^2\right]\\
&= \min_{\theta}\frac{1}{2}  \mathbb{E}_{p_{\text {data}}(\mathbf{x})} \mathbb{E}_{N(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)}\left[\left\|\sigma^2 \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})+\tilde{\mathbf{x}}-\mathbf{x}\right\|_2^2\right]\\
&= \min_{\theta}\frac{1}{2}  \mathbb{E}_{p_{\text {data}}(\mathbf{x})}  \mathbb{E}_{N(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)}\left[\left\|D_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)-\mathbf{x}\right\|_2^2\right]\\
&= \min_{\theta} \mathbb{E}_{p_{\text {data}}(\mathbf{y})}  \mathbb{E}_{N({\mathbf{n}}; 0, \sigma^2 I)}\left[\left\|D_{\boldsymbol{\theta}}({\mathbf{y}+\mathbf{n}}, \sigma)-\mathbf{y}\right\|_2^2\right]\\
\end{aligned}$$
	where x = y + n
## 假設 target distribution $P_{data}(y)$ 是 "low rank" Gaussian distribution $N(\mu, \Sigma)$

x = y + n 就是兩個 Gaussian convolution.  結果 p(x) = p(y+n) mean = $\mu$, final variance = $\Sigma + \sigma^2 I$.  
$s(x, \sigma) = \nabla \log p(x) = (\sigma^2 I + \Sigma)^{-1} (\mu - x)$
$D(x, \sigma) = \sigma^2 s + x$
![[Pasted image 20250412004414.png]]


$\sigma^2$  小，基本就得到 ~ $\mathbf{x} + \frac{\sigma^2 I}{\Sigma}\boldsymbol{\mu} = y +n + du$.   因爲無法完全 predict a noise (but low rank), 或是無法完全 denoise a noise (but low rank).  但是 good SNR.
反之，得到 u (mean), 合理的估計。

![[Pasted image 20250414233221.png]]
以上是 static，也就是第一個 loop (inner).  再來是第二個 loop, 如何改變 $\sigma$, Probability Flow (PF) ODE.
![[Pasted image 20250414235540.png]]
![[Pasted image 20250414235816.png]]


## Gaussian Mixture Model (GMM)

![[Pasted image 20250412005713.png]]

![[Pasted image 20250412005800.png]]



#### Low Dimension Manifold Assumption 
利用 (1) $\mathbf{x}$ 的 high dimensionality and (2) 實際 data 是在 low dimension manifold assumption.  我們可以利用兩種解法：
- Random projection
- Noise scheduling or annealing Langevin method

我們聚焦在第二種解法如下：
### NCSN from Stanford: Noise Conditional Score Network

- Forward path (training only):  add noise or perturbation, 非常簡單
- Reverse path (training and sampling): Annealing Langevin method (SDE)

It first perturbs the data point $\mathbf{x}$ with a pre-specified noise distribution $q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$ and then employs score matching to estimate the score of the perturbed data distribution $q_\sigma(\tilde{\mathbf{x}}) \triangleq \int q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) p_{\text {data }}(\mathbf{x}) \mathrm{d} \mathbf{x}$. The objective was proved equivalent to the following:

$$
\min_{\theta}\frac{1}{2} \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \, p_{\text {data }}(\mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_2^2\right]
$$

The optimal score network that minimizes Eq. satisfies $\mathbf{s}_{\boldsymbol{\theta}^*}(\mathbf{x})=\nabla_{\mathbf{x}} \log q_\sigma(\mathbf{x})$. However, $\mathbf{s}_{\boldsymbol{\theta}^*}(\mathbf{x})=\nabla_{\mathbf{x}} \log q_\sigma(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ is true only when the noise is small enough such that $q_\sigma(\mathbf{x}) \approx p_{\text {data }}(\mathbf{x})$.

#### Additive Gaussian Noise Perturbation
我們還沒真正討論 pre-specified noise distribution $q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$，最簡單的是 additive Gaussian noise，其表示式為 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{\epsilon}$，此處 $\mathbf{\epsilon}$ 是外加的 zero mean, unit variance 的 high dimension noise vector.  因此    
$$q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = N(\tilde{\mathbf{x}};\mathbf{x}, \sigma^2 I) = \frac{1}{\sqrt{2 \pi \sigma}}e^{-\frac{(\tilde{\mathbf{x}}-\mathbf{x})^2}{2\sigma^2}}$$
驗算一下：如果 $\sigma$ 非常小 $\sigma \approx 0$，$q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \approx \delta(\tilde{\mathbf{x}}-\mathbf{x})$，$q_\sigma(\tilde{\mathbf{x}}) \triangleq \int q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) p_{\text {data }}(\mathbf{x}) \mathrm{d} \mathbf{x} \approx p_{data}(\tilde{\mathbf{x}})$. 合理。接下來是重點：

$$\nabla_{\mathbf{x}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}$$ Score matching 可以簡化成
$$\begin{aligned}
&\min_{\theta}\frac{1}{2} \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \,p_{\text {data }}(\mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_2^2\right]\\
&= \min_{\theta}\frac{1}{2} \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \,p_{\text {data }}(\mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}\right\|_2^2\right]\\
\end{aligned}$$
上式富有物理意義。首先期望值是在 data manifold 加上 Gaussian noise 附近才有值。
- 如果 $\sigma$ 很小，基本就是在 data manifold 附近 L2 norm 要越小越好，最好是 0.
- 也就是 $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2} = \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})+\frac{\mathbf{\epsilon}}{\sigma} = \mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}+\sigma \mathbf{\epsilon}})+\frac{\mathbf{\epsilon}}{\sigma}\approx 0$,  其中 $\mathbf{\epsilon}$ 是 zero-mean, unit variance 的 high dimension noise vector.   也就是 $\mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}+\sigma \mathbf{\epsilon}})\approx -\frac{\mathbf{\epsilon}}{\sigma}$, why?
- (Wrong) 乍看非常奇怪，因爲 $\sigma\approx 0$，$\mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}+\sigma \mathbf{\epsilon}})\approx\mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}})\approx - \frac{\mathbf{\epsilon}}{\sigma}$  顯然 image manifold 上的 score function 不應該是非常大的 random noise! 而且和 image 的分佈無關，why?

#### 以下是我的解釋
$\mathbf{x}$ (image manifold) 是在低維空間，可以視為一個點。因為 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{\epsilon}$ ，所以 $\tilde{\mathbf{x}}, \mathbf{\epsilon}$ 才是高維空間。**有兩個類比**：
- 物理的超弦理論需要 10 維時空 (9 space + 1 time) 保持數學自洽，現實世界卻是 4 維時空 (3 space + 1 time).  一個解釋是每個 4 維時空的點都包含 6 維的微小流形 (manifold)。某一個 4 維時空點內部的 6 維流形也許對應 image manifold, 如下圖。  
- 好萊塢的電影**荷頓奇遇記 (Horton Hears a Who!)** 中在一顆灰塵中有一個 Whoville 王國。就是 4 維時空點中的 4 維時空。也就是 globle 是 8 維時空。
- 在高維空間的引力場無法直接應用在低維 manifold.  但是**高維空間的引力場可以指引並收斂到低維的 manifold.**  
- 如何要 sample 低維 manifold 上的不同點（或是在低維 manifold 上移動）？因為沒有低維的 local 引力場，因此先加上高維 random noise 再藉由高維引力場收斂到不同的低維 manifold 位置。

![[Pasted image 20250405204424.png]]

**重新檢視上式的物理意義：**
- 在 data manifold 附近，blurred score function (高維) 就是 noise (高維) 方向：$\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})=\mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}+\sigma \mathbf{\epsilon}})\approx -\frac{\mathbf{\epsilon}}{\sigma}$
- (高維) $\lim_{\sigma\to 0}\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})=\mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}+\sigma \mathbf{\epsilon}}) \nsim \mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}})$. 因為 $\mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}})$ 是定義在低維的 manifold (or point)的向量場。就像黑洞外面的**高維引力場**無法提供黑洞內部的**低維引力場**。 
- 不過 $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})$ 的外部引力場在控制 $\sigma$ 由大變小的確可以有效指引火箭到達黑洞的邊界 during sampling (inferencing).  也就是 sampling phase: $\mathbf{x} = \tilde{\mathbf{x}} + {\sigma^2} \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})\approx \tilde{\mathbf{x}} - {\sigma} \mathbf{\epsilon}$  
	- Step 1: 其實就是 Langevin dynamic equation，$\epsilon = \sigma^2$
	- ![[Pasted image 20250405223832.png]]
	- Step 2:  因為 $\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{\epsilon}$ , 其實 sampling 過程就是 denoiser.
- 因為我們沒有真正黑洞內部的 (低維) 引力場，也就是我們無法直接在黑洞內部移動。如何找到其他 image manifold 上的 image or guided image by text/image?  簡而言之，如何避免陷入一個固定的黑洞？ Answer Add noise!! 先在黑洞外 (或者內) random walk，再 denoise 到 image manifold。這是上式的 $\mathbf{z}_i$。
- Neural network $\theta$ 是在近似什麼？
	- 一定是近似一個 vector field.  每一個空間的高維點 input, 都產生一個對應的高維 output vector.  所以是 vector field.
	- 一般還有一個 scalar $\sigma_t$ 或是 $t$ 的 input as the input of the neural network $\theta$.  這樣可以 share weights for 所有的 noise level.  而不用對於每一個 $\sigma_t$ 都訓練一個 $\theta_t$.
	- **方法一是用 neural network 近似 score function**，也就是 predict “inverse noise”: $\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})=\mathbf{s}_{\boldsymbol{\theta}}({\mathbf{x}+\sigma \mathbf{\epsilon}})\approx -\frac{\mathbf{\epsilon}}{\sigma}$  
	- 方法一的問題是當加的 noise 很小，i.e. $\sigma$ (分母) 很小，predict score function 非常困難，因為 score function dynamic range 非常大，連帶影響  $\theta$ 的 dynamic range 也很大。
	- **方法二是用 neural network 近似 (or predict) noise**： ${\sigma^2} \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})\approx \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\sigma \mathbf{\epsilon}$ ，可以避免 numerical problem.  
	- 雖然 predict noise 聽起來很奇怪，不過確實有效，而且 noise 和原來的 image $\mathbf{x}$ 無關，所以計算上更容易收斂。一旦 noise is predicted, 只需要在 input 加上 residue branch 減掉 noise 就可以 generate image.
	- **方法三是 neural network 近似 denoiser.**  就是直接 predict clean image, i.e.   $D_{\theta}(\tilde{\mathbf{x}}, \sigma) \approx \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathbf{x}$.  看起來似乎更合理。而且 denoiser 是已經廣泛研究和使用的 building block.  甚至可以直接拿現有 denoiser 用 diffusion process 生成 image.  不過現有的 denoiser 可能需要針對不同的 $\sigma$ 做一些調整。
	- 方法二 (predict noise) 或是方法三 (denoised image) 看起來好像只差了一個 "$+ \tilde{\mathbf{x}}$"。但方法二似乎比較普遍因為在 DDPM 的每次迭代是要 predict "less noisy image"，所以方法三還要把 predicted clean image 再加回 scaled noise (還要再計算 noise)，脱褲子放屁。方法二 predict noise 更容易，直接 scale predicted noise 再從 input blurred image 扣掉一部分 noise。


## Summary : Noise Predictor and Denoiser
原始的問題是用 neural network 近似 score function: $s_{\theta}(\mathbf{x}) \approx \nabla_\mathbf{x} \log p_{data}(\mathbf{x})$, 不過不可行。
替代法案是
1. Noise predictor (score function):  ${\sigma^2} \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})\approx \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\sigma \mathbf{\epsilon}$
2. Denoiser (removed noise from blurred image):  $D_{\theta}(\tilde{\mathbf{x}}, \sigma) \approx \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathbf{x}$
注意這裏的 denoiser 是回復原來的 image,  和 DDPM 減少 noise 不同。

#### 一維例子
$\mathbf{x}$ 包含三個 images, 每一個的 probability density 用（低維）delta function 表示，其 log likelihood 的微分， 也就是 score function  $s(\mathbf{x}) = \nabla_\mathbf{x} \log p_{data}(\mathbf{x})$  基本是 delta function 的微分，基本是一個 ugly 無法定義的函數。
如果 randomly 選擇一個 x, 無法沿著其 score function 的指引得到原來的 images.

神奇的是：如果 $\mathbf{x}$ 加上一些 Gaussian noise 變成 blurred images  $\tilde{\mathbf{x}}$ , 其對應的 pdf or likelihood function 是 delta function 和 Gaussian function 的 convolution，也就是 mixture of Gaussian 如下圖上。   
- 每一個 log likelihood of Gaussian 得到 $-\frac{(\tilde{\mathbf{x}}-\mathbf{x})^2}{2\sigma^2} + C$ ，這是開口向下的拋物線。因為數值精度或是有限的 dataset，log likelihood 不可能無限小，而是有一個 floor，如下圖上。
- **Score function 是 log likelihood 微分之後是線性函數 $-\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}$ center 在 $\mathbf{x}_j$，斜率是 $-\frac{1}{\sigma^2}$，對應的斜率只會在小於幾個 $\pm \sigma$ 區間, 區間以外會變成 0, 如下圖下。
- Dataset $\mathbf{x}_j$ 是在非常低維空間，從高維的角度可以視為點。 只決定 0 點的位置。但是 $\sigma$ 決定斜率和區間。
- Noise annealing:  正確的做法是先讓 $\sigma$ 很大，可以指引方向。再來逐步縮小，讓 blurred image 變成 clear image.

![[Pasted image 20250318112923.png]]



### Tweedie Estimator

上述是 diffusion 的基本原理：在完全不知道低維 manifold $p_{data}(\mathbf{x})$ 長的什麽樣子，從一個高維 random sample 經過引力場 $\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = -\sigma \mathbf{\epsilon}$ 的引導得到低維 image samples，這是蠻神奇的。數學上可以用 Tweedie Estimator 説明。

Tweedie estimator 是一種參數估計方法 formulate 如下：
- 假設 $p(\theta)$ 是 **unknown prior**  
- 但是 **conditional probability 是 Gaussian**: $p(x\mid \theta) = N(\theta, \sigma^2)$.  
- 一般 **marginal pdf**, $p(x)=\int_{-\infty}^{\infty} p(x \mid \theta)\, p(\theta) \mathrm{d} \theta$, **不是 Gaussian** (除非 prior 是 Gaussian).  
- 目標是從觀察到的 samples $x$ 中推斷未知參數 $\theta$。

儘管 marginal pdf $p(x) = \int_{-\infty}^{\infty} p(x \mid \theta)\, p(\theta) \mathrm{d} \theta$ 一般不是高斯分布，但我們可以利用貝葉斯定理來進行有效的參數估計。這樣，我們可以得到後驗分布：

$$
p(\theta \mid x) = \frac{p(x \mid \theta)p(\theta)}{p(x)}
$$
依照 Bayes view $\theta$ 是一個 distribution 而不是一個固定的參數。最好的估計就是計算期望值，也就是 MMSE。以下就是 Tweedie estimator.
$$\theta_{MMSE} = \mathbb{E}_{p(\theta \mid x)}\left[\theta \mid x\right] = \mathbb{E}\left[\theta \mid x\right] = x+\sigma^2 \frac{\mathrm{~d}}{\mathrm{~d} x} \log p(x)$$
如果 $x$ 是一個高維向量 $\mathbf{x}$ 例如 image,  $$\theta_{MMSE} = \mathbb{E}_{p(\theta \mid \mathbf{x})}\left[\theta \mid \mathbf{x}\right] = \mathbb{E}\left[\theta \mid \mathbf{x}\right] = \mathbf{x}+\sigma^2 \nabla_{\mathbf{x}} \log p(\mathbf{x})$$
重點是 x 是 observable.

我們直接帶入 $\mathbf{x} \to \tilde{\mathbf{x}}, \theta \to \mathbf{x}$
$$\mathbf{x}_{MMSE} = \mathbb{E}_{p(\mathbf{x} \mid \tilde{\mathbf{x}})}\left[\mathbf{x} \mid \tilde{\mathbf{x}}\right] = \mathbb{E}\left[\mathbf{x} \mid \tilde{\mathbf{x}}\right] = \tilde{\mathbf{x}}+\sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}})$$
If p(x) 是 Gaussian -> ...

If many samples .... ->





在某一些條件可以得到 score function:  $\nabla_{\mathbf{x}} \log p(\mathbf{x})$
3. 如果 prior $p(\theta)$ 是 exponential family

https://web.stanford.edu/class/stats300a/HW/hw2.pdf
https://zhuanlan.zhihu.com/p/432212884

![[Pasted image 20250408002500.png]]



雖然我們沒有直接的 score function:  $\nabla_{\mathbf{x}} \log p(\mathbf{x})$, 但是在某些情況下可以用 neural network 訓練一個近似解。

N 改成 discrete pdf but exponential family, ok???

如果我們有很多 $\mathbf{x}$ samples 
$$\theta_{TW} = \mathbb{E}_{p(\mathbf{x})} \mathbb{E}_{p(\theta \mid \mathbf{x})}\left[\theta \mid \mathbf{x}\right] = \Sigma_{\mathbf{x}_i} \left[\mathbf{x}_i+\sigma^2 \nabla_{\mathbf{x}} \log p(\mathbf{x_i})\right]$$




Tweedie estimator 通常會選擇某種形式的先驗，例如共軛先驗，以便使計算變得更簡單。在實踐中，這可能意味著選擇一個方便的或有解釋意義的先驗，以便能夠從觀察到的數據中獲取有用的信息。

最後，透過最大化後驗概率（MAP）或採用其他貝葉斯方法，可以得到對於未知參數 $θ\) 的有效估計。這些方法在許多應用中都表現出色，尤其是在樣本量較小或者噪音較大的情況下，因為它們能夠充分利用先驗知識。.  

高維低維
https://zhuanlan.zhihu.com/p/594007789





- 因爲 data manifold $\mathbf{x}$ 的 dimension 比起 random Gaussian noise $\mathbf{\epsilon}$ 的 dimension 低非常多。比較正確的看法是把分解成 manifold 分量 + non-manifold 分量：
- x = x_m + x_n.   x+ sigma




Reverse path: Annealed Langevin drift，**有兩個 loops** 
	Outer loop: $\epsilon$ 從大變小 for denoise scheduling, **稱爲 annealing.**
	Inner loop: $\mathbf{z}_t$ for score directed random walk to converge higher probability region, **就是 Langevin drift + random walk.**

$$
\begin{aligned}
&\tilde{\mathbf{x}}_t=\tilde{\mathbf{x}}_{t-1}+\frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p\left(\tilde{\mathbf{x}}_{t-1}\right)+\sqrt{\epsilon} \mathbf{z}_t\\
&\text { where } \mathbf{z}_t \sim \mathcal{N}(0, I) \text {. The distribution of } \tilde{\mathbf{x}}_T \text { equals } p(\mathbf{x}) \text { when } \epsilon \rightarrow 0 \text { and } T \rightarrow \infty \text {, }
\end{aligned}
$$

列出 forward and reverse path SDEs and annealing Langevin equation.

基本是 noise scheduling or annealing Langevin method 解決 (1) 不 robust, (2) gradient of score function (NxN) 不 scalable  問題。

**不過新的問題是 too slow!!  有兩個 loops，而且都是 sequential 或是 markov state transition!!  我們需要先把一個 loop 解決掉，這就是 DDPM 的做法**

- **DDPM: Inner loop 用 denoiser 取代 Langevin drift 可以平行處理，一步到位！**
- **DDIM: outer loop 用 flow 取代 annealing process, 同樣一步到位！**

## Path 2 : DDPM - Diffusion Method (Berkeley)

DDPM paper 從逐步 noise/denoise 出發。使用 Markov Chain 實現，加上用 ELBO 的理論解釋。

Forward add noise (只用於 training):  MCMC : 似乎和 NCSN 的 noisy method 一樣？**理論 NO, 用 MCMC 逐步加 noise**
- **實際 YES,** 直接加上不同程度的 noise 訓練 denoise, 而不是用 MCMC, 因為太慢了。
Reverse path:  把原來的 annealed Lagenvin drift (2 loops: 1 loop for noise/denoise scheduling, 1 loop for score directed drift to higher probability region) 轉換成 denoiser?  因為 denoiser 是已經做好的 IP, 可以把 2 loops 變成 1 loop only focusing on noise/denoise scheduling.

這是 Diffusion method 的真正開山之作

不過我們可以用 NCSN 的框架解釋 DDPM,  這就是下面 paper 的主要訴求。
Random Walks with Tweedie: A Unified Framework for Diffusion Models
https://arxiv.org/pdf/2411.18702

最關鍵的是下表：

![[Pasted image 20250408230947.png]]

SGM (Score-based Generative Model) = NCSN (Noise Conditioned Score Network)
VE-SDE: Variance Explode Stochastic Differential Equation
VP-SDE: Variance Preserve Stocastic Differential Equation


不過 DDPM 利用 MCMC 還是比較好的方法，因為可以把兩個 loops 減少成一個 loop.


### 比較 NCSN vs. DDPM 
相同點
- 都有 outer loop 的 noise/denoise scheduling 部分
不同點 
- NCSN:  NN model the **score function**, with noise/denoise level 是 additional input, **不是 Markov chain**
- DDPM:  NN model the **noise predictor or denoiser**.  不同的 noise/denoise level **是 Markov chain**

更細的比較

### **1. Forward Process (Noise Addition)**

This phase progressively adds noise to a data sample x0x_0x0​, making it more random until it approximates pure noise.

|                             | **NCSN**                                                                                                          | **DDPM**                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| NN model                    | Score function (vector field, 就是 noise/denoise 的方向), 會多一個 noise level input, 因為不是 Markov chain                    Noise predictor, 基本也是 denoise 的方向 (減掉 noise).  沒有 noise level input, 因為是 iterative denoise ,  |
| **Type of Forward Process** | (理論) Continuous-time stochastic process. (實務) Discrete-time stochastic process.                                   | Discrete-time Markov ch                                                                     |
| **Noise Schedule**          | Uses a **variance-preserving** Stochastic Differential Equation (SDE), typically a **Langevin diffusion process** | Uses a predefined **discrete sequence** of noise levels ( q                                 |
| **Noise Distribution**      | Typically an **Ornstein-Uhlenbeck process** or a Variance Exploding (VE) SDE                                      | Gaussian noise with a pre-specified variance sch                                            |

---

### **2. Backward Process (Denoising or Sampling)**

This phase attempts to recover the original sample x0x_0x0​ from noise by running a reverse process.

| |**NCSN**|**DDPM**|
|---|---|---|
|**Backward Process Type**|Uses **score matching** to estimate gradients of the log probability of noisy data|Uses a **discrete denoising process** where each step learns ( p_\theta(x_{t-1}|
|**Score Estimation**|Trains a **score network sθ(xt,t)s_\theta(x_t, t)sθ​(xt​,t)** to estimate **∇xlog⁡pt(x)\nabla_x \log p_t(x)∇x​logpt​(x)**, the gradient of log density|Trains a model ϵθ(xt,t)\epsilon_\theta(x_t, t)ϵθ​(xt​,t) to **predict noise ϵ\epsilonϵ** directly|
|**Sampling Method**|Uses **Langevin dynamics** (a type of stochastic gradient descent)|Uses a **learned denoising process** based on variational inference|
|**Reverse Path**|Continuous reverse SDE (solved via numerical integration)|Discrete reverse Markov chain|



## Path 3 : VDM - Variational Diffusion Method (Google)

其實這個方法在 DDPM 好像有帶到。基本是把 Diffusion 視為 iterative VAE.
利用 ELBO 得到 image.
Path 1/2/3 應該是等價的。



## Path 4:  DDIM: Flow 這是另一條路





## Takeaways

Score matching is the key!  等價於 denoise, why?
看 Tweedie's formula!

x~ = x + sigma^2 + blurred score!

![[Pasted image 20250318112923.png]]

![[Pasted image 20250323001649.png]]

Tweedie’s Formula [8]. InEnglish,Tweedie’s Formulastates that the true mean of an exponential family distribution, given samples drawn from it, can be estimated by the maximum likelihood estimate of the samples (aka empirical mean) plus some correction term involving the score of the estimate. In the case of  just one observed sample,the empirical mean is just the sample itself. It is commonly used to mitigate sample bias; if observed samples all lie on one end of the underlying distribution, then the negative score becomes large and corrects  the naive maximum likelihood estimate of the samples towards the true mean.



DDPM vs. DDIM


類似 discrete case. 

DDPM: predict noise using score matching!  如上式
DDIM: predict $x_0$ directly $E[x_0 \vert x_t]$j.  也就是 flow model
CM: consistency mode:  利用 NN 直接 predict ODE output

![[Pasted image 20250322172344.png]]



DDPM 的兩種解釋。有三條大路通羅馬嗎？
- Lagenven dynamics: score matching,  random walk and reverse walk
- DDPM: noise estimation and denoising!
- Hierarchy VAE: ELBO
- The result is the same!



4.  Diffusion 使用的 chain rule 是 based on Markovian (所以和 Auto-Regressive 不同)
5.  KL Divergence vs. W distance and their close form in Gaussian distribution
6. Mutual information:  KL of P(x, y) // p(x) p(y)

爲什麽 AI 可以處理 ill-conditioned problem?  因爲有 underlying PDF!!  如果我們知道 PDF 或是可以 estimate PDF (或是用 data training 接近  underlying pdf), 我們就可以解決或是 optimize 很多 ill-conditioned problem!!



### Appendix A: Tweedie Estimation

我們使用 $p(x \mid \theta)=\mathcal{N}\left(\theta, \sigma^2\right)$，同时先写出边缘分布的形式 $p(x)=\int_{-\infty}^{\infty} p(x \mid \theta) p(\theta) \mathrm{d} \theta$


$$
\begin{aligned}
&\mathbb{E}[\theta \mid x] \\
& =\int_{-\infty}^{\infty} \theta p(\theta \mid x) \mathrm{d} \theta \\
& =\int_{-\infty}^{\infty} \theta \frac{p(x \mid \theta) p(\theta)}{p(x)} \mathrm{d} \theta \\
& =\frac{\int_{-\infty}^{\infty} \theta p(x \mid \theta) p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\int_{-\infty}^{\infty} \theta \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\sigma)^2}{2 \sigma^2}} p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\int_{-\infty}^{\infty}\left[\sigma^2 \frac{\theta-x}{\sigma^2} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta)+x \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta)\right] \mathrm{d} \theta}{p(x)} \\
& =\frac{\int_{-\infty}^{\infty} \sigma^2 \frac{\theta-x}{\sigma^2} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta) \mathrm{d} \theta+\int_{-\infty}^{\infty} x \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 \sigma^2}} p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \int_{-\infty}^{\infty} \frac{\mathrm{d}\left[\frac{1}{\sqrt{2 \alpha^2}} e^{-\frac{(x-\theta)^2}{2 r^2}}\right]}{\mathrm{d} x} p(\theta) \mathrm{d} \theta+\int_{-\infty}^{\infty} x \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\theta)^2}{2 p^2}} p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \int_{-\infty}^{\infty} \frac{\mathrm{d} p(x \mid \theta)}{\mathrm{d} x} p(\theta) \mathrm{d} \theta+\int_{-\infty}^{\infty} x p(x \mid \theta) p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \frac{\mathrm{~d}}{\mathrm{~d} x} \int_{-\infty}^{\infty} p(x \mid \theta) p(\theta) \mathrm{d} \theta+x \int_{-\infty}^{\infty} p(x \mid \theta) p(\theta) \mathrm{d} \theta}{p(x)} \\
& =\frac{\sigma^2 \frac{\mathrm{~d} p(x)}{\mathrm{d} x}+x p(x)}{p(x)} \\
& =x+\sigma^2 \frac{\mathrm{~d}}{\mathrm{~d} x} \log p(x)
\end{aligned}
$$