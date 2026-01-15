---
title: Math AI - Gaussian Flow
date: 2025-06-08 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2025-0505-Flow_Matching]]"
categories:
  - GenAI
tags:
  - Fisher
  - Flow-Model
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
  - Statistics
---






### Flow $\phi_t$ = 連續的 (自己到自己) 座標變換
**Flow** $\phi_t$ 定義:  $\mathbf{x}_t \triangleq \phi_t\left(\mathbf{x}_0\right)$,  顯然 $t=0 \to \mathbf{x}_0 = \phi_0\left(\mathbf{x}_0\right)$    $\phi_t(\mathbf{x})$ 必須可微分且可逆。

同時 $x_t$  也要滿足運動方程：$\frac{d \mathbf{x}_t}{d t}=u_t(\mathbf{x}_t)$
所以會得到以下的方程式：
$$\frac{d \phi_t(\mathbf{x}_0)}{d t}=u_t\left(\phi_t\left(\mathbf{x}_0\right)\right)$$
因為 $\phi_t(\mathbf{x})$ 是自己到自己的座標變換，在某些情況，可以省掉 subscript.

$$\frac{d \phi_t(\mathbf{x})}{d t}=u_t\left(\phi_t\left(\mathbf{x}\right)\right)$$
有無窮多的 $\phi_t$ and $u_t$ 滿足 $p_0(x)$ 到 $p_1(x)$ 的分佈轉換。最直接而且簡單的就是 linear interpolation flow.   $x_t = t x_0 + (1-t) x_1$

Gaussian Flow 就是假設 $x_0$ and $x_1$  都是 Gaussian distributions.  因此 linear interpolation 都是 Gaussian distribution.   這樣的 flow 稱爲 Gaussian flow. 

The vector field $\mathbf{u}_t(\mathbf{x})$ for flow matching between two **independent** Gaussian distributions $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ is derived under a **Gaussian probability path** where the mean and covariance are linearly interpolated:

$$p_t = \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$$
$$
\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1
$$
$$
\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1
$$

The vector field is given by:

$$
\mathbf{u}_t(x) = \underbrace{\dot{\boldsymbol{\mu}}_t}_{\text{Mean component}\,} + \underbrace{\frac{1}{2} \dot{\boldsymbol{\Sigma}_t} \boldsymbol{\Sigma}_t^{-1} ((\mathbf{x} - \boldsymbol{\mu}_t)}_{\text{Covariance component}}
$$
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + ( t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0)\boldsymbol{\Sigma}_t^{-1}  (\mathbf{x} - \boldsymbol{\mu}_t)}
$$

If $\boldsymbol{\Sigma}_0$ and $\boldsymbol{\Sigma}_1$ **commute**, the flow simplifies to: 
$$
\boxed{\phi_t(\mathbf{x}_0) = \mathbf{x}_t = \boldsymbol{\mu}_t + \boldsymbol{\Sigma}_t^{1/2} \boldsymbol{\Sigma}_0^{-1/2} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)}
$$

Continuity equation (conservation of probability)
$$
\frac{d}{dt} \log p_t(\mathbf{x}) = - \nabla \cdot \mathbf{u}_t(\mathbf{x}) =-\frac{\mathbf{d}}{2}\frac{ d}{dt}\log\Sigma_t\quad \mathbf{d}\text{ is the dimension}
$$

## 兩類 Gaussian Flow

### Uncondition Gaussian Flow

上述公式就是 uncondition Gaussian flow.   

如果 $\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1$,  $\boldsymbol{\Sigma}_t$ 變成兩端寬 (t = 0 and 1), 但是中間窄的 flow,  如下圖


![[Pasted image 20250604104124.png]]

### Special case:  isotropic Gaussian flow

Under independent coupling:
- $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$
- $\mathbf{x}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$

With linear interpolation $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$, the marginal is Gaussian:
$$
\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \sigma_t^2 \mathbf{I}),
$$
where:
- **Mean**: $\boldsymbol{\mu}_t = (1-t)\boldsymbol{\mu}_0 + t\boldsymbol{\mu}_1$
- **Variance**: $\sigma_t^2 = (1-t)^2 \sigma_0^2 + t^2 \sigma_1^2$


The form below **holds generally for isotropic Gaussians** $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$ under independent coupling.  而且下式是座標無關形式！
$$\mathbf{u}_t(\mathbf{x}) = \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma_t^{2}}}{2\sigma_t^{2}} (\mathbf{x} - \boldsymbol{\mu}_t)= \dot{\boldsymbol{\mu}}_t + \frac{\dot{\sigma_t}}{\sigma_t} (\mathbf{x} - \boldsymbol{\mu}_t)$$
**Final form:**
$$\boxed{\mathbf{u}_t(x) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left(\dfrac{-(1-t){\sigma}_0^2 + t{\sigma}_1^2}{(1-t)^2{\sigma}_0^2 + t^2{\sigma}_1^2}\right)(\mathbf{x} - \boldsymbol{\mu}_t)}$$


If $\sigma_0^2=\sigma_1^2$ and $\boldsymbol{\mu}_0 = -\boldsymbol{\mu}$ and $\boldsymbol{\mu}_1 = +\boldsymbol{\mu}$, 上式可以進一步簡化：
$$
\boxed{\mathbf{u}_t(\mathbf{x}) = \frac{(2t-1)\mathbf{x} + \boldsymbol{\mu}}{2t^2-2t+1} }
$$

$$
\boxed{\phi_t(\mathbf{x}_0) = \mathbf{x}_t = \boldsymbol{\mu}_t + \frac{ {\sigma}_t} { {\sigma}_0} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right)=(2t-1) \boldsymbol{\mu}+\sqrt{2t^2-2t+1}\left( \mathbf{x}_0 + \boldsymbol{\mu} \right)}
$$
**注意這個 uncondition flow （從 $\mathbf{x}_0$ 到 $\mathbf{x}_0+2\boldsymbol{\mu}$）流不是直綫！**

同樣 conservation of probability 如下，也是座標無關形式！
$$
\boxed{\frac{d}{dt} \log p_t(\mathbf{x}) = - \nabla \cdot \mathbf{u}_t(\mathbf{x}) = -\frac{\mathbf{d}}{2}\frac{ d}{dt}\log\sigma_t^2=-\mathbf{d}\frac{ d}{dt}\log\sigma_t=-\mathbf{d}\frac{2t-1}{2t^2-2t+1}}
$$



### Condition Gaussian Flow, given $\mathbf{x}_1$ sampled from $p_1(\mathbf{x})$

#### Condition Gaussian Flow 視爲 Uncondition Gaussian Flow 的特例

Under independent coupling:
- $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$
- $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{x}_1, 0)$

With linear interpolation $\mathbf{x}_t \mid \mathbf{x}_1 = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$, the conditional is Gaussian:
$$
\mathbf{x}_t(\mathbf{x}_1) \sim \mathcal{N}(\boldsymbol{\mu}_t(\mathbf{x}_1), \sigma_t^2(\mathbf{x}_1) \mathbf{I}),
$$
where:
- **Mean**: $\boldsymbol{\mu}_t(\mathbf{x}_1) = (1-t)\boldsymbol{\mu}_0 + t\mathbf{x}_1$
- **Variance**: $\sigma_t^2(\mathbf{x}_1) = (1-t)^2 \sigma_0^2$


The form below **holds generally for isotropic Gaussians** $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$ and $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$ under independent coupling.  而且下式是座標無關形式！
$$\mathbf{u}_t(\mathbf{x}\mid \mathbf{x}_1) = \dot{\boldsymbol{\mu}}_t(\mathbf{x}_1) + \frac{\dot{\sigma_t^{2}}(\mathbf{x}_1)}{2\sigma_t^{2}(\mathbf{x}_1)} (\mathbf{x} - \boldsymbol{\mu}_t(\mathbf{x}_1))= \dot{\boldsymbol{\mu}}_t(\mathbf{x}_1) + \frac{\dot{\sigma_t}(\mathbf{x}_1)}{\sigma_t(\mathbf{x}_1)} (\mathbf{x} - \boldsymbol{\mu}_t(\mathbf{x}_1))$$

上式可以進一步簡化：
$$\begin{align}
\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) &=  \dot{\boldsymbol{\mu}}_t(\mathbf{x}_1) + \frac{\dot{\sigma_t}(\mathbf{x}_1)}{\sigma_t(\mathbf{x}_1)} (\mathbf{x}_t - \boldsymbol{\mu}_t(\mathbf{x}_1)) \\
&= \mathbf{x}_1 - \boldsymbol{\mu}_0 + \frac{-\sigma_0}{(1-t)\sigma_0}(\mathbf{x}_t-\boldsymbol{\mu}_t(\mathbf{x}_1)) \\
&=\frac{(\mathbf{x}_1 - \boldsymbol{\mu}_0)(1-t)}{1-t} + \frac{-\mathbf{x}_t+(1-t)\boldsymbol{\mu}_0 + t\mathbf{x}_1}{1-t}\\
&=\frac{\mathbf{x}_1 -\mathbf{x}_t}{1-t} \\
\end{align}$$
代入  $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$, 上公式有一個非常直觀的幾何解釋，就是一條直綫的向量從 $\mathbf{x}_0$ 指向 $\mathbf{x}_1$。and $t=1$.  所以 condition Gaussian flow 是一個圓錐流從 $p_0(\mathbf{x})$ distribution 流入一個點 $\mathbf{x}_1$.  

$$\begin{align}
\boxed{\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) =\frac{\mathbf{x}_1 -\mathbf{x}_t}{1-t}=\mathbf{x}_1 -\mathbf{x}_0} \\
\end{align}$$


![[Pasted image 20250514121948.png]]

Condition Flow:
$$\begin{align}
\psi_t(\mathbf{x}_0\mid \mathbf{x}_1) &= \mathbf{x}_t \mid \mathbf{x}_1 = \boldsymbol{\mu}_t + \frac{ {\sigma}_t} { {\sigma}_0} \left( \mathbf{x}_0 - \boldsymbol{\mu}_0 \right) \\
&= (1-t)\boldsymbol{\mu}_0 + t\mathbf{x}_1 + (1-t)(\mathbf{x}_0 - \boldsymbol{\mu}_0)\\
&=  (1-t)\mathbf{x}_0 + t\mathbf{x}_1
\end{align}$$
$$
\boxed{\psi_t(\mathbf{x}_0\mid \mathbf{x}_1)= (1-t)\mathbf{x}_0 + t\mathbf{x}_1}
$$
**No surprise： 注意  這個 condition flow 流是從 $\mathbf{x}_0$ 到給定 $\mathbf{x}_1$ 直綫！**

同樣 conservation of probability 如下，也是座標無關形式！
$$
\boxed{\frac{d}{dt} \log p_t(\mathbf{x}\mid \mathbf{x}_1) = - \nabla \cdot \mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_1) = -\frac{\mathbf{d}}{2}\frac{ d}{dt}\log\sigma_t^2(\mathbf{x}_1)=-\mathbf{d}\frac{ d}{dt}\log\sigma_t(\mathbf{x}_1)=\frac{\mathbf{d}}{1-t}}
$$



### Conditional Gaussian Probability
接下來要計算 condition flow.  在此之前先看 conditional probability (都是 Gaussian):

| Conditional      | Mean                                                             | Covariance                                                           |
| ---------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| $p(x_t\mid x_0)$ | $(1-t)x_0 + t\,\mu_1$                                            | $t^2\,\Sigma_1$                                                      |
| $p(x_t\mid x_1)$ | $(1-t)\,\mu_0 + t\,x_1$                                          | $(1-t)^2\,\Sigma_0$                                                  |
| $p(x_1\mid x_t)$ | $\displaystyle\mu_1 + t\,\Sigma_1\,\Sigma_t^{-1}(x_t-\mu_t)$     | $\displaystyle\Sigma_1 - t^2\,\Sigma_1\,\Sigma_t^{-1}\,\Sigma_1$     |
| $p(x_0\mid x_t)$ | $\displaystyle\mu_0 + (1-t)\,\Sigma_0\,\Sigma_t^{-1}(x_t-\mu_t)$ | $\displaystyle\Sigma_0 - (1-t)^2\,\Sigma_0\,\Sigma_t^{-1}\,\Sigma_0$ |

**我們可以反過來從 condition Gaussian flow 推論 uncondition Gaussian flow 或是其他任何的 $p_1(x)$ distribution.   這正是 Lipman 在 flow matching 使用的方法。**
#### Uncondition Gaussian Flow 視爲 Condition Gaussian Flow 的特例


從 condition flow 直觀的幾何解釋，就是一條直綫的向量從 $\mathbf{x}_0$ 指向 $\mathbf{x}_1$。and $t=1$.  所以 condition Gaussian flow 是一個圓錐流從 $p_0(\mathbf{x})$ distribution 流入一個點 $\mathbf{x}_1$.  
$$\begin{align}
\boxed{\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) =\frac{\mathbf{x}_1 -\mathbf{x}}{1-t}} \\
\end{align}$$
對每個 $x_1$ 積分，the marginal vector field is:
$$
u_t(x) = \mathbb{E}_{x_1 \sim p_{1|t}} \left[ u_t(x \mid x_1) \right] = \int u_t(x \mid x_1) {p_t(x_1 \mid x)} dx_1
$$

1. $p_t(x \mid x_1) = \mathcal{N}(x; tx_1 - (1-t)\boldsymbol{\mu}, (1-t)^2\mathbf{I})$
2. $p_t(x) = \mathcal{N}(x; (2t-1)\boldsymbol{\mu}, (2t^2-2t+1)\mathbf{I})= \mathcal{N}(x; \boldsymbol{\mu}_t, \sigma^2_t \mathbf{I})$ (marginal distribution) 
	-  $\boldsymbol{\mu}_t = (2t-1)\boldsymbol{\mu} , \,\,\sigma^2_t = 2t^2-2t+1$

The posterior is Gaussian:
$$\begin{align}
p_{1|t}(x_1|x) &= \mathcal{N}\left( x_1; \frac{tx + (1-t)\boldsymbol{\mu}}{2t^2-2t+1}, \frac{(1-t)^2}{2t^2-2t+1} \mathbf{I}\right)\\
&= \mathcal{N}\left( x_1; \frac{tx + (1-t)\boldsymbol{\mu}}{\sigma^2_t}, \frac{(1-t)^2}{\sigma^2_t} \mathbf{I}\right)\\
\end{align}$$

The conditional vector field should be: 
$$
\begin{aligned}
\mathbf{u}_t(x) & =\mathbb{E}_{x_1 \sim p_{1 \mid t}}\left[\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1)\right] \\
& =\frac{\mathbb{E}_{x_1 \sim p_{1 \mid t}}[\mathbf{x}_1] -\mathbf{x}}{1-t} \\
&= \frac{\boldsymbol{\mu}_1 + t\,\boldsymbol{\Sigma}_1\,\boldsymbol{\Sigma}_t^{-1}(\mathbf{x}-\boldsymbol{\mu}_t)-\mathbf{x}}{1-t}\\
&= (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + ( t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0)\boldsymbol{\Sigma}_t^{-1}  (\mathbf{x} - \boldsymbol{\mu}_t)
\end{aligned}
$$
最後一步的推導非常繁瑣，我用 DeepSeek 證明。

另一個比較簡單的方法：
$$\begin{align}
\boxed{\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) =\frac{\mathbf{x}_1 -\mathbf{x}}{1-t}= \mathbf{x}_1-\mathbf{x}_0} \\
\end{align}$$
$$
\begin{aligned}
\mathbf{u}_t(x) & =\mathbb{E}_{x_1 \sim p_{1 \mid t}}\left[\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1)\right] \\
& =\mathbb{E}_{x_1 \sim p_{1 \mid t}}\left[\mathbf{x}_1-\mathbf{x}_0\right] \\
& =\mathbb{E}_{x_1 \sim p_{1 \mid t}}\left[\mathbf{x}_1\right] -  \mathbb{E}_{x_0 \sim p_{0 \mid t}}\left[\mathbf{x}_0\right]\\
&= {\boldsymbol{\mu}_1 + t\,\boldsymbol{\Sigma}_1\,\boldsymbol{\Sigma}_t^{-1}(\mathbf{x}-\boldsymbol{\mu}_t)}-(\boldsymbol{\mu}_0 + (1-t)\,\boldsymbol{\Sigma}_0\,\boldsymbol{\Sigma}_t^{-1}(\mathbf{x}-\boldsymbol{\mu}_t))\\
&= (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + ( t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0)\boldsymbol{\Sigma}_t^{-1}  (\mathbf{x} - \boldsymbol{\mu}_t)
\end{aligned}
$$
這裏的關鍵是第二步到第三步爲什麽成立? Appendix B.

或是利用更 general 的 linear separable form, **不需要 $x_0, x_1$ 是 Gaussian, 或是 independent.**
Given: $\mathbf{x}_t = f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1)$ (linear separable form) $$
  \mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t)} [f_0(\mathbf{x}_0)] = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t)} [f_0(\mathbf{x}_0)]
  $$
因為 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ 滿足 linear separable form,  所以
$$
  \mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t)} [\mathbf{x}_0] = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t)} [\mathbf{x}_0]
  $$

重點是 $x_t$?


## Full-Rank to Low-Rank Gaussian Flow

Condition Gaussian Flow 基本是一種 degenerate flow, 因為最後收斂到一點 ($\mathbf{x}_1$).
一個自然的問題是 $p_1$ 是一個 low-rank Gaussian, i.e. 
$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$
$$p_t = \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$$
$$
\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1
$$
$$
\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1
$$
where $\boldsymbol{\Sigma}_1$ has rank $r<\mathbf{d}$, the full rank dimension.

基本結論一樣：

- **向量场**：$\mathbf{u}_t(\mathbf{x}) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \left[ t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t)$


- 对于 $t \in [0, 1)$，$\boldsymbol{\Sigma}_t$ 是正定可逆的（因为 $(1-t)^2 \boldsymbol{\Sigma}_0$ 正定，且 $t^2 \boldsymbol{\Sigma}_1$ 半正定），因此 $\boldsymbol{\Sigma}_t^{-1}$ 存在。
- 在 $t = 1$ 时，$\boldsymbol{\Sigma}_t = \boldsymbol{\Sigma}_1$ 可能奇异（秩 $r < d$），此时向量场可能未定义或需要特殊处理（如使用伪逆），但通常考虑 $t < 1$。
- 由于 $\boldsymbol{\Sigma}_1$ 是低秩（秩 $r$)，可以利用矩阵求逆引理（Sherman-Morrison-Woodbury）简化 $\boldsymbol{\Sigma}_t^{-1}$。设 $\boldsymbol{\Sigma}_1 = U \Lambda U^T$，其中 $U$ 是 $d \times r$ 矩阵（列正交），$\Lambda$ 是 $r \times r$ 正定对角矩阵。则：
  $$
  \boldsymbol{\Sigma}_t^{-1} = \frac{1}{(1-t)^2} \left( \boldsymbol{\Sigma}_0^{-1} - \boldsymbol{\Sigma}_0^{-1} U \left[ \Lambda^{-1} + \frac{t^2}{(1-t)^2} (U^T \boldsymbol{\Sigma}_0^{-1} U) \right]^{-1} U^T \boldsymbol{\Sigma}_0^{-1} \right)
  $$

- **流**：解 ODE $\frac{d\mathbf{x}_t}{dt} = v_t(\mathbf{x}_t)$ 初值 $\mathbf{x}_0$，流映射为 $\mathbf{x}_t = \boldsymbol{\mu}_t + \Phi(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0)$，其中 $\Phi(t)$ 满足 $\frac{d\Phi}{dt} = A(t) \Phi$，$\Phi(0) = I$。


### 重要的特例：$\Sigma_0 = \sigma^2_0 I$ 
#### 总结：简化后的表达式
| 组件        | 简化表达式                                                                                                                                                                                                                                                                                                                                                    |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **协方差矩阵** | $\boldsymbol{\Sigma}_t = (1-t)^2 \sigma_0^2 \mathbf{I} + t^2 \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$                                                                                                                                                                                                                                                   |
| **向量场**   | $v_t(\mathbf{x}) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \mathbf{U} \mathbf{A}_t \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}_t) - \dfrac{1}{1-t} (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x} - \boldsymbol{\mu}_t)$ <br> $\mathbf{A}_t = \text{diag}\left( \dfrac{t \lambda_i - (1-t) \sigma_0^2}{(1-t)^2 \sigma_0^2 + t^2 \lambda_i} \right)$ |
| **流映射**   | $\mathbf{x}_t = \boldsymbol{\mu}_t + \mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0) + (1-t) (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0)$ <br> $\mathbf{D}_t = \text{diag}\left( \dfrac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_i } }{ \sigma_0 } \right)$                                       |

### 关键特性：
1. **显式解析解**：流映射有闭式表达式，无需数值积分
2. **子空间行为**：
   - $\text{Col}(\mathbf{U})$ 上：缩放 $\mathbf{D}_t$（依赖 $\lambda_i$）
   - $\text{Col}(\mathbf{U})^\perp$ 上：线性收缩 $(1-t)$
1. **边界一致性**：
   - $t=0$：$\mathbf{D}_0 = \mathbf{I}_r$，流映射为恒等变换
   - $t=1$：$\mathbf{D}_1 = \frac{1}{\sigma_0} \sqrt{\mathbf{\Lambda}}$，$\boldsymbol{\Sigma}_t = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$
1. **计算效率**：仅需特征值分解 $\boldsymbol{\Sigma}_1$，避免高维矩阵求逆

在向量场和流映射的表达式中，**收缩项**（contraction term）指的是正交补空间上的分量：
$$
-\dfrac{1}{1-t} (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x} - \boldsymbol{\mu}_t)
$$
和流映射中的对应项：
$$
(1-t) (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0)
$$


#### 比較 Low-to-High-Rank Diffusion (Reverse 是 High to Low Rank)
Reference: unreasonable effectiveness of Gaussian Score Approx. for Diffusion

注意：Diffusion 的 convention 和 flow 相反：Diffusion $x_T \equiv x_0$ in flow;  Diffusion $x_0 \equiv x_1$ in flow.

#### VE ODE

we allow $\boldsymbol{\Sigma}$ have rank $r \leq D$. At noise scale $\sigma$, the score is that of $\mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma}+\sigma^2 \mathbf{I}\right)$, which reads
假設 $p_0 \sim N\left(\mu, \sigma_{\min }^2 \sim 0\right)$ and $p_T \sim N\left(\mu, \sigma_{\max}^2\right)$

$$
\mathbf{D}(\mathbf{x}, \sigma)=\boldsymbol{\mu}+\sum_{k=1}^r \frac{\lambda_k}{\lambda_k+\sigma^2}\left[\mathbf{u}_k \cdot(\mathbf{x}-\boldsymbol{\mu})\right] \mathbf{u}_k
$$

$$
\dot{\mathbf{x}}_t=\frac{\dot{\sigma}}{\sigma}\left(\mathbf{I}-\mathbf{U} \tilde{\boldsymbol{\Lambda}}_\sigma \mathbf{U}^T\right)\left(\mathbf{x}_t-\mu\right)
$$
 
The above VE ODE is linear, and its dynamics along each principal axis $\boldsymbol{u}_k$ are independent. Solving it in the usual way (see Appendix D.1), we find
![[Pasted image 20250614231415.png]]
$$
\begin{align}
\mathbf{x}_t & =\boldsymbol{\mu}+\frac{\sigma_t}{\sigma_T} \mathbf{x}_T^{\perp}+\sum_{k=1}^r \psi\left(t, \lambda_k\right) c_k(T) \mathbf{u}_k & \psi(t, \lambda):=\sqrt{\frac{\sigma_t^2+\lambda}{\sigma_T^2+\lambda}} \\
\mathbf{x}_T^{\perp}&=\left(\mathbf{I}-\mathbf{U U}^T\right)\left(\mathbf{x}_T-\boldsymbol{\mu}\right) & c_k(T):=\mathbf{u}_k^T\left(\mathbf{x}_T-\boldsymbol{\mu}\right) .
\end{align}
$$
#### VP ODE
Transition probability $p\left(\mathbf{x}_t \mid \mathbf{x}_0\right)=\mathcal{N}\left(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I}\right)$. 
VP-SDE is equivalent to introducing a time-dependent scaling term $\alpha_t$ in Eq. 2. Thus, we can obtain the solution by substituting $\mathbf{x}_t \mapsto \mathbf{x}_t / \alpha_t$ and $\sigma_t \mapsto \sigma_t / \alpha_t$. The solution reads

$$
\begin{gathered}
\mathbf{x}_t=\alpha_t \boldsymbol{\mu}+\frac{\sigma_t}{\sigma_T} \overline{\mathbf{x}}_T^{\perp}+\sum_{k=1}^r \bar{\psi}\left(t, \lambda_k\right) \bar{c}_k(T) \mathbf{u}_k \quad \bar{\psi}(t, \lambda):=\sqrt{\frac{\sigma_t^2+\lambda \alpha_t^2}{\sigma_T^2+\lambda \alpha_T^2}} \\
\overline{\mathbf{x}}_T^{\perp}:=\left(\mathbf{I}-\mathbf{U}^T \mathbf{U}\right)\left(\mathbf{x}_T-\alpha_T \boldsymbol{\mu}\right) \quad \bar{c}_k(T):=\mathbf{u}_k^T\left(\mathbf{x}_T-\alpha_T \boldsymbol{\mu}\right) \\
\mathbf{D}(t)=\boldsymbol{\mu}+\sum_{k=1}^r \bar{\xi}\left(t, \lambda_k\right) \bar{c}_k(T) \mathbf{u}_k \quad \bar{\xi}(t, \lambda):=\frac{\alpha_t \lambda}{\sqrt{\left(\alpha_t^2 \lambda+\sigma_t^2\right)\left(\alpha_T^2 \lambda+\sigma_T^2\right)}}
\end{gathered}
$$


Diffusion 包含三個 components: (i) the distribution mean, (ii) an off-manifold component, and (iii) an on-manifold component.   我們可以和 Flow 對比

以下是 Flow:

$$
\mathbf{x}_t = \boldsymbol{\mu}_t + \underbrace{\sum_{k=1}^r d_k(t) \langle \mathbf{u}_k, \mathbf{x}_0 - \boldsymbol{\mu}_0 \rangle \mathbf{u}_k}_{\text{子空间 } \text{Col}(\mathbf{U}) \text{ 上的缩放}} + \underbrace{(1-t) (\mathbf{I} - \mathbf{U}\mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0)}_{\text{正交补空间收缩}}
$$

- Distribution mean
	- Flow: $\mu_t$,   
	- Diffusion: $\mu$ 應該一樣？ 
- Off-manifold component  
	- Flow $\mathbf{x}_t^{\perp}= (1-t) (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0)$.   $t\to 1$ 線性收斂到 0
	- Diffusion:  $\mathbf{x}_t^{\perp}=\frac{\sigma_t}{\sigma_T}\left(\mathbf{I}-\mathbf{U U}^T\right)\left(\mathbf{x}_T-\boldsymbol{\mu}\right)$.    $t\to 0$ 收斂到 $\frac{\sigma_{min}}{\sigma_{max}}$ 乘上 $x_T-\mu$ 在補空間投影。$\sigma_t$ 應該不是 linear in t?  而是等比？depends on VE or VP?
- On-manifold component
	- Flow =  $\mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)$ where $\mathbf{D}_t = \text{diag}\left( \dfrac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_i } }{ \sigma_0 } \right)$ or
	- Flow = $\sum\limits_{k=1}^r d_k(t) \langle \mathbf{u}_k, \mathbf{x}_0 - \boldsymbol{\mu}_0 \rangle \mathbf{u}_k$  where $d_k(t) = \dfrac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_k } }{ \sigma_0 }$
- 
	- VE (constant mean) Diffusion = $\sum\limits_{k=1}^r \psi_k(t) \langle \mathbf{u}_k, \mathbf{x}_T - \boldsymbol{\mu} \rangle  \mathbf{u}_k$  where  $\psi_k(t)=\sqrt{\dfrac{\sigma_t^2+\lambda_k}{\sigma_T^2+\lambda_k}}$



Solution with data scaling term. The popular VP-SDE (Song et al., 2021) is an alternative to the EDM formulation, and its forward process is characterized by the Using the VP-SDE is equivalent to introducing a time-dependent scaling term $\alpha_t$ in Eq. 2. Thus, we can obtain the solution by substituting $\mathbf{x}_t \mapsto \mathbf{x}_t / \alpha_t$ and $\sigma_t \mapsto \sigma_t / \alpha_t$. The solution reads

$$
\begin{gathered}
\mathbf{x}_t=\alpha_t \mu+\frac{\sigma_t}{\sigma_T} \overline{\mathbf{x}}_T^{\perp}+\sum_{k=1}^r \bar{\psi}\left(t, \lambda_k\right) \bar{c}_k(T) \mathbf{u}_k \quad \bar{\psi}(t, \lambda):=\sqrt{\frac{\sigma_t^2+\lambda \alpha_t^2}{\sigma_T^2+\lambda \alpha_T^2}} \\
\overline{\mathbf{x}}_T^{\perp}:=\left(\mathbf{I}-\mathbf{U}^T \mathbf{U}\right)\left(\mathbf{x}_T-\alpha_T \mu\right) \quad \bar{c}_k(T):=\mathbf{u}_k^T\left(\mathbf{x}_T-\alpha_T \mu\right) \\
\mathbf{D}(t)=\mu+\sum_{k=1}^r \bar{\xi}\left(t, \lambda_k\right) \bar{c}_k(T) \mathbf{u}_k \quad \bar{\xi}(t, \lambda):=\frac{\alpha_t \lambda}{\sqrt{\left(\alpha_t^2 \lambda+\sigma_t^2\right)\left(\alpha_T^2 \lambda+\sigma_T^2\right)}}
\end{gathered}
$$

![[Pasted image 20250614162432.png]]

he distribution mean term does not change throughout sample ge.. eration.  The off-manifold component shrinks to zero as $t \rightarrow 0$. The on-manifold component, which is determined by the manifold-projected difference between $\mathbf{x}$ and $\boldsymbol{\mu}$, evolves independently according to $\psi(t, \lambda)$ along each PC direction.


![[Pasted image 20250607094307.png]]
假設 $p_0 \sim N(\mu_0, \sigma^2_{min}\sim 0)$ and  $p_T \sim N(\mu_T, \sigma^2_{max})$ 

![[Pasted image 20250613164514.png]]



## Low-Rank to Low-Rank Gaussian Flow (好像用途有限？)

$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$
$$p_t = \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$$
$$
\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1
$$
$$
\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1
$$
where $\boldsymbol{\Sigma}_0$ and $\boldsymbol{\Sigma}_1$ have rank $r<\mathbf{d}$, the full rank dimension.

### 总结
| 组件       | 表达式                                                                                                                                                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **向量场**  | $v_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left[ t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{+} (\mathbf{x} - \boldsymbol{\mu}_t)$                                         |
| **流映射**  | $\mathbf{x}_t = \boldsymbol{\mu}_t + \Phi(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0)$ <br> $\frac{d\Phi(t)}{dt} = \left[ t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{+} \Phi(t), \quad \Phi(0) = I$ |
| **概率守恒** | 连续性方程成立，概率质量守恒                                                                                                                                                                                                                         |

### 关键说明：
1. **伪逆处理低秩**：  
   - $\boldsymbol{\Sigma}_t^{+}$ 在 $\boldsymbol{\Sigma}_t$ 秩退化时至关重要（如 $t=0,1$ 或 $\text{Col}(\boldsymbol{\Sigma}_0) \cap \text{Col}(\boldsymbol{\Sigma}_1) \neq \{0\}$)。
   - 伪逆计算：若 $\boldsymbol{\Sigma}_t = U_t D_t U_t^T$（特征值分解），则 $\boldsymbol{\Sigma}_t^{+} = U_t D_t^{+} U_t^T$，其中 $D_t^{+}$ 将非零对角元取倒数。

2. **数值稳定性**：  
   - 当 $t \to 0^+$ 或 $t \to 1^-$ 时，$\boldsymbol{\Sigma}_t$ 可能接近奇异，需正则化或子空间投影。


向量場僅在前 $r$ 維子空間（由 $\mathbf{U}$ 張成）上有非零值，而後 $d-r$ 維上，由於 $\mathbf{U}\mathbf{A}_t\mathbf{U}^T$ 的作用，結果為0，所以：

$v_t(\mathbf{x}) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \text{（僅在前 r 個分量上的線性變換）}$

如果我們將 $\mathbf{x}$ 分解為 $\mathbf{x} = \mathbf{x}_{\parallel} + \mathbf{x}_{\perp}$，其中 $\mathbf{x}_{\parallel} = \mathbf{U}\mathbf{U}^T\mathbf{x}$ 是投影到子空間的分量，$\mathbf{x}_{\perp}$ 是正交補空間的分量，則：

$\mathbf{U}^T \mathbf{x}_{\perp} = 0$，所以 $\mathbf{U}\mathbf{A}_t\mathbf{U}^T \mathbf{x}_{\perp} = 0$。

因此，向量場在正交補空間上為常數 $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$，而在子空間上是一個線性變換。

然而，注意 $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$ 是一個常數向量，但我們可以將其分解為子空間分量和正交分量。由於分佈僅在子空間上有變化，我們假設 $\boldsymbol{\mu}_0$ 和 $\boldsymbol{\mu}_1$ 也在該子空間上？不一定，但我們沒有此假設。所以 $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$ 可能包含正交分量。


### 簡化版 $p_0$ 是 low-rank 而且 isotropic

假設 $\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{I}_r$,  也就是 rank r 的 identity matrix.  簡化上面的推導

| 組件 | 簡化後的表達式 |
|------|----------------|
| **協方差矩陣** | $\boldsymbol{\Sigma}_t = \mathbf{U} \mathbf{S}_t \mathbf{U}^T$ <br> $\mathbf{S}_t = \text{diag}\left( (1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)} \right)$ |
| **向量場** | $v_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \mathbf{U} \mathbf{A}_t \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}_t)$ <br> $\mathbf{A}_t = \text{diag}\left( \frac{t \lambda_1^{(i)} - (1-t) \sigma_0^2}{(1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)}} \right)$ |
| **流映射** | $\mathbf{x}_t = \boldsymbol{\mu}_t + \mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)$ <br> $\mathbf{D}_t = \text{diag}\left( \frac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)} } }{ \sigma_0 } \right)$ |
| **概率守恒** | 成立，且流映射是解析的 |

如果假設 $\boldsymbol{\Sigma}_1 = \sigma_1^2 \mathbf{I}_r$, i.e. $\lambda_1^{(i)}=\sigma_1^2$,  $U=U^T=I$, 就回到 isotropic case.



再來是變化球，就是 Gaussian flow on Riemann manifold!
這是從歐式空間的擴展。

## Gaussian Flow on General Geometry

$p: \mathcal{M}\to \mathbb{R}_+$ and $\int_{\mathcal{M}} p(x) dv = 1$
Equation 1 是 flow 和 vector field 的關係 on manifold.
Equation 2 是 conservation of probability.
這裡和歐式空間的 Gaussian flow 都一樣，除了是在 manifold 上的微分和積分。
![[Pasted image 20250610101032.png]]

再來是關鍵的部分：**condition vector field!**

一樣
![[Pasted image 20250610104338.png]]

![[Pasted image 20250610104401.png]]

#### Flow matching 三部曲
![[Pasted image 20250610104510.png]]
![[Pasted image 20250610104520.png]]
![[Pasted image 20250610104546.png]]

重點在歐式空間直線變成黎曼空間的測地線 (geodesic), $\psi_t(x_0\mid x_1)$ 是 geodesic

![[Pasted image 20250610104957.png]]

#### 不同的地方：
d 是 manifold 上的距離。對應歐式空間的 $d(x, x_1) = \|x-x_1\|$
$\kappa(t)$ 是 scheduling, $\kappa(0)=1, \kappa(1)=0$.  對應 flow matching 使用 linear scheduling  $\kappa(t) = 1-t$

![[Pasted image 20250610112047.png]]
Equation (13) 是普遍的表示法

$$
u_t(x \mid x_1) = \frac{d \log \kappa(t)}{dt} \, d(x, x_1) \cdot \frac{\nabla d(x, x_1)}{\|\nabla d(x, x_1)\|_g^2}
$$

可以證明 $d(x_t, x_1)=\kappa(t) d(x, x_1)$, 所以可以簡化：
![[Pasted image 20250610181313.png]]


檢查 **Euclidean metric**:

* $d(x, x_1) = \|x - x_1\|$ (Euclidean norm)
* $\nabla d(x, x_1) = \frac{x - x_1}{\|x - x_1\|}$
* $\|\nabla d(x, x_1)\|^2 = 1$
-  $\kappa(t) = 1-t$

Then the vector field becomes:
$$
u_t(x \mid x_1) = \frac{d \log \kappa(t)}{dt} \|x - x_1\| \cdot \left(\frac{x - x_1}{\|x - x_1\|} \right)
= \frac{d \log \kappa(t)}{dt} (x - x_1)
$$
where $\frac{d \log \kappa(t)}{dt}=\frac{-1}{1-t}$.

So it simplifies to:
$$
u_t(x \mid x_1) = \frac{x_1-x}{1-t}
$$
所以我們後面都用 $\kappa(t)=1-t$
![[Pasted image 20250610181515.png]]
![[Pasted image 20250610181540.png]]

等價於歐式空間
$$
x_t = \exp_{x_1}\left( \kappa(t) \cdot \log_{x_1}(x_0) \right)
= x_1 + \kappa(t)(x_0 - x_1) =(1 - \kappa(t))x_1 + \kappa(t)x_0
$$
![[Pasted image 20250610182751.png]]





## Reference

MIT 6.S. 84: Flow Matching and Diffusion Models https://www.youtube.com/watch?v=GCoP2w-Cqtg&t=28s&ab_channel=PeterHolderrieth

Yaron Meta paper: [[2210.02747] Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

An Introduction to Flow Matching: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html



## Appendix 
### Appendix A: Conditional Gaussian Distribution


We have two independent Gaussians

$$
x_0 \sim \mathcal{N}(\mu_0,\Sigma_0), 
\quad
x_1 \sim \mathcal{N}(\mu_1,\Sigma_1),
$$

and define the linear interpolation

$$
x_t = (1-t)\,x_0 + t\,x_1.
$$

One checks immediately that

$$
x_t \sim \mathcal{N}\bigl(\mu_t,\Sigma_t\bigr),
\quad
\mu_t = (1-t)\,\mu_0 + t\,\mu_1,
\quad
\Sigma_t = (1-t)^2\,\Sigma_0 \;+\; t^2\,\Sigma_1.
$$

---

## 1. $p(x_t\mid x_0)$

Since $x_t = (1-t)x_0 + t\,x_1$ with $x_1$ independent,

$$
x_t \mid x_0 
\;\sim\; 
\mathcal{N}\!\bigl((1-t)x_0 + t\,\mu_1,\;t^2\,\Sigma_1\bigr).
$$

---

## 2. $p(x_t\mid x_1)$

Likewise,

$$
x_t \mid x_1
\;\sim\;
\mathcal{N}\!\bigl((1-t)\,\mu_0 + t\,x_1,\;(1-t)^2\,\Sigma_0\bigr).
$$

---

## 3. $p(x_t\mid x_0,\,x_1)$

Here $x_t$ is exactly $(1-t)x_0 + t\,x_1$ with no randomness, i.e.

$$
p(x_t\mid x_0,x_1)
= \mathcal{N}\!\bigl((1-t)x_0 + t\,x_1,\;0\bigr).
$$

---

## 4. $p(x_1\mid x_t)$

First form the joint Gaussian of $\,(x_t,x_1)$.  Its mean is $\bigl[\mu_t,\mu_1\bigr]$ and its covariance block‐matrix is

$$
\begin{pmatrix}
\Sigma_t & t\,\Sigma_1 \$$6pt]
t\,\Sigma_1 & \Sigma_1
\end{pmatrix}.
$$

By the standard conditional‐Gaussian formula, one gets

$$
x_1\mid x_t
\;\sim\;
\mathcal{N}\!\Bigl(
\mu_1 
\;+\; t\,\Sigma_1\,\Sigma_t^{-1}\,(x_t-\mu_t)
\;,\;
\Sigma_1 \;-\; t^2\,\Sigma_1\,\Sigma_t^{-1}\,\Sigma_1
\Bigr).
$$

---

## 5. $p(x_0\mid x_t)$

Similarly the joint of $(x_0,x_t)$ has mean $\bigl[\mu_0,\mu_t\bigr]$ and covariance

$$
\begin{pmatrix}
\Sigma_0 & (1-t)\,\Sigma_0 \$$6pt]
(1-t)\,\Sigma_0 & \Sigma_t
\end{pmatrix},
$$

so

$$
x_0\mid x_t
\;\sim\;
\mathcal{N}\!\Bigl(
\mu_0
\;+\;(1-t)\,\Sigma_0\,\Sigma_t^{-1}\,(x_t-\mu_t)
\;,\;
\Sigma_0 \;-\;(1-t)^2\,\Sigma_0\,\Sigma_t^{-1}\,\Sigma_0
\Bigr).
$$

---

### Compact Table of Results

| Conditional       | Mean                                                             | Covariance                                                           |
| ----------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| $x_t\mid x_0$     | $(1-t)x_0 + t\,\mu_1$                                            | $t^2\,\Sigma_1$                                                      |
| $x_t\mid x_1$     | $(1-t)\,\mu_0 + t\,x_1$                                          | $(1-t)^2\,\Sigma_0$                                                  |
| $x_t\mid x_0,x_1$ | $(1-t)x_0 + t\,x_1$                                              | $0$                                                                  |
| $x_1\mid x_t$     | $\displaystyle\mu_1 + t\,\Sigma_1\,\Sigma_t^{-1}(x_t-\mu_t)$     | $\displaystyle\Sigma_1 - t^2\,\Sigma_1\,\Sigma_t^{-1}\,\Sigma_1$     |
| $x_0\mid x_t$     | $\displaystyle\mu_0 + (1-t)\,\Sigma_0\,\Sigma_t^{-1}(x_t-\mu_t)$ | $\displaystyle\Sigma_0 - (1-t)^2\,\Sigma_0\,\Sigma_t^{-1}\,\Sigma_0$ |

All of these are genuine multivariate Gaussians, with the special “zero‐variance” case for the bridge $p(x_t\mid x_0,x_1)$.



### Appendix B: 驗證第二步到第三步的推導

給定的推導步驟如下：
$$
\begin{aligned}
\mathbf{u}_t(\mathbf{x}) & = \mathbb{E}_{\mathbf{x}_1 \sim p_{1 \mid t}}\left[\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1)\right] \\
& = \mathbb{E}_{\mathbf{x}_1 \sim p_{1 \mid t}}\left[\mathbf{x}_1 - \mathbf{x}_0\right] \\
& = \mathbb{E}_{\mathbf{x}_1 \sim p_{1 \mid t}}\left[\mathbf{x}_1\right] - \mathbb{E}_{\mathbf{x}_0 \sim p_{0 \mid t}}\left[\mathbf{x}_0\right] \quad \text{(這一步需要驗證)} \\
&= \cdots
\end{aligned}
$$
#### 關鍵問題：爲什麼 $\mathbb{E}_{\mathbf{x}_1 \sim p_{1 \mid t}}[\mathbf{x}_0] = \mathbb{E}_{\mathbf{x}_0 \sim p_{0 \mid t}}[\mathbf{x}_0]$？

### Rigorous Mathematical Proof

We need to prove that given the linear constraint $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ and the joint distribution $p(\mathbf{x}_0, \mathbf{x}_1)$, the following equality holds for any fixed $\mathbf{x}_t = \mathbf{x}$:

$$
\mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})} [\mathbf{x}_0] = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})} [\mathbf{x}_0]
$$

#### Step 1: Express $\mathbf{x}_0$ as a function of $\mathbf{x}_1$ and $\mathbf{x}_t$
From the linear interpolation constraint:
$$
\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1
$$
Solve for $\mathbf{x}_0$:
$$
\mathbf{x}_0 = \frac{\mathbf{x}_t - t\mathbf{x}_1}{1-t} \quad \text{(for $t < 1$)}
$$

#### Step 2: Left-hand side expectation
$$
\mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})} [\mathbf{x}_0] = \int \mathbf{x}_0  p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})  d\mathbf{x}_1
$$
Substitute $\mathbf{x}_0 = \frac{\mathbf{x} - t\mathbf{x}_1}{1-t}$:
$$
= \int \left( \frac{\mathbf{x} - t\mathbf{x}_1}{1-t} \right) p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})  d\mathbf{x}_1
$$

#### Step 3: Separate into two terms
$$
= \frac{1}{1-t} \left[ \mathbf{x} \int p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})  d\mathbf{x}_1 - t \int \mathbf{x}_1 p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})  d\mathbf{x}_1 \right]
$$
The first integral is 1 (probability density), and the second is $\mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]$:
$$
= \frac{1}{1-t} \left[ \mathbf{x} - t  \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}] \right]
$$

#### Step 4: Right-hand side expectation
$$
\mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})} [\mathbf{x}_0] = \int \mathbf{x}_0  p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})  d\mathbf{x}_0
$$
This is the definition of the conditional expectation:
$$
= \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]
$$

#### Step 5 (關鍵！): Connect both sides using the constraint
From the linear constraint $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$, take conditional expectation given $\mathbf{x}_t = \mathbf{x}$:
$$
\mathbb{E}[\mathbf{x}_t \mid \mathbf{x}_t = \mathbf{x}] = (1-t) \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] + t \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]
$$
Since $\mathbb{E}[\mathbf{x}_t \mid \mathbf{x}_t = \mathbf{x}] = \mathbf{x}$:
$$
\mathbf{x} = (1-t)  \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] + t  \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]
$$
Solve for $\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$:
$$
\mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}] = \frac{\mathbf{x} - t  \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]}{1-t}
$$

#### Step 6: Conclusion
From Steps 3 and 5:
$$
\mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})} [\mathbf{x}_0] = \frac{\mathbf{x} - t  \mathbb{E}[\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x}]}{1-t} = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]
$$
Thus:
$$
\boxed{\mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t)} [\mathbf{x}_0] = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t)} [\mathbf{x}_0]}
$$

### Key Insight
The proof relies on two fundamental properties:
1. **Deterministic relationship**: $\mathbf{x}_0$ is a linear function of $\mathbf{x}_1$ given $\mathbf{x}_t$
2. **Law of total expectation**: The constraint $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ forces the conditional expectations to satisfy the same linear relationship.

This holds for any joint distribution where the conditional expectations exist, not just Gaussians. The Gaussian assumption is only needed for the closed-form expressions in subsequent steps.


後續步驟使用高斯分佈的條件期望公式（線性插值下條件分佈仍是高斯）是正確的，但這一步的成立依賴於條件期望的基本性質，與分佈的具體形式無關。

### Appendix C: Proof for Generalized Linear Separable Case

Given:
- $\mathbf{x}_t = f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1)$ (linear separable form)
- We need to prove:
  $$
  \mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t)} [f_0(\mathbf{x}_0)] = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t)} [f_0(\mathbf{x}_0)]
  $$

---

#### Step 1: Express $f_0(\mathbf{x}_0)$ using the constraint
From $\mathbf{x}_t = f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1)$, we solve for $f_0(\mathbf{x}_0)$:
$$
f_0(\mathbf{x}_0) = \mathbf{x}_t - f_1(\mathbf{x}_1)
$$

#### Step 2: Left-hand side expectation
Conditioned on $\mathbf{x}_t = \mathbf{x}$:
$$
\mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t = \mathbf{x})} [f_0(\mathbf{x}_0)] = \mathbb{E}_{\mathbf{x}_1 \mid \mathbf{x}} [ \mathbf{x} - f_1(\mathbf{x}_1) ]
$$
By linearity of expectation:
$$
= \mathbf{x} - \mathbb{E}_{\mathbf{x}_1 \mid \mathbf{x}} [f_1(\mathbf{x}_1)]
$$

#### Step 3: Take conditional expectation of the constraint
Apply $\mathbb{E}[\cdot \mid \mathbf{x}_t = \mathbf{x}]$ to both sides of $\mathbf{x}_t = f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1)$:
$$
\mathbb{E}[\mathbf{x}_t \mid \mathbf{x}_t = \mathbf{x}] = \mathbb{E}[f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1) \mid \mathbf{x}_t = \mathbf{x}]
$$
Left side simplifies to $\mathbf{x}$ (since $\mathbf{x}_t$ is conditioned to $\mathbf{x}$). Right side expands by linearity:
$$
\mathbf{x} = \mathbb{E}[f_0(\mathbf{x}_0) \mid \mathbf{x}_t = \mathbf{x}] + \mathbb{E}[f_1(\mathbf{x}_1) \mid \mathbf{x}_t = \mathbf{x}]
$$

#### Step 4: Solve for $\mathbb{E}[f_0(\mathbf{x}_0) \mid \mathbf{x}_t = \mathbf{x}]$
Rearrange the equation:
$$
\mathbb{E}[f_0(\mathbf{x}_0) \mid \mathbf{x}_t = \mathbf{x}] = \mathbf{x} - \mathbb{E}[f_1(\mathbf{x}_1) \mid \mathbf{x}_t = \mathbf{x}]
$$

#### Step 5: Equate both expressions
From Step 2 and Step 4:
$$
\mathbb{E}_{\mathbf{x}_1 \mid \mathbf{x}} [f_0(\mathbf{x}_0)] = \mathbf{x} - \mathbb{E}_{\mathbf{x}_1 \mid \mathbf{x}} [f_1(\mathbf{x}_1)] = \mathbb{E}[f_0(\mathbf{x}_0) \mid \mathbf{x}_t = \mathbf{x}]
$$
The rightmost term is equivalent to:
$$
\mathbb{E}[f_0(\mathbf{x}_0) \mid \mathbf{x}_t = \mathbf{x}] = \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t)} [f_0(\mathbf{x}_0)]
$$

---

### Final Result
$$
\boxed{
\mathbb{E}_{\mathbf{x}_1 \sim p(\mathbf{x}_1 \mid \mathbf{x}_t)} [f_0(\mathbf{x}_0)]
= \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0 \mid \mathbf{x}_t)} [f_0(\mathbf{x}_0)]
}
$$

### Key Insights
1. **Linearity is Crucial**:  
   The proof hinges on the linear separability $\mathbf{x}_t = f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1)$, which allows:
   - Algebraic isolation of $f_0(\mathbf{x}_0)$
   - Linearity of expectation to split terms

2. **No Distributional Assumptions**:  
   The result holds for **any joint distribution** where:
   - The constraint $\mathbf{x}_t = f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1)$ exists
   - Conditional expectations are well-defined

3. **Geometric Intuition**:  
   The constraint creates a hyperplane $\mathbf{x} = f_0(\mathbf{x}_0) + f_1(\mathbf{x}_1)$. Conditioning on $\mathbf{x}_t = \mathbf{x}$ restricts $(\mathbf{x}_0, \mathbf{x}_1)$ to this hyperplane, forcing agreement of conditional expectations.


## Appendix D: Low Rank Gaussian Flow

### 回答

给定一个低秩高斯分布的过程，其中 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，且概率分布为 $p_t = \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$，其中：
- $\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$
- $\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1$

这里，$\boldsymbol{\Sigma}_1$ 的秩为 $r < d$（$d$ 是完整维度），$\boldsymbol{\Sigma}_0$ 是满秩正定矩阵。假设 $\mathbf{x}_0 \sim p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ 和 $\mathbf{x}_1 \sim p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ 是独立的。以下计算向量场 (vector field)、流 (flow) 和概率守恒 (conservation of probability)。

---

### 1. 向量场 (Vector Field)
向量场 $v_t(\mathbf{x})$ 定义为在时间 $t$ 和位置 $\mathbf{x}$ 处的速度场，它驱动概率分布的演化。具体地，$v_t(\mathbf{x}) = \mathbb{E}\left[\frac{d\mathbf{x}_t}{dt} \mid \mathbf{x}_t = \mathbf{x}\right]$。给定 $\frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0$，我们需要计算条件期望 $\mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]$。

通过高斯条件期望公式，得到：
$$
v_t(\mathbf{x}) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \left[ t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
其中：
- $\boldsymbol{\mu}_t = (1-t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$
- $\boldsymbol{\Sigma}_t = (1-t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1$

#### 说明：
- 对于 $t \in [0, 1)$，$\boldsymbol{\Sigma}_t$ 是正定可逆的（因为 $(1-t)^2 \boldsymbol{\Sigma}_0$ 正定，且 $t^2 \boldsymbol{\Sigma}_1$ 半正定），因此 $\boldsymbol{\Sigma}_t^{-1}$ 存在。
- 在 $t = 1$ 时，$\boldsymbol{\Sigma}_t = \boldsymbol{\Sigma}_1$ 可能奇异（秩 $r < d$），此时向量场可能未定义或需要特殊处理（如使用伪逆），但通常考虑 $t < 1$。
- 由于 $\boldsymbol{\Sigma}_1$ 是低秩（秩 $r$)，可以利用矩阵求逆引理（Sherman-Morrison-Woodbury）简化 $\boldsymbol{\Sigma}_t^{-1}$。设 $\boldsymbol{\Sigma}_1 = U \Lambda U^T$，其中 $U$ 是 $d \times r$ 矩阵（列正交），$\Lambda$ 是 $r \times r$ 正定对角矩阵。则：
  $$
  \boldsymbol{\Sigma}_t^{-1} = \frac{1}{(1-t)^2} \left( \boldsymbol{\Sigma}_0^{-1} - \boldsymbol{\Sigma}_0^{-1} U \left[ \Lambda^{-1} + \frac{t^2}{(1-t)^2} (U^T \boldsymbol{\Sigma}_0^{-1} U) \right]^{-1} U^T \boldsymbol{\Sigma}_0^{-1} \right)
  $$
  代入 $v_t(\mathbf{x})$ 可进一步简化计算，但表达式较复杂。

---

### 2. 流 (Flow)
流定义为向量场 $v_t(\mathbf{x})$ 的积分，即求解常微分方程 (ODE) 以获得从初始点 $\mathbf{x}_0$ 到时间 $t$ 的轨迹 $\mathbf{x}_t$：
$$
\frac{d\mathbf{x}_t}{dt} = v_t(\mathbf{x}_t), \quad \mathbf{x}_t|_{t=0} = \mathbf{x}_0
$$
代入向量场后：
$$
\frac{d\mathbf{x}_t}{dt} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \left[ t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{-1} (\mathbf{x}_t - \boldsymbol{\mu}_t)
$$

#### 性质与解法：
- **线性时变系统**：向量场是 $\mathbf{x}_t$ 的线性函数，因此 ODE 是线性时变的。令 $\mathbf{z}_t = \mathbf{x}_t - \boldsymbol{\mu}_t$（中心化坐标），则 ODE 简化为：
  $$
  \frac{d\mathbf{z}_t}{dt} = A(t) \mathbf{z}_t, \quad \mathbf{z}_t|_{t=0} = \mathbf{x}_0 - \boldsymbol{\mu}_0
  $$
  其中 $A(t) = \left[ t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{-1}$.
- **解的形式**：流映射 $\phi_t: \mathbb{R}^d \to \mathbb{R}^d$ 将 $\mathbf{x}_0$ 映射到 $\mathbf{x}_t$：
  $$
  \mathbf{x}_t = \phi_t(\mathbf{x}_0) = \boldsymbol{\mu}_t + \Phi(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0)
  $$
  这里，$\Phi(t)$ 是状态转移矩阵，满足：
  $$
  \frac{d\Phi(t)}{dt} = A(t) \Phi(t), \quad \Phi(0) = I
  $$
- **解析解**：一般情况下，$\Phi(t)$ 无闭式解，需数值积分（如使用 ODE 求解器）。在特殊情况下（如标量或 $\boldsymbol{\Sigma}_0, \boldsymbol{\Sigma}_1$ 成比例），可简化。
- **概率演化**：当初始点 $\mathbf{x}_0 \sim p_0$ 时，流映射确保 $\mathbf{x}_t \sim p_t$，即边缘分布正确演化。

---

### 3. 概率守恒 (Conservation of Probability)
概率守恒指概率质量在演化过程中保持不变，由连续性方程 (continuity equation) 描述：
$$
\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla \cdot (p_t(\mathbf{x}) v_t(\mathbf{x})) = 0
$$
其中 $p_t(\mathbf{x})$ 是概率密度函数，$\nabla \cdot$ 是散度算子。

#### 验证：
- 给定 $p_t = \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$ 和向量场 $v_t(\mathbf{x})$，概率守恒成立，因为：
  - 向量场 $v_t(\mathbf{x})$ 是通过条件期望构造的，确保它生成指定的边缘分布 $p_t$。
  - 高斯分布的线性演化满足连续性方程。具体验证可通过计算：
    $$
    \frac{\partial p_t}{\partial t} = \text{（高斯密度的导数）}, \quad \nabla \cdot (p_t v_t) = p_t \nabla \cdot v_t + v_t \cdot \nabla p_t
    $$
    并证明其和为零。由于表达式复杂，这里省略详细计算，但结论成立。
- **关键点**：无扩散项（纯确定性流），因此概率守恒由向量场保证。低秩 $\boldsymbol{\Sigma}_1$ 不影响守恒性（对 $t < 1$）。

---

### 总结
- **向量场**：$v_t(\mathbf{x}) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \left[ t \boldsymbol{\Sigma}_1 - (1-t) \boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{-1} (\mathbf{x} - \boldsymbol{\mu}_t)$
- **流**：解 ODE $\frac{d\mathbf{x}_t}{dt} = v_t(\mathbf{x}_t)$ 初值 $\mathbf{x}_0$，流映射为 $\mathbf{x}_t = \boldsymbol{\mu}_t + \Phi(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0)$，其中 $\Phi(t)$ 满足 $\frac{d\Phi}{dt} = A(t) \Phi$，$\Phi(0) = I$。
- **概率守恒**：连续性方程成立，概率质量守恒。

此模型适用于流匹配 (flow matching) 或概率路径分析，其中低秩 $\boldsymbol{\Sigma}_1$ 在 $t \to 1$ 时需注意奇异点处理。


### 簡化版

给定假设 $\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{I}$（各向同性满秩协方差）和 $\boldsymbol{\Sigma}_1 = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$（秩 $r$ 的低秩矩阵），其中 $\mathbf{U}$ 是 $d \times r$ 列正交矩阵（$\mathbf{U}^T\mathbf{U} = \mathbf{I}_r$），$\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_r)$ 是正定对角矩阵。以下简化向量场 $v_t(\mathbf{x})$ 和流映射 $\mathbf{x}_t$ 的表达式。

---

简化后的协方差矩阵
$$
\boldsymbol{\Sigma}_t = (1-t)^2 \sigma_0^2 \mathbf{I} + t^2 \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T
$$

---

### 1. 向量场 (Vector Field) 的简化
利用子空间分解（$\text{Col}(\mathbf{U})$ 和 $\text{Col}(\mathbf{U})^\perp$），向量场简化为：
$$
v_t(\mathbf{x}) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \left[ \mathbf{U} \mathbf{A}_t \mathbf{U}^T - \frac{1}{1-t} (\mathbf{I} - \mathbf{U} \mathbf{U}^T) \right] (\mathbf{x} - \boldsymbol{\mu}_t)
$$
其中：
- $\mathbf{A}_t = \text{diag}\left( a_i(t) \right)_{i=1}^r$ 是 $r \times r$ 对角矩阵
- $a_i(t) = \dfrac{t \lambda_i - (1-t) \sigma_0^2}{(1-t)^2 \sigma_0^2 + t^2 \lambda_i}$

#### 几何解释：
- 在 $\text{Col}(\mathbf{U})$ 子空间：线性变换 $\mathbf{U} \mathbf{A}_t \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}_t)$
- 在正交补空间 $\text{Col}(\mathbf{U})^\perp$：收缩项 $-\dfrac{1}{1-t} (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x} - \boldsymbol{\mu}_t)$
- 全局平移：$\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$

---

### 2. 流映射 (Flow Map) 的显式解析解
流映射有闭式解：
$$
\mathbf{x}_t = \boldsymbol{\mu}_t + \left[ \mathbf{U} \mathbf{D}_t \mathbf{U}^T + (1-t) (\mathbf{I} - \mathbf{U} \mathbf{U}^T) \right] (\mathbf{x}_0 - \boldsymbol{\mu}_0)
$$
其中：
- $\mathbf{D}_t = \text{diag}\left( d_i(t) \right)_{i=1}^r$ 是 $r \times r$ 对角矩阵
- $d_i(t) = \dfrac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_i } }{ \sigma_0 }$

#### 推导说明：
1. **子空间分解**：令 $\mathbf{z}_t = \mathbf{x}_t - \boldsymbol{\mu}_t$
2. **在 $\text{Col}(\mathbf{U})$ 上**：
   $$
   \frac{d}{dt}(\mathbf{U}^T \mathbf{z}_t) = \mathbf{A}_t (\mathbf{U}^T \mathbf{z}_t) \implies \mathbf{U}^T \mathbf{z}_t = \mathbf{D}_t \mathbf{U}^T \mathbf{z}_0
   $$
3. **在 $\text{Col}(\mathbf{U})^\perp$ 上**：
   $$
   \frac{d}{dt} \left[ (\mathbf{I} - \mathbf{U} \mathbf{U}^T) \mathbf{z}_t \right] = -\frac{1}{1-t} (\mathbf{I} - \mathbf{U} \mathbf{U}^T) \mathbf{z}_t
   $$
   解得：$(\mathbf{I} - \mathbf{U} \mathbf{U}^T) \mathbf{z}_t = (1-t) (\mathbf{I} - \mathbf{U} \mathbf{U}^T) \mathbf{z}_0$
4. **合并结果**：
   $$
   \mathbf{z}_t = \mathbf{U} \mathbf{D}_t \mathbf{U}^T \mathbf{z}_0 + (1-t) (\mathbf{I} - \mathbf{U} \mathbf{U}^T) \mathbf{z}_0
   $$

---

### 3. 概率守恒验证
- **边缘分布一致性**：若 $\mathbf{x}_0 \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$，则 $\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$
- **支撑集不变性**：
  $$
  \text{Supp}(p_t) = \boldsymbol{\mu}_t + \text{Col}(\mathbf{U}), \quad \forall t \in [0,1]
  $$
- **连续性方程**：向量场构造保证 $\dfrac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0$

---

### 总结：简化后的表达式
| 组件        | 简化表达式                                                                                                                                                                                                                                                                                                                                                    |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **协方差矩阵** | $\boldsymbol{\Sigma}_t = (1-t)^2 \sigma_0^2 \mathbf{I} + t^2 \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$                                                                                                                                                                                                                                                   |
| **向量场**   | $v_t(\mathbf{x}) = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0 + \mathbf{U} \mathbf{A}_t \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}_t) - \dfrac{1}{1-t} (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x} - \boldsymbol{\mu}_t)$ <br> $\mathbf{A}_t = \text{diag}\left( \dfrac{t \lambda_i - (1-t) \sigma_0^2}{(1-t)^2 \sigma_0^2 + t^2 \lambda_i} \right)$ |
| **流映射**   | $\mathbf{x}_t = \boldsymbol{\mu}_t + \mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0) + (1-t) (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0)$ <br> $\mathbf{D}_t = \text{diag}\left( \dfrac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_i } }{ \sigma_0 } \right)$                                       |

### 关键特性：
1. **显式解析解**：流映射有闭式表达式，无需数值积分
2. **子空间行为**：
   - $\text{Col}(\mathbf{U})$ 上：缩放 $\mathbf{D}_t$（依赖 $\lambda_i$）
   - $\text{Col}(\mathbf{U})^\perp$ 上：线性收缩 $(1-t)$
1. **边界一致性**：
   - $t=0$：$\mathbf{D}_0 = \mathbf{I}_r$，流映射为恒等变换
   - $t=1$：$\mathbf{D}_1 = \frac{1}{\sigma_0} \sqrt{\mathbf{\Lambda}}$，$\boldsymbol{\Sigma}_t = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$
1. **计算效率**：仅需特征值分解 $\boldsymbol{\Sigma}_1$，避免高维矩阵求逆


### Appendix E: Low-Rank to Low-Rank Gaussian Flow

考虑两个低秩高斯分布之间的插值过程：
- $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$
- 边缘分布 $p_t = \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$  
其中：
  - $\boldsymbol{\mu}_t = (1 - t) \boldsymbol{\mu}_0 + t \boldsymbol{\mu}_1$
  - $\boldsymbol{\Sigma}_t = (1 - t)^2 \boldsymbol{\Sigma}_0 + t^2 \boldsymbol{\Sigma}_1$
- $\boldsymbol{\Sigma}_0$ 和 $\boldsymbol{\Sigma}_1$ 均为秩 $r < d$ 的对称半正定矩阵（$d$ 为全维度）。

假设 $\mathbf{x}_0 \sim p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ 和 $\mathbf{x}_1 \sim p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ 独立。以下计算向量场、流映射和概率守恒。

---

### 1. 向量场 (Vector Field)
向量场 $v_t(\mathbf{x})$ 定义为：  
$$
v_t(\mathbf{x}) = \mathbb{E}\left[\frac{d\mathbf{x}_t}{dt} \mid \mathbf{x}_t = \mathbf{x}\right] = \mathbb{E}[\mathbf{x}_1 - \mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x}]
$$

通过高斯条件期望公式（使用伪逆处理低秩协方差矩阵）：  
$$
v_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left[ t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{+} (\mathbf{x} - \boldsymbol{\mu}_t)
$$  
其中 $\boldsymbol{\Sigma}_t^{+}$ 是 $\boldsymbol{\Sigma}_t$ 的 Moore-Penrose 伪逆。

#### 关键推导：
- 联合分布 $\begin{bmatrix} \mathbf{x}_0 \\ \mathbf{x}_1 \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_0 \\ \boldsymbol{\mu}_1 \end{bmatrix}, \begin{bmatrix} \boldsymbol{\Sigma}_0 & 0 \\ 0 & \boldsymbol{\Sigma}_1 \end{bmatrix} \right)$
- $\mathbf{x}_t = \begin{bmatrix} (1-t)I & tI \end{bmatrix} \begin{bmatrix} \mathbf{x}_0 \\ \mathbf{x}_1 \end{bmatrix}$
- 协方差计算：  
  $$
  \text{Cov}(\mathbf{x}_1 - \mathbf{x}_0, \mathbf{x}_t) = t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0, \quad \text{Cov}(\mathbf{x}_t) = \boldsymbol{\Sigma}_t
  $$
- 伪逆 $\boldsymbol{\Sigma}_t^{+}$ 处理秩退化（当 $\text{rank}(\boldsymbol{\Sigma}_t) < d$）。

---

### 2. 流映射 (Flow Map)
流映射通过求解 ODE 获得：  
$$
\frac{d\mathbf{x}_t}{dt} = v_t(\mathbf{x}_t), \quad \mathbf{x}_t|_{t=0} = \mathbf{x}_0
$$  
代入向量场：  
$$
\frac{d\mathbf{x}_t}{dt} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left[ t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{+} (\mathbf{x}_t - \boldsymbol{\mu}_t)
$$

#### 解析形式：
令 $\mathbf{z}_t = \mathbf{x}_t - \boldsymbol{\mu}_t$（中心化坐标），则：  
$$
\frac{d\mathbf{z}_t}{dt} = C(t) \mathbf{z}_t, \quad C(t) = \left[ t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{+}
$$  
初始条件 $\mathbf{z}_0 = \mathbf{x}_0 - \boldsymbol{\mu}_0$。解为：  
$$
\mathbf{z}_t = \Phi(t) \mathbf{z}_0
$$  
其中 $\Phi(t)$ 是状态转移矩阵，满足：  
$$
\frac{d\Phi(t)}{dt} = C(t) \Phi(t), \quad \Phi(0) = I
$$  
最终流映射：  
$$
\mathbf{x}_t = \boldsymbol{\mu}_t + \Phi(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0)
$$

#### 特性：
- 线性变换保持高斯性：若 $\mathbf{x}_0 \sim p_0$，则 $\mathbf{x}_t \sim p_t$。
- $\Phi(t)$ 无闭式解，需数值积分（如 ODE 求解器）。

---

### 3. 概率守恒 (Conservation of Probability)
概率守恒由连续性方程描述：  
$$
\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla \cdot \left( p_t(\mathbf{x}) v_t(\mathbf{x}) \right) = 0
$$

#### 验证：
1. **边缘分布一致性**：  
   由构造，流映射将 $p_0$ 映射到 $p_t$，因此概率质量守恒：  
   $$
   \int p_t(\mathbf{x})  d\mathbf{x} = 1, \quad \forall t \in [0,1]
   $$
   
2. **连续性方程成立**：  
   - 向量场 $v_t(\mathbf{x})$ 通过条件期望定义，确保生成指定边缘分布 $p_t$。
   - 退化高斯分布下（$\text{rank}(\boldsymbol{\Sigma}_t) < d$），方程在分布支撑集上成立：
     $$
     \text{Supp}(p_t) = \boldsymbol{\mu}_t + \text{Col}(\boldsymbol{\Sigma}_t)
     $$
     其中 $\text{Col}(\cdot)$ 表示列空间。

---

### 总结
| 组件       | 表达式                                                                                                                                                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **向量场**  | $v_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \left[ t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{+} (\mathbf{x} - \boldsymbol{\mu}_t)$                                         |
| **流映射**  | $\mathbf{x}_t = \boldsymbol{\mu}_t + \Phi(t) (\mathbf{x}_0 - \boldsymbol{\mu}_0)$ <br> $\frac{d\Phi(t)}{dt} = \left[ t\boldsymbol{\Sigma}_1 - (1-t)\boldsymbol{\Sigma}_0 \right] \boldsymbol{\Sigma}_t^{+} \Phi(t), \quad \Phi(0) = I$ |
| **概率守恒** | 连续性方程成立，概率质量守恒                                                                                                                                                                                                                         |

### 关键说明：
1. **伪逆处理低秩**：  
   - $\boldsymbol{\Sigma}_t^{+}$ 在 $\boldsymbol{\Sigma}_t$ 秩退化时至关重要（如 $t=0,1$ 或 $\text{Col}(\boldsymbol{\Sigma}_0) \cap \text{Col}(\boldsymbol{\Sigma}_1) \neq \{0\}$)。
   - 伪逆计算：若 $\boldsymbol{\Sigma}_t = U_t D_t U_t^T$（特征值分解），则 $\boldsymbol{\Sigma}_t^{+} = U_t D_t^{+} U_t^T$，其中 $D_t^{+}$ 将非零对角元取倒数。

2. **数值稳定性**：  
   - 当 $t \to 0^+$ 或 $t \to 1^-$ 时，$\boldsymbol{\Sigma}_t$ 可能接近奇异，需正则化或子空间投影。

此模型为**确定性流**（无扩散项），适用于流匹配（Flow Matching）和生成模型中的概率路径构建。

### 簡化版

給定假設 $\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{I}_r$（秩 $r$ 的協方差矩陣），其中 $\mathbf{I}_r$ 表示 $d \times d$ 的對角矩陣，前 $r$ 個對角元素為 $1$，其餘為 $0$。為簡化推導，我們假設 $\boldsymbol{\Sigma}_1$ 與 $\boldsymbol{\Sigma}_0$ 有相同的列空間（即存在 $d \times r$ 列正交矩陣 $\mathbf{U}$ 滿足 $\mathbf{U}^T\mathbf{U} = \mathbf{I}_r$，使得 $\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{U}\mathbf{U}^T$ 且 $\boldsymbol{\Sigma}_1 = \mathbf{U} \mathbf{\Lambda}_1 \mathbf{U}^T$，其中 $\mathbf{\Lambda}_1$ 是 $r \times r$ 正定對角矩陣 $\mathbf{\Lambda}_1 = \text{diag}(\lambda_1^{(1)}, \dots, \lambda_1^{(r)})$）。則推導簡化如下：

---

### 簡化後的協方差矩陣
- $\boldsymbol{\Sigma}_t = (1-t)^2 \sigma_0^2 \mathbf{U}\mathbf{U}^T + t^2 \mathbf{U} \mathbf{\Lambda}_1 \mathbf{U}^T = \mathbf{U} \mathbf{S}_t \mathbf{U}^T$
- 其中 $\mathbf{S}_t = (1-t)^2 \sigma_0^2 \mathbf{I}_r + t^2 \mathbf{\Lambda}_1$ 是 $r \times r$ 對角矩陣，元素為：
  $$
  s_i(t) = (1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)}, \quad i=1,\dots,r
  $$

---

### 1. 向量場 (Vector Field)
向量場表達式簡化為：
$$
v_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \mathbf{U} \mathbf{A}_t \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}_t)
$$
其中：
- $\mathbf{A}_t = \left[ t \mathbf{\Lambda}_1 - (1-t) \sigma_0^2 \mathbf{I}_r \right] \mathbf{S}_t^{-1}$ 是 $r \times r$ **對角矩陣**
- $\mathbf{S}_t^{-1}$ 是對角矩陣，元素為 $1/s_i(t)$
- $\mathbf{A}_t$ 的對角元素為：
  $$
  a_i(t) = \frac{t \lambda_1^{(i)} - (1-t) \sigma_0^2}{(1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)}}
  $$

#### 性質：
- 向量場僅在 $\mathbf{U}$ 張成的 $r$ 維子空間上變化，正交補空間上為常數 $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$。
- $\mathbf{U}^T v_t(\mathbf{x}) = \mathbf{U}^T(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \mathbf{A}_t \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}_t)$（$r$ 維表示）。

---

### 2. 流映射 (Flow Map)
流映射有 **解析解**：
$$
\mathbf{x}_t = \boldsymbol{\mu}_t + \mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)
$$
其中 $\mathbf{D}_t = \frac{1}{\sigma_0} \sqrt{\mathbf{S}_t}$ 是 $r \times r$ **對角矩陣**，元素為：
$$
d_i(t) = \frac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)} } }{ \sigma_0 }
$$

#### 推導：
1. 令 $\tilde{\mathbf{z}}_t = \mathbf{U}^T (\mathbf{x}_t - \boldsymbol{\mu}_t)$（$r$ 維中心化坐標）。
2. 解 ODE：
   $$
   \frac{d\tilde{\mathbf{z}}_t}{dt} = \mathbf{A}_t \tilde{\mathbf{z}}_t, \quad \tilde{\mathbf{z}}_0 = \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)
   $$
3. 對角元素 $a_i(t) = \frac{1}{2} \frac{d}{dt} \ln s_i(t)$，積分得：
   $$
   \int_0^t a_i(s)  ds = \frac{1}{2} \ln \frac{s_i(t)}{s_i(0)} = \ln \sqrt{ \frac{s_i(t)}{\sigma_0^2} }
   $$
4. 解為 $\tilde{z}_t^{(i)} = \exp\left( \int_0^t a_i(s)  ds \right) \tilde{z}_0^{(i)} = \sqrt{ \frac{s_i(t)}{\sigma_0^2} }  \tilde{z}_0^{(i)}$。
5. 還原到 $d$ 維空間即得流映射。

---

### 3. 概率守恒 (Conservation of Probability)
概率守恒自動滿足：
1. **邊緣分布一致性**：若 $\mathbf{x}_0 \sim p_0$，則 $\mathbf{x}_t \sim \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$。
2. **支撐集不變**：$\text{Supp}(p_t) = \boldsymbol{\mu}_t + \text{Col}(\mathbf{U})$。
3. **連續性方程**：向量場 $v_t(\mathbf{x})$ 的構造保證 $\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0$ 在子空間上成立。

---

### 總結
| 組件 | 簡化後的表達式 |
|------|----------------|
| **協方差矩陣** | $\boldsymbol{\Sigma}_t = \mathbf{U} \mathbf{S}_t \mathbf{U}^T$ <br> $\mathbf{S}_t = \text{diag}\left( (1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)} \right)$ |
| **向量場** | $v_t(\mathbf{x}) = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \mathbf{U} \mathbf{A}_t \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}_t)$ <br> $\mathbf{A}_t = \text{diag}\left( \frac{t \lambda_1^{(i)} - (1-t) \sigma_0^2}{(1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)}} \right)$ |
| **流映射** | $\mathbf{x}_t = \boldsymbol{\mu}_t + \mathbf{U} \mathbf{D}_t \mathbf{U}^T (\mathbf{x}_0 - \boldsymbol{\mu}_0)$ <br> $\mathbf{D}_t = \text{diag}\left( \frac{ \sqrt{ (1-t)^2 \sigma_0^2 + t^2 \lambda_1^{(i)} } }{ \sigma_0 } \right)$ |
| **概率守恒** | 成立，且流映射是解析的 |

### 關鍵說明：
1. **子空間約束**：整個演化限制在 $\text{Col}(\mathbf{U})$ 上，正交補空間為常數平移 $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$。
2. **數值穩定性**：當 $t \to 0^+$ 或 $t \to 1^-$ 時：
   - $t=0$：$\mathbf{D}_t = \mathbf{I}_r$，流映射為恆等變換。
   - $t=1$：$\mathbf{D}_t = \frac{1}{\sigma_0} \sqrt{\mathbf{\Lambda}_1}$，需確保 $\lambda_1^{(i)} > 0$。
1. **適用性**：此簡化適用於 $\boldsymbol{\Sigma}_0$ 為各向同性（$\sigma_0^2 \mathbf{I}_r$）且 $\boldsymbol{\Sigma}_1$ 與其同子空間的情形。若子空間不同，需額外處理交空間投影。


## Appendix F

Great — you're looking at the geodesic interpolation expression in Riemannian geometry:

$$
x_t = \exp_{x_1}\left( \kappa(t) \log_{x_1}(x_0) \right)
\quad \text{for} \quad t \in [0,1]
$$

This is the general form of interpolation along the geodesic from $x_0$ to $x_1$ using a Riemannian manifold. Let’s **simplify this in Euclidean space**, where things become much more straightforward.

---

### In Euclidean Space:

* $\log_{x_1}(x_0) = x_0 - x_1$
  (The "logarithmic map" just gives the vector pointing from $x_1$ to $x_0$)
* $\exp_{x_1}(v) = x_1 + v$
  (The "exponential map" just adds the tangent vector to the point)

So plugging into the original expression:

$$
x_t = \exp_{x_1}\left( \kappa(t) \cdot \log_{x_1}(x_0) \right)
= x_1 + \kappa(t)(x_0 - x_1)
$$

Rewriting:

$$
x_t = (1 - \kappa(t))x_1 + \kappa(t) x_0
$$

or equivalently:

$$
x_t = (1 - \kappa(t))x_1 + \kappa(t)x_0
$$

---

### ✅ Summary

In **Euclidean space**, the expression simplifies to a **linear interpolation** between $x_1$ and $x_0$, controlled by $\kappa(t)$:

$$
x_t = (1 - \kappa(t)) x_1 + \kappa(t) x_0
$$

This gives you a straight-line path from $x_1$ to $x_0$ as $t$ goes from 0 to 1 (note: this direction is reversed from the usual $x_0 \to x_1$, because it's anchored at $x_1$).

Let me know if you want to [visualize this interpolation](f) or [see how it generalizes to manifolds like spheres or hyperbolic space](f).


## Appendix G: 正交补空间 (Orthogonal Complement)

设 $\mathbf{U}$ 是一个 $d \times r$ 矩阵（$r < d$)，其列向量 $\{\mathbf{u}_1, \dots, \mathbf{u}_r\}$ 是标准正交基（即 $\mathbf{U}^T\mathbf{U} = \mathbf{I}_r$）。  
- **子空间**：$\text{Col}(\mathbf{U}) = \{\mathbf{U}\mathbf{y} \mid \mathbf{y} \in \mathbb{R}^r\}$（由 $\mathbf{U}$ 的列张成的 $r$ 维子空间）。  
- **正交补空间**：  
  $$
  \text{Col}(\mathbf{U})^\perp = \left\{ \mathbf{v} \in \mathbb{R}^d \mid \mathbf{v} \perp \mathbf{u}_i,\  \forall i=1,\dots,r \right\}
  $$
  即所有与 $\text{Col}(\mathbf{U})$ 正交的向量构成的 $(d-r)$ 维子空间。

---

### 证明：$\mathbf{I} - \mathbf{U}\mathbf{U}^T$ 是向正交补空间的投影矩阵

#### 步骤 1: 验证幂等性（Idempotence）  
投影矩阵需满足 $\mathbf{P}^2 = \mathbf{P}$:  
$$
(\mathbf{I} - \mathbf{U}\mathbf{U}^T)^2 = \mathbf{I} - 2\mathbf{U}\mathbf{U}^T + \mathbf{U}\mathbf{U}^T\mathbf{U}\mathbf{U}^T = \mathbf{I} - 2\mathbf{U}\mathbf{U}^T + \mathbf{U} \underbrace{(\mathbf{U}^T\mathbf{U})}_{\mathbf{I}_r} \mathbf{U}^T = \mathbf{I} - \mathbf{U}\mathbf{U}^T.
$$

#### 步骤 2: 验证对称性（Symmetry）  
投影矩阵需满足 $\mathbf{P}^T = \mathbf{P}$:  
$$
(\mathbf{I} - \mathbf{U}\mathbf{U}^T)^T = \mathbf{I}^T - (\mathbf{U}\mathbf{U}^T)^T = \mathbf{I} - \mathbf{U}\mathbf{U}^T.
$$

#### 步骤 3: 验证值域（Range）  
对任意 $\mathbf{x} \in \mathbb{R}^d$，证明 $(\mathbf{I} - \mathbf{U}\mathbf{U}^T)\mathbf{x} \in \text{Col}(\mathbf{U})^\perp$:  
$$
\mathbf{U}^T \left[ (\mathbf{I} - \mathbf{U}\mathbf{U}^T)\mathbf{x} \right] = \mathbf{U}^T\mathbf{x} - \mathbf{U}^T\mathbf{U}\mathbf{U}^T\mathbf{x} = \mathbf{U}^T\mathbf{x} - \mathbf{U}^T\mathbf{x} = \mathbf{0}.
$$  
因此 $(\mathbf{I} - \mathbf{U}\mathbf{U}^T)\mathbf{x}$ 与 $\text{Col}(\mathbf{U})$ 的所有基向量正交。

#### 步骤 4: 验证零空间（Nullspace）  
若 $\mathbf{x} \in \text{Col}(\mathbf{U})$，则 $\mathbf{x} = \mathbf{U}\mathbf{y}$ 对某个 $\mathbf{y} \in \mathbb{R}^r$:  
$$
(\mathbf{I} - \mathbf{U}\mathbf{U}^T)\mathbf{x} = \mathbf{U}\mathbf{y} - \mathbf{U}\mathbf{U}^T\mathbf{U}\mathbf{y} = \mathbf{U}\mathbf{y} - \mathbf{U}\mathbf{y} = \mathbf{0}.
$$  
即 $\text{Col}(\mathbf{U})$ 是零空间。

---

### 几何解释
1. **投影分解**：  
   任意向量 $\mathbf{x} \in \mathbb{R}^d$ 可唯一分解为：  
   $$
   \mathbf{x} = \underbrace{\mathbf{U}\mathbf{U}^T\mathbf{x}}_{\in \text{Col}(\mathbf{U})} + \underbrace{(\mathbf{I} - \mathbf{U}\mathbf{U}^T)\mathbf{x}}_{\in \text{Col}(\mathbf{U})^\perp}.
   $$
   - $\mathbf{U}\mathbf{U}^T\mathbf{x}$：$\mathbf{x}$ 在 $\text{Col}(\mathbf{U})$ 上的投影。  
   - $(\mathbf{I} - \mathbf{U}\mathbf{U}^T)\mathbf{x}$：$\mathbf{x}$ 在 $\text{Col}(\mathbf{U})^\perp$ 上的投影。

2. **收缩机制**：  
   在流映射中，正交补空间分量被乘以 $(1-t)$：  
   $$
   (\mathbf{I} - \mathbf{U}\mathbf{U}^T)(\mathbf{x}_t - \boldsymbol{\mu}_t) = (1-t) (\mathbf{I} - \mathbf{U}\mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0).
   $$  
   - 当 $t \to 1^-$ 时，$(1-t) \to 0$，因此该分量收缩至零。  
   - 在 $t=1$ 时，所有点坍缩到子空间 $\boldsymbol{\mu}_1 + \text{Col}(\mathbf{U})$。

---

### 示例：$d=3$, $r=1$（目标为一条直线）
- 设 $\mathbf{U} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$，则 $\text{Col}(\mathbf{U})$ 是 $x$-轴。  
- 投影矩阵：  
  $$
  \mathbf{U}\mathbf{U}^T = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}, \quad \mathbf{I} - \mathbf{U}\mathbf{U}^T = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}.
  $$  
- 对任意点 $\mathbf{x} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$:  
  $$
  (\mathbf{I} - \mathbf{U}\mathbf{U}^T)\mathbf{x} = \begin{bmatrix} 0 \\ y \\ z \end{bmatrix} \quad \text{（$yz$-平面上的分量）}.
  $$  
- 在流映射中：  
  $$
  \begin{bmatrix} 0 \\ y_t \\ z_t \end{bmatrix} = (1-t) \begin{bmatrix} 0 \\ y_0 \\ z_0 \end{bmatrix} \implies \text{当 } t=1 \text{ 时，} y_1=z_1=0.
  $$  
  所有点收缩到 $x$-轴上。

---

### 结论
- **$\mathbf{I} - \mathbf{U}\mathbf{U}^T$** 是向正交补空间 $\text{Col}(\mathbf{U})^\perp$ 的投影矩阵。  
- **收缩项**：在流映射中乘以 $(1-t)$，确保当 $t \to 1$ 时，正交补空间分量收缩至零。  
- **物理意义**：实现概率分布从高维空间到低秩子空间的坍缩，是支撑集拓扑变化的数学表达。

### 为什么称为“收缩项”？
1. **收缩行为**：
   - 该项在正交补空间 $\text{Col}(\mathbf{U})^\perp$ 上引入了一个**随时间收缩至零**的分量。
   - 系数 $(1-t)$ 是 $t$ 的线性函数：当 $t \to 1^-$ 时，$(1-t) \to 0$，导致该分量衰减到零。

2. **几何解释**：
   - 初始时刻 ($t=0$)：  
     $$
     (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0) = \text{正交补空间上的初始偏移}
     $$
   - 演化过程 ($0 < t < 1$)：  
     $$
     \| (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_t - \boldsymbol{\mu}_t) \| = (1-t) \cdot \| (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_0 - \boldsymbol{\mu}_0) \|
     $$
     模长按比例 $(1-t)$ 线性收缩。
   - 终点时刻 ($t=1$)：  
     $$
     (\mathbf{I} - \mathbf{U} \mathbf{U}^T)(\mathbf{x}_1 - \boldsymbol{\mu}_1) = 0
     $$
     **正交补空间分量被完全压缩到零**。

3. **物理意义**：
   - 该项将数据从高维空间 ($d$ 维) 收缩到低秩子空间 $\text{Col}(\mathbf{U})$ ($r$ 维)。
   - 在 $t=1$ 时，所有概率质量被压缩到子空间 $\boldsymbol{\mu}_1 + \text{Col}(\mathbf{U})$ 上（即 $\boldsymbol{\Sigma}_1$ 的支撑集）。

---

### 收缩项的动力学来源
在向量场中，收缩项的形式为：
$$
\underbrace{-\dfrac{1}{1-t}}_{\text{收缩强度}} \cdot \underbrace{(\mathbf{I} - \mathbf{U} \mathbf{U}^T)}_{\text{正交补投影}} (\mathbf{x} - \boldsymbol{\mu}_t)
$$
- **负号**表示运动方向指向原点（相对 $\boldsymbol{\mu}_t$）。
- **系数 $-\frac{1}{1-t}$** 在 $t \to 1^-$ 时发散到 $-\infty$，表示收缩速度无限增大。
- **投影算子 $(\mathbf{I} - \mathbf{U} \mathbf{U}^T)$** 确保只影响与子空间 $\text{Col}(\mathbf{U})$ 正交的分量。

---

### 为什么需要收缩项？
1. **支撑集演化**：
   - 初始分布 $p_0 = \mathcal{N}(\boldsymbol{\mu}_0, \sigma_0^2 \mathbf{I})$ 的支撑集是 **整个 $\mathbb{R}^d$**（满秩）。
   - 最终分布 $p_1 = \mathcal{N}(\boldsymbol{\mu}_1, \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T)$ 的支撑集是 **$r$ 维子空间** $\boldsymbol{\mu}_1 + \text{Col}(\mathbf{U})$。
   - 收缩项迫使正交补空间的分量衰减到零，实现支撑集从 $\mathbb{R}^d$ 到 $r$ 维子空间的拓扑变化。

2. **概率守恒的必然要求**：
   - 若没有收缩项，流映射无法将高维高斯分布压缩到低秩子空间。
   - 收缩项确保在 $t=1$ 时，分布精确坍缩到目标子空间，同时保持边缘分布 $p_t$ 的一致性。

