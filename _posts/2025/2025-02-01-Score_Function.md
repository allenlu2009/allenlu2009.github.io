---
title: Math Stat I - Likelihood, Score Function, and Fisher Information
date: 2025-02-01 23:10:08
typora-root-url: ../../allenlu2009.github.io
categories:
  - GenAI
tags:
  - Diffusion
  - Fisher
  - GenAI
  - Information
  - Math-AI
  - Math_AI
  - Score
---




<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>

## Introduction

Score function 聽起來很普通，但是在統計和機器學習常常用到。我們這裡比較三種常用的 score funciton. 

### Fisher Score of Frequentist ($\theta$ is fixed parameters)

最早 score function 是 Fisher 在 log likelihood function 首先使用。這裡的 $\theta$ 是個未知的參數，可以是 1 個變數的或是多個變數。Score function 就是 log likelihood function 的梯度，也就是 1st order derivative.  如果 $\theta \in R^1$，非常簡單，score function 就是斜率。如果是 $\theta \in R^n$， $s_{\theta} \in R^{n\times n}$.
Score function 為 0 的點，原則上就是 maximum likelihood 的參數值。

$$
S(\theta) = \frac{\partial}{\partial \theta} \log \mathcal{L}(\theta; x)
$$

Log likelihood function 的**負**二階導數是在 maximum likelihood 為 0 的負值稱為 Fisher information，也就是負曲率，永遠為正。而且可以證明 Fisher information = E\[(score)^2\].  也就是說是 score function 的 variance!  

和 information theory 的 Shannon information 不同，Fisher information 越大，代表 (負) 曲率大，也就是 likelihood function 很窄，代表估計參數越準確。反之 Fisher information 越小，代表參數估計不準確。在 frequentist 的眼中，Fisher information 是作用在 likelihood function.  Shannon information 是作用在 pdf.  兩者並不相同。


### Expand Fisher Score to Bayesian ($\theta$ is a distribution)

對於 Bayesian 而言，$\theta$  基本是一個 hidden random variable, 稱為 **prior information**, 而不是一個固定的參數。基本把原來的 likelihood function 變成 conditional pdf.
$$
S(\theta) = \frac{\partial}{\partial \theta} \log p(X | \theta)
$$
However, Bayesian inference incorporates prior information about parameters through prior distributions, leading to a posterior distribution defined by Bayes' theorem:
$$
p(\theta | X) = \frac{p(X | \theta)p(\theta)}{p(X)}
$$

#### Fisher信息在貝葉斯統計中的作用

1. **Non-Informative Priors**：通過Jeffrey規則，先驗分布應該與Fisher信息的平方根成正比 (就是 score function 的標準差)，這有助於確保先驗在模型重新參數化下的不變性。
2. **Posterior Distribution**：隨著樣本大小的增加，後驗分布趨近於正態分布。這種漸近正態分布的 covariance 由Fisher信息決定。
3. **拉普拉斯近似**：Fisher信息被用來估計 log posterior 在其模式附近的曲率。

### Fisher Score is Gradient ($\theta$ is a neural network to approximation)

此處 score function 基本只和 p(x) 的分佈有關。其實就是 gradient or vector field.  
如果是 $x \in R^n$， $s_{x} \in R^{n\times n}$.  Score function 在 diffusion 扮演非常重要的角色。


### 比較

這裡是**最大似然估計（MLE）**、**貝葉斯統計**和**擴散方法**中的得分函數定義比較：

| **方面**    | **MLE Score**                                                           | **貝葉斯 Score**                                     | **Diffusion Score**                                        |
| --------- | ----------------------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| **定義**    | 對模型參數的對數似然函數的梯度。                                                        | 對對數似然函數的梯度，但在貝葉斯框架內解釋。                            | 相對於資料的概率密度函數（pdf）的梯度。                                      |
| **數學形式**  | $S(\theta) = \nabla_\theta \log L(\theta; x)$, 其中 $L(\theta; x)$ 是似然函數。 | $s(\theta) = \nabla_\theta \log p(X\vert \theta)$ | $s(x) = \nabla_x \log p_{\theta}(x)$, 其中 $p(x)$ 是資料分布的pdf。 |
| **目的**    | 通過解決 $S(\theta) = 0$ 來尋找參數估計值。                                          | 用於更新對貝葉斯推斷中參數的信息信念。                               | 通過學習噪聲資料分佈中的梯度來指導樣本生成，用於生成建模。                              |
| **背景和應用** | 頻率派參數估計。通過最大似然估計來估算模型參數。                                                | 貝葉斯推斷，包括先驗和後驗資訊。更新貝葉斯分析中的後驗分佈。                    | 在 score-base diffusion 學習擴散模型中的逆過程，從噪聲生成影像樣本。              |
| **How?**  | 已知 $S(\theta)=0$ 求解 $\theta$                                            | How?                                              | 訓練 $\theta$ neural net.  具體方法是 denoiser!                   |
