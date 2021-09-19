---
title: Math AI: Deterministic and Probabilistic?
categories:
- AI
tags: [machine learning, deep learning, ML]
typora-root-url: ../../allenlu2009.github.io
---

Machine learning 就是一個 determinstic 和 probabilistic 擺盪和交織的過程。

(Input) dataset 一般是 deterministic.  

傳統的 machine learning technique, e.g. linear regression, SVM, decision tree, etc. 也很多是 deterministic math.  

但是 ML 背後的 modeling, 邏輯，和解釋卻可以用 probabilistic 統一解釋。例如 logistic regression, 甚至 neural network 的分類網路卻可以有 probability 的詮釋。

最後一類從頭到尾都是 probabilistic.  例如 bayesian inference, variational autoencoder.    



|                         | Input           | Model                 | Output                       | Comment                        |
| ----------------------- | --------------- | --------------------- | ---------------------------- | ------------------------------ |
| Linear regression       | (D) dataset     | (D) linear function   | (D) error                    |                                |
| Logistic regression     | (D) dataset     | (D) logistic function | (P) probability distribution | distribution is a (D) function |
| Classification NN       | (D) dataset     | (D) NN                | (P) probability distribution |                                |
| VAE encoder, training   | (D) 1-data      | (D) NN                | (P) random variable          | parameter to RV                |
| VAE decoder, generation | (P) RV 1-sample | (D) NN                | (P) random variable          | output sample                  |
| SVM                     | (D) dataset     | (D) kernel function   | (D) binary                   |                                |



absolute deterministic:  function, NN? (input/output), not parameter
deterministic/probabilistic:  random variable, distribution; ideal is random, but representation (function) is determinstic!
probabilistic:  sampling, MC

single data-point and collective data