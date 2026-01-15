---
title: Attention as Kernel and Random Features
date: 2024-12-05 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-10-10-Attention_Math]]"
categories:
  - Science
tags:
  - Attention
  - Kernel
  - LLM
  - SVM
  - Transformer
---

[[2023-03-26-Transformer_LLM]] , [[2023-02-18-Attn_All_U_Need_Visual]]

https://teddykoker.com/2020/11/performers/
https://zetongqi.github.io/posts/2020/06/blog-post-7/

## SVM Kernel Approximation

支援向量機（SVM）是最佳的機器學習演算法之一。它具有良好的泛化能力，較低的過擬合風險，能很好地應用於高維數據。此外，核技巧（kernel trick）使得可以有效地將特徵空間提升到更高維度。而且，SVM 的優化問題通常是二次規劃 (QP) 問題，這類問題能**高效求解**，並且保證得到**全局最小值**。

然而，**SVM 也有一些缺點，其中最大的問題是它在處理大型數據集時擴展性較差，且推理時間與訓練數據集的大小有關。** 對於大小為 $n$ 的訓練集，其中 $x \in \mathbb{R}^d$，計算核矩陣 kernel matrix, $K_{ij} =k(\mathbf{x}_i, \mathbf{x}_j)$, 的計算複雜度為 $O(n^2d)$，而推理的計算複雜度為 $O(nd)$。這種時間複雜度在面對大型訓練集時表現不佳。

**注意，這裏的計算複雜度和計算 attention matrix，$\mathbf{A} = \operatorname{softmax}(\mathbf{Q} \mathbf{K}^{\top})$ 完全一樣！**


<img src="/media/image-20241027221621.png" alt="20241027221621" style="zoom:60%;" />

## SVM Code Example

Insert code

```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Create moon dataset
X, y = make_moons(n_samples=2000, noise=0.3)
colors = ['red', 'blue']
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors), s=2)
#plt.show()

# Update labels for SVM (-1 and 1 instead of 0 and 1)
y[y == 0] = -1

# Split dataset into training and testing sets
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Train SVM classifier with RBF kernel
svm_clf = SVC(kernel='rbf', C=10)
svm_clf.fit(X_train, y_train)

# Evaluate the model
accuracy = svm_clf.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Get support vectors
support_vectors = svm_clf.support_vectors_
support_indices = svm_clf.support_

# Plot support vectors and non-support vectors
X_non_sup_vec = X_train[~np.isin(np.arange(len(X_train)), support_indices)]
y_non_sup_vec = y_train[~np.isin(np.arange(len(y_train)), support_indices)]

fig, ax = plt.subplots()
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c="black", s=6, marker='x', label="Support Vectors")
ax.scatter(X_non_sup_vec[:, 0], X_non_sup_vec[:, 1], c=y_non_sup_vec, cmap=matplotlib.colors.ListedColormap(colors), s=2)
ax.legend(loc="upper right")
plt.show()

```

原始數據

Dataset: 1600 points,  2 classes.

![[Pasted image 20241206170156.png]]

SVM with RBF Kernel (map from 2D to infty D) for Classification
包含 ??? supporting vectors, and test accuracy:  92.5%

![[Pasted image 20241206170228.png]]


## Fourier Random Feature


Insert code

```python
import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers


def metropolis_hastings(p, dim, iter=1000):
    x = np.zeros(dim)
    samples = np.zeros((iter, dim))

    for i in range(iter):
        x_next = x + np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
        if np.random.rand() < p(x_next) / p(x):
            x = x_next
        samples[i] = x

    return samples

def gaussian_fourier(w):
    return (2*np.pi)**(-w.shape[0]/2) * np.exp(-LA.norm(w)/2)

def laplacian_fourier(w):
    p = 1
    for i in range(w.shape[0]):
        p *= 1 / (np.pi * (1 + w[i]))
    return p

class random_fourier_features_svm:
    def __init__(self, kernel_fourier=gaussian_fourier, D=500):
        self.kernel_fourier = kernel_fourier
        self.D = D
        
    def _feature_mapping(self, X, omega, bias):
        X_map = np.zeros((X.shape[0], self.D))
        for i in range(X.shape[0]):
            for j in range(self.D):
                X_map[i, j] = np.cos(np.dot(omega[j], X[i]) + bias[j]) * np.sqrt(2/self.D)
        return X_map
    
    def _draw_features(self, d, max_iter=1000):
        s = metropolis_hastings(self.kernel_fourier, dim=d, iter=max_iter)
        omega = s[np.random.randint(low=10, high=max_iter, size=self.D)]
        bias = np.random.uniform(low=0, high=2*np.pi, size=self.D)
        self.omega = omega
        self.bias = bias
        
    def fit(self, X_train, y_train, max_iter=1000, C=10):
        d = X_train.shape[-1]
        self._draw_features(d, max_iter=max_iter)
        X_train_map = self._feature_mapping(X_train, self.omega, self.bias)
        m,n = X_train_map.shape
        y_train = y_train.reshape(-1,1) * 1.
        K = np.dot(X_train_map, X_train_map.T)
        H = np.outer(y_train, y_train) * K
        
        P = matrix(H)
        q = matrix(-np.ones(m))
        G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        A = matrix(y_train, (1, m))
        b = matrix(0.)

        #Run solver
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        alphas = alphas.reshape(alphas.shape[0])

        sup_vec_idx = np.argwhere(np.logical_or(alphas > 1e-4, alphas < -1e-4))
        sup_vec_idx = sup_vec_idx.reshape(sup_vec_idx.shape[0])
        X_sup_vec = X_train_map[sup_vec_idx]
        y_sup_vec = y_train[sup_vec_idx]
        alphas_sup_vec = alphas[sup_vec_idx]
        
        w = np.zeros(self.D)
        for i in range(alphas_sup_vec.shape[0]):
            w += alphas_sup_vec[i] * y_sup_vec[i] * X_sup_vec[i]
        b = - (np.max(np.dot(X_train_map[np.where(y_train == -1)[0]], w)) + np.min(np.dot(X_train_map[np.where(y_train == 1)[0]], w))) / 2
        self.w = w
        self.b = b
        self.sup_vec_idx = sup_vec_idx
        
    def evaluate(self, X_test, y_test):
        X_test_map = self._feature_mapping(X_test, self.omega, self.bias)
        return np.sum((np.sign(np.dot(X_test_map, self.w) + self.b) == y_test).astype("int")) / X_test_map.shape[0]
    

X, y = make_moons(n_samples=2000, noise=0.3)
colors = ['red','blue']
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors), s=2)
y[y == 0] = -1
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)    


clf = random_fourier_features_svm()
clf.fit(X_train, y_train)
accuracy = clf.evaluate(X_test, y_test)
#accuracy = clf.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")


X_sup_vec_original = X_train[clf.sup_vec_idx]
X_non_sup_vec = X_train[~np.isin(np.arange(len(X_train)), clf.sup_vec_idx)]
y_non_sup_vec = y_train[~np.isin(np.arange(len(y_train)),clf. sup_vec_idx)]
y_non_sup_vec = y_non_sup_vec.reshape(y_non_sup_vec.shape[0])
fig, ax = plt.subplots()
colors = ['red','blue']
ax.scatter(X_sup_vec_original[:, 0], X_sup_vec_original[:, 1], c="black", s=6, marker='x', label="Support Vectors")
ax.scatter(X_non_sup_vec[:, 0], X_non_sup_vec[:, 1], c=y_non_sup_vec, cmap=matplotlib.colors.ListedColormap(colors), s=2)
ax.legend(loc="upper right")
plt.show()


```


SVM with RBF Kernel (map from 2D to 500 D) for Classification
包含 ~380 supporting vectors, and test accuracy:  91.5%-93.25%

Solver: CVX QP (Quad Programming)

![[Pasted image 20241206170624.png]]


## Source

- Linear Attention 打改变 Transformer 大模型结构垄断 : https://www.bilibili.com/video/BV1V7s9etEmQ/?spm_id_from=333.999.0.0&vd_source=a99fc6374f18662fe559d32fdc3a80cd

- Transformer are RNNs: https://arxiv.org/pdf/2006.16236

- TransNormerLLM:  https://arxiv.org/pdf/2307.14995

- Universal Transformer:  https://arxiv.org/pdf/1807.03819

- Transformer Quality in Linear Time: https://arxiv.org/pdf/2202.10447



