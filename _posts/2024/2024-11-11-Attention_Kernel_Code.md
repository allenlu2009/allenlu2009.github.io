---
title: Attention as SVM Kernel Interpretation
date: 2024-11-11 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1027-Attention_Kernel]]"
categories:
  - GenAI
tags:
  - Attention
  - Kernel
  - LLM
  - Transformer
---

[[2024-10-11-Linear_Attention]]
[[2024-10-10-Attention_Math]]

## Source
Good summary article of Performer: https://ahmdtaha.medium.com/rethinking-attention-with-performers-part-ii-final-c1809de2478a

科學空間，check it : https://www.kexue.fm/archives/7546/comment-page-1

FAST algorithm : https://arxiv.org/pdf/2009.14794
https://teddykoker.com/2020/11/performers/

https://teddykoker.com/2020/11/performers/



## Performer 的基礎：FAVOR+

![[Pasted image 20241111233110.png]]

1. Fast-Attention (FA),
2. **P**ositive Random Feature (**+/P**RF),
3. Orthogonal Random features (ORF).


### **FA : Fast Attention**

更有趣的想法，是把 $\phi(Q), \phi(K)$ 視爲 frequency transform, i.e. (random) FFT.

![[Pasted image 20241111233206.png]]

$L_L$是 typo, 應該是$I_L$.
![[Pasted image 20241111233229.png]]

By changing the order of matrix multiplication, this formulation escapes the _O(L²)_ space complexity and reduces time complexity from _O(L² d)_ to _O(Lrd)_

### **Positive Random Feature (PRF)**



$$
e^{q \cdot k}=\mathrm{E}_{\omega \sim \mathrm{N}\left(\omega ; 0,1_d\right)}\left[e^{w \cdot q-\|q\|^2 / 2} \times e^{w \cdot k-\|k\|^2 / 2}\right] \approx \frac{1}{\sqrt{m}}\left(\begin{array}{c}
e^{w_1 \cdot q-||q||^2 / 2} \\
e^{w_2 \cdot q-||q||^2 / 2} \\
\ldots \\
e^{w_m \cdot q-| | q \|^2 / 2}
\end{array}\right) \cdot \frac{1}{\sqrt{m}}\left(\begin{array}{c}
e^{w_1 \cdot k-||k||^2 / 2} \\
e^{w_2 \cdot k-||k||^2 / 2} \\
\ldots \\
e^{w_m \cdot k-||k||^2 / 2}
\end{array}\right)=\tilde{q} \cdot \tilde{k}
$$

上式表示从标准正态分布$\mathrm{N}\left(\omega ; 0,1_d\right)$中采样足够多的$\omega$，然后计算$e^{w \cdot q-|q|^2 / 2} \times e^{w \cdot k-|k|^2 / 2}$的数学期望，结果等于$e^{q \cdot k}$。**上式有非常的好處是均為正值！**  原來的 Kernel method 是用$\sin, \cos$就會有正有負。

上式可以另外表示 (更簡潔) 如下：$\mathbf{z} = \mathbf{x} + \mathbf{y}$

$$
\mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)}\left[\exp \left(\omega^{\top} \mathbf{x}-\frac{\|\mathbf{x}\|^2}{2}\right) \exp \left(\omega^{\top} \mathbf{y}-\frac{\|\mathbf{y}\|^2}{2}\right)\right]=\Lambda \mathbb{E}_{\omega \sim \mathcal{N}\left(0, \mathbf{I}_d\right)} \cosh \left(\omega^{\top} \mathbf{z}\right)
$$

$$
\text { where } \Lambda=\exp \left(-\frac{\|\mathbf{x}\|^2+\|\mathbf{y}\|^2}{2}\right) \text { and cosh is hyperbolic cosine. }
$$


![[Pasted image 20241111233536.png]]

![[Pasted image 20241111233555.png]]

## Code

https://github.com/teddykoker/performer/blob/main/performer.py

最重要的就是 $\phi(x)$ function

The following code define transformed feature mapping $\phi(x)$ given by:

$$
\phi(x) = \frac{h(x)}{\sqrt{m}} \left[ f_1(x R^\top),\ f_2(x R^\top),\ \dots,\ f_k(x R^\top) \right]
$$

```python
# random feature map
def phi(h, fs, random_feats):
    return lambda x: (
        h(x)
        / np.sqrt(m)
        * np.concatenate(
            [f(np.einsum("...d,md->...m", x, random_feats)) for f in fs],
            axis=-1,
        )
    )

```

說明如下

The provided code defines a function `phi` that constructs and returns another function (a lambda function) based on the inputs `h`, `fs`, and `random_feats`. Here's a step-by-step explanation of what the code does:

1. **Inputs**:
   - `h`: A function that takes an input `x` and returns a scalar or vector. It could represent a scaling factor or some transformation applied to `x`.
   - `fs`: A list of functions `[f1, f2, ..., fk]`. Each `f` in `fs` is a function that operates on the result of a linear transformation of `x`.
   - `random_feats`: A NumPy array of shape `(m, d)`, where `m` is the number of random features and `d` is the dimensionality of the input `x`. This array acts as a matrix of random weights for feature transformation.

2. **Lambda Function**:
   - The `phi` function returns a lambda function that takes an input `x`.

3. **Inside the Lambda Function**:
   - **Linear Transformation**:
     - `np.einsum("...d,md->...m", x, random_feats)` computes the matrix multiplication of `x` with the transpose of `random_feats`. This results in a new vector of size `m` for each input `x`. Essentially, it's computing$x R^\top$, where $R$ is `random_feats`.
   - **Applying Functions from `fs`**:
     - For each function `f` in `fs`, it applies `f` to the result of the linear transformation. This is done through a list comprehension:
       ```python
       [f(np.einsum("...d,md->...m", x, random_feats)) for f in fs]
       ```
     - Each `f` transforms the linearly projected features.
   - **Concatenation**:
     - The results from all functions in `fs` are concatenated along the last axis using `np.concatenate(..., axis=-1)`. This forms a single vector containing all transformed features.
   - **Scaling**:
     - The concatenated vector is multiplied by `h(x) / np.sqrt(m)`. This scales the features by the function `h(x)` and normalizes by the square root of the number of random features `m`.

4. **Final Output**:
   - The lambda function returns the final transformed and scaled feature vector for any input `x`.

### Final Formula

The mathematical representation of the code is as follows:

Let:
- $x \in \mathbb{R}^d$ be the input vector.
- $h: \mathbb{R}^d \rightarrow \mathbb{R}$ be a scaling function.
- $R \in \mathbb{R}^{m \times d}$ be the random features matrix (`random_feats`).
- $\{ f_1, f_2, \dots, f_k \}$ be the list of functions in `fs`.

Then, the transformed feature mapping $\phi(x)$ is given by:

$$
\phi(x) = \frac{h(x)}{\sqrt{m}} \left[ f_1(x R^\top),\ f_2(x R^\top),\ \dots,\ f_k(x R^\top) \right]
$$

Where:
- $x R^\top$computes the linear transformation of$x$using the random features.
- $f_i(x R^\top)$applies the $i$-th function to the transformed features.
- The square brackets $[ \cdot ]$ denote the concatenation of the resulting vectors along the last axis.
- The division by $\sqrt{m}$ normalizes the feature vector.

**Explanation of the Formula Components**:

- **Linear Transformation** ($x R^\top$):
  - Projects the input vector$x$into a new feature space using the random features matrix$R$.
- **Function Applications** ($f_i(x R^\top)$):
  - Each function$f_i$transforms the projected features in a specific way, allowing for nonlinear transformations or feature mappings.
- **Concatenation**:
  - Combining the outputs from all $f_i$ functions expands the feature representation, capturing diverse aspects of the data.
- **Scaling and Normalization**:
  - Multiplying by $h(x)$ scales the entire feature vector, possibly incorporating additional information from $x$.
  - Dividing by $\sqrt{m}$ ensures that the magnitude of the feature vector doesn't grow uncontrollably with the number of random features.

**Usage Context**:

This kind of feature mapping is often used in machine learning algorithms, such as kernel approximations or feature transformations for neural networks, to project input data into higher-dimensional spaces where it may be more easily separable or to capture nonlinear relationships.

#### Einstein Summation

The code snippet `np.einsum("...d,md->...m", x, random_feats)` performs a specific tensor contraction (generalized matrix multiplication) between the arrays `x` and `random_feats`. Here's a detailed mathematical explanation of what this code does:

### Shapes of the Inputs

- **`x`**: An array of shape `(..., d)`. The `...` represents any number of leading dimensions (could be none), and `d` is the size of the last dimension.
- **`random_feats`**: An array of shape `(m, d)`, where `m` is the number of random features, and `d` matches the last dimension of `x`.

### Einstein Summation Notation

The `np.einsum` function uses Einstein summation notation to specify tensor contractions. The string `"...d,md->...m"` indicates:

- **`...d`**: The array `x` with any number of leading dimensions (`...`) and a last dimension `d`.
- **`md`**: The array `random_feats` with dimensions `m` and `d`.
- **`...m`**: The desired output shape after summing over the dimension `d`.

### Mathematical Formula

The operation computes the following:

$$
\text{output}_{..., m} = \sum_{d=1}^{d} x_{..., d} \cdot \text{random\_feats}_{m, d}= \sum_{d=1}^{d} x_{..., d} \cdot R_{m, d}
$$

This means for each index in the leading dimensions (`...`) and for each `m` from `1` to `m`:

1. **Element-wise Multiplication**: Multiply each element of `x` along its last dimension `d` with the corresponding element in `random_feats` along dimension `d`.
2. **Summation over `d`**: Sum these products over the index `d`.
3. **Result**: The result is stored in the output array at position `(..., m)`.

### Equivalent Matrix Multiplication

This operation is equivalent to a matrix multiplication between `x` and the transpose of `random_feats`:

$$
\text{output} = x \cdot \text{random\_feats}^\top = x \cdot R^\top
$$

- **If `x` is 2D** (shape `(n, d)`):
  - The result is of shape `(n, m)`.
  - Each row of `x` is multiplied by the transpose of `random_feats`, resulting in a new feature representation.
- **For higher dimensions**:
  - The operation is broadcasted over the leading dimensions.

### Visual Representation

Imagine `x` and `random_feats` as matrices:

- **`x`**:
  $$
  \begin{bmatrix}
  x_{..., 1} & x_{..., 2} & \dots & x_{..., d}
  \end{bmatrix}
  $$
- **`random_feats`**:
  $$
  \begin{bmatrix}
  R_{1, 1} & R_{1, 2} & \dots & R_{1, d} \\
  R_{2, 1} & R_{2, 2} & \dots & R_{2, d} \\
  \vdots & \vdots & \ddots & \vdots \\
  R_{m, 1} & R_{m, 2} & \dots & R_{m, d}
  \end{bmatrix}
  $$

- **Output**:
  $$
  \text{output}_{..., m} = x_{..., :} \cdot R_{m, :}^\top
  $$

### Summary

- **Purpose**: Projects the input `x` into a new feature space using the random feature matrix `random_feats`.
- **Operation**: Computes the dot product between `x` and each row of `random_feats`, summing over the shared dimension `d`.
- **Result**: An array of shape `(..., m)` containing the transformed features.

**In essence**, the code performs a linear transformation of `x` using the random features, effectively mapping `x` into a new feature space of dimension `m`.

---

**Note**: This kind of operation is common in machine learning for feature transformation, where random projections are used to approximate kernel functions or to reduce dimensionality while preserving the structure of the data.


接下來就是定義 $h(x)$ 和 $f(x)$

- $\hat{SM}^{\text{trig}}$ 如下：$h(x) = \exp({\vert x\vert}^2/2)$,  $f_1(x) = \sin(2\pi x)$, $f_2(x) = \cos(2\pi x)$
- $\hat{SM}^{+}$ 如下：$h(x) = \exp(-{\vert x\vert}^2/2)$,  $f_1(x) = \exp(x)$

```python
# Performer "sin/cos" attention
def sincos_att_hat(q, k, v, random_feats, normalize=True):
    def h(x):
        return np.exp(np.square(x).sum(axis=-1, keepdims=True) / 2)

    sin = lambda x: np.sin(2 * np.pi * x)
    cos = lambda x: np.cos(2 * np.pi * x)

    kernel = phi(h, [sin, cos], random_feats)
    return att_hat(q, k, v, kernel, normalize)


# Performer "positive" attention
def positive_att_hat(q, k, v, random_feats, normalize=True):
    def h(x):
        return np.exp(-np.square(x).sum(axis=-1, keepdims=True) / 2)

    kernel = phi(h, [np.exp], random_feats)
    return att_hat(q, k, v, kernel, normalize)
```

### **ORF**

Gram-Schmidt renormalization.  就是把$\omega_1, \omega_2, ...\omega_m$ 重新 renormalize 成正交 basis.

![[Pasted image 20241111233652.png]]

GS renormalization 實際的計算可以使用 QR decomposition.

$$
\begin{aligned}
&\begin{aligned}
A & =\left[\begin{array}{lll}
\mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_3
\end{array}\right] \\
& =\left[\begin{array}{lll}
\mathbf{q}_1 & \mathbf{q}_2 & \mathbf{q}_3
\end{array}\right]\left[\begin{array}{ccc}
r_{11} & r_{12} & r_{13} \\
0 & r_{22} & r_{23} \\
0 & 0 & r_{33}
\end{array}\right] \\
& =\left[\begin{array}{lll}
\mathbf{q}_1 & \mathbf{q}_2 & \mathbf{q}_3
\end{array}\right]\left[\begin{array}{ccc}
\mathbf{q}_1^T \mathbf{x}_1 & \mathbf{q}_1^T \mathbf{x}_2 & \mathbf{q}_1^T \mathbf{x}_3 \\
0 & \mathbf{q}_2^T \mathbf{x}_2 & \mathbf{q}_2^T \mathbf{x}_3 \\
0 & 0 & \mathbf{q}_3^T \mathbf{x}_3
\end{array}\right] \\
& =Q R .
\end{aligned}\\
&\text { 因為 } \mathbf{y}_i=\left\|\mathbf{y}_i\right\| \mathbf{q}_i \text { 且 } \mathbf{q}_i \perp \operatorname{span}\left\{\mathbf{y}_1, \ldots, \mathbf{y}_{i-1}\right\} \text { ，主對角元 } r_{i i} \text { 亦可表示為 }\\
&r_{i i}=\mathbf{q}_i^T \mathbf{x}_i=\mathbf{q}_i^T \mathbf{y}_i=\mathbf{q}_i^T\left(\left\|\mathbf{y}_i\right\| \mathbf{q}_i\right)=\left\|\mathbf{y}_i\right\|
\end{aligned}
$$
Q 矩陣就是 unitary matrix (正交且長度為 1)。


```python
# generate IID Gaussian random features
def iid_gaussian(m, d):
    return np.random.normal(size=(m, d))

# generate orthogonal Gaussian random features
def orthogonal_gaussian(m, d):
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = np.linalg.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = np.vstack(blocks)
    matrix /= np.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix

```

重點在於 `np.linalg.gr`  做 QR decomposition of a normal distribution.  為什麼是 d x d 方陣？解釋如下。

# Generating Orthogonal Gaussian Random Features

The provided code defines a function `orthogonal_gaussian(m, d)` that generates an $m \times d$matrix of orthogonal Gaussian random features. This matrix can be used in machine learning applications for feature transformations, especially in scenarios where orthogonal projections are beneficial.

Here's a detailed explanation of the code:
### Final Matrix $W$

$$
W = \frac{1}{\sqrt{\text{num\_squares} + \frac{r}{d}}} \begin{bmatrix}
Q_1^\top \\
Q_2^\top \\
\vdots \\
Q_{\text{num\_squares}}^\top \\
Q_{\text{partial}}^\top \text{ (if remainder exists)}
\end{bmatrix}
$$


```python
def orthogonal_gaussian(m, d):
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = np.linalg.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = np.vstack(blocks)
    matrix /= np.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix
```

## Step-by-Step Explanation

### 1. Function Definition

```python
def orthogonal_gaussian(m, d):
```

- **Purpose**: To generate an $m \times d$ matrix of orthogonal Gaussian random features.
- **Parameters**:
  - `m`: Number of random features (rows of the output matrix).
  - `d`: Dimensionality of each feature (columns of the output matrix).

### 2. Inner Function: Generating Orthogonal Square Matrices

```python
def orthogonal_square():
    # create orthogonal square matrix using Gram-Schmidt
    q, _ = np.linalg.qr(iid_gaussian(d, d))
    return q.T
```

- **Purpose**: Generates a $d \times d$ orthogonal matrix.
- **Process**:
  - `iid_gaussian(d, d)`: Generates a $d \times d$ matrix with entries drawn from the standard normal distribution $\mathcal{N}(0, 1)$.
  - `np.linalg.qr(...)`: Performs QR decomposition on the generated matrix.
    - `q`: An orthogonal matrix (columns are orthonormal vectors).
    - `_`: The upper triangular matrix `R` from the QR decomposition (not used).
  - `q.T`: Transposes `q` to get orthonormal vectors as rows.

### 3. Calculating the Number of Full Blocks

```python
num_squares = int(m / d)
```

- **Purpose**: Determines how many full $d \times d$ orthogonal matrices fit into `m`.
- **Calculation**:
  - `num_squares`: The integer division of `m` by `d`.

### 4. Creating the Blocks

```python
blocks = [orthogonal_square() for _ in range(num_squares)]
```

- **Purpose**: Generates a list of full $d \times d$ orthogonal matrices.
- **Result**: A list containing `num_squares` orthogonal matrices.

### 5. Handling the Remainder

```python
remainder = m - d * num_squares
if remainder:
    blocks.append(orthogonal_square()[:remainder])
```

- **Purpose**: Accounts for any remaining rows if `m` is not a multiple of `d`.
- **Process**:
  - `remainder`: Calculates the number of additional rows needed.
  - If `remainder` is non-zero:
    - Generates another $d \times d$ orthogonal matrix.
    - Takes the first `remainder` rows using `[:remainder]`.
    - Appends this partial block to the list of blocks.

### 6. Stacking the Blocks

```python
matrix = np.vstack(blocks)
```

- **Purpose**: Vertically stacks all the blocks to form a single matrix.
- **Result**: An $m \times d$ matrix.

### 7. Normalizing the Matrix

```python
matrix /= np.sqrt(num_squares + remainder / d)
```

- **Purpose**: Normalizes the matrix to maintain consistent variance.
- **Explanation**:
  - Divides the entire matrix by the square root of the total number of blocks (including partial blocks as fractions).
  - Ensures that the magnitude of the random features does not grow with the number of blocks.

### 8. Optional Scaling (Commented Out)

```python
# matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix
```

- **Purpose**: If uncommented, this line would scale the matrix by multiplying it with a diagonal matrix.
- **Effect**:
  - Each row of the matrix would be scaled by $\sqrt{d}$.
  - Currently commented out, so it does not affect the execution.

### 9. Returning the Matrix

```python
return matrix
```

- **Result**: The final $m \times d$ matrix of orthogonal Gaussian random features is returned.

## Mathematical Interpretation

### Generating Orthogonal Matrices

- **Orthogonal Square Matrices**:
  - For each full block, we generate a $d \times d$ matrix $A$with entries $A_{ij} \sim \mathcal{N}(0, 1)$.
  - Perform QR decomposition: $A = QR$.
  - The orthogonal matrix $Q$satisfies $Q^\top Q = I$.

### Assembling the Final Matrix

- **Full Blocks**:
  - We have $\text{num\_squares}$full blocks of size $d \times d$.
- **Partial Block**:
  - If there is a remainder $r$, we have an additional block of size $r \times d$.
- **Stacking**:
  - The blocks are stacked vertically to form matrix $W$of size $m \times d$.
- **Normalization**:
  - The matrix $W$is normalized by dividing by $\sqrt{\text{num\_squares} + \frac{r}{d}}$.

### Final Matrix $W$

$$
W = \frac{1}{\sqrt{\text{num\_squares} + \frac{r}{d}}} \begin{bmatrix}
Q_1^\top \\
Q_2^\top \\
\vdots \\
Q_{\text{num\_squares}}^\top \\
Q_{\text{partial}}^\top \text{ (if remainder exists)}
\end{bmatrix}
$$

- $Q_i^\top$: The transpose of the orthogonal matrix from the $i$-th QR decomposition.
- **Properties**:
  - The rows of $W$ are orthonormal within each block.
  - The scaling ensures that the overall variance is consistent.

## Purpose and Applications

- **Random Projections**: Orthogonal random projections can preserve the structure of the data better than random Gaussian projections.
- **Kernel Approximation**: In techniques like Random Fourier Features, orthogonal projections can improve the approximation of kernel functions.
- **Neural Networks**: Initializing weights with orthogonal matrices can help in training deep networks by maintaining the variance of activations across layers.

## Additional Notes

- **Why Use QR Decomposition?**:
  - QR decomposition provides an efficient way to obtain an orthogonal matrix from a random Gaussian matrix.
- **Normalization Factor**:
  - The normalization ensures that when the matrix $W$is used to project data, the resulting features have a controlled variance, preventing the projections from becoming too large or too small.
- **Commented Scaling Line**:
  - The commented line suggests an alternative method for scaling, potentially adjusting the variance based on the input dimensionality $d$.

## Example Usage

Assuming you have input data `X` of shape `(n_samples, d)`, you can use the generated matrix to project your data:

```python
W = orthogonal_gaussian(m, d)
X_projected = X @ W.T  # Resulting shape: (n_samples, m)
```

- **Result**: `X_projected` contains the transformed features.

---

**Summary**: The `orthogonal_gaussian` function generates an $m \times d$ matrix of orthogonal Gaussian random features by stacking orthogonal matrices obtained from QR decompositions of Gaussian random matrices. The matrix is normalized to maintain consistent variance when used for projections, making it suitable for various machine learning applications that benefit from orthogonal random features.



## Source

- Linear Attention 打改变 Transformer 大模型结构垄断 : https://www.bilibili.com/video/BV1V7s9etEmQ/?spm_id_from=333.999.0.0&vd_source=a99fc6374f18662fe559d32fdc3a80cd

- Transformer are RNNs: https://arxiv.org/pdf/2006.16236

- TransNormerLLM:  https://arxiv.org/pdf/2307.14995

- Universal Transformer:  https://arxiv.org/pdf/1807.03819

- Transformer Quality in Linear Time: https://arxiv.org/pdf/2202.10447



