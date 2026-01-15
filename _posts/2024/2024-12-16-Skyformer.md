---
title: Sky (Symmetric) Transformer
date: 2024-12-08 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-08-14-LLM_Adapting]]"
categories:
  - Science
tags:
  - Attention
  - LLM
---


## Source

Original Skyformer paper (2021 BERT): https://arxiv.org/pdf/2111.00035
Another symmetric attention (2024 BERT): https://arxiv.org/pdf/2406.06366

SKY (Symmetrization Kernelized attention for NYstrom emthod)
## Overview


## Pytorch Matrix Multiplication Trade-offs 

From Perplexity

|Method|Usability|Performance|Flexibility|
|---|---|---|---|
|`@` Operator|Very intuitive|Fast|Limited|
|`torch.matmul()`|Clear and explicit|Fast|Limited|
|`torch.einsum()`|Complex notation|Potentially slower|Highly flexible|


## Conclusion

For most standard matrix multiplications, either the `@` operator or `torch.matmul()` will suffice and perform efficiently. However, when dealing with more complex tensor operations that require specific summation patterns or manipulations, `torch.einsum()` offers significant advantages despite potential performance trade-offs. Choosing the right method depends on the specific requirements of your computation task in PyTorch.


Another explanation from ChatGPT.

|**Method**|**Flexibility**|**Performance**|**Readability**|**Notes**|
|---|---|---|---|---|
|`torch.mm`|Low|High|High|Best for 2D tensors only.|
|`torch.matmul`|High|Moderate|Moderate|Handles batch dimensions; prone to silent broadcasting.|
|`torch.bmm`|Moderate|High|Moderate|Optimized for 3D batched tensors.|
|`torch.einsum`|Very High|Moderate|Low-Moderate|Great for complex operations; slower than specialized ops.|
|`torch.tensordot`|High|Moderate|Moderate|Flexible contractions, but requires explicit dims.|
|`@` operator|Low|High|High|Clean and concise for matrix multiplication.|
|`torch.outer`|Low|High|High|Specialized for outer products.|
|`torch.mul`|Moderate|High|High|Element-wise only.|

For **general-purpose use**:

- Use `torch.matmul` for its versatility.
- Use `torch.einsum` for advanced operations requiring flexibility.
- Use `torch.bmm` or `torch.mm` for performance-critical, batch-specific tasks.


## Skyformer High-Level Overview
This code implements a specific type of transformer attention mechanism named `Skyformer`, which leverages sketching techniques to approximate the kernel-based attention, aiming to reduce computational costs for long sequences. The model includes specialized kernels (`kernel_RS_RBF`, `kernel_SM`, etc.) and incorporates iterative inverse approximation (`iterative_inv`) for normalizing the sketched attention weights.

### Code Breakdown

#### Linearized Attention

- **`linear_attention(q, k, v)`**
- 只用三行 code 就描述 linear attention with normalization!    
1. **Linearized Softmax Attention**:
   Compute the attention as:
   $$
   \text{LinearAttention}(Q, K, V) = D^{-1}(Q \cdot (K^T \cdot V))
   $$
   Where:
   - $D$ is the normalization term defined as:
     $$
     D = Q \cdot K^T \cdot \mathbf{1}
     $$
     This sums over the keys.

3. **Steps in Linear Attention**:
   - Compute the cumulative sum of the keys:
     $$
     K_{\text{cumsum}} = \sum K
     $$
   - Normalize using $D^{-1}$:
     $$
     D^{-1} = \frac{1}{Q \cdot K_{\text{cumsum}}}
     $$
   - Calculate the context vector:
     $$
     \text{Context} = \sum K \cdot V
     $$
   - Combine to produce the output:
     $$
     \text{Output} = D^{-1} \cdot (\text{Context} \cdot Q)
     $$


- **`rbf_attention(q, k, v)`**:
   - Performs RBF kernel-based attention similar to `linear_attention` but with an additional normalization step.

#### Kernel Functions
These compute kernel transformations based on the input query (q) and key (k) matrices.

1. **`kernel_SM`**:
   - Computes the standard softmax kernel for given inputs. 
   - Applies matrix exponentiation directly to the inner product of `q` and `k`.

2. **`kernel_RS_SM`**:
   - Similar to `kernel_SM` but includes random sign multiplication for efficient computation.
   - Handles cases where sketched keys are accumulated (`X2_accu=True`).

3. **`kernel_RS_RBF`**:
   - Computes the RBF kernel with random sign multiplication for approximation.
   - Includes distance-based scaling using the diagonal of `q` and `k`.



#### Sketching Mechanism

- **`kernel_sketch`**:
   - Projects the concatenated `q` and `k` matrices into a lower-dimensional space using a sketching matrix.
   - Computes sketched softmax kernels (`AS`) for dimensionality reduction.

#### Inverse Normalization

- **`iterative_inv(mat, n_iter=6)`**:
   - Iteratively computes an approximate matrix inverse using a series expansion.
   - Starts with an initial approximation (`V`) scaled by the largest sum of rows/columns.
   - Refines the inverse approximation using recursive updates.

#### Skyformer Class

**Initialization**:
```python
class Skyformer(nn.Module):
    def __init__(self, config):
        ...
```
- Initializes hyperparameters such as sequence length (`max_seq_len`), number of features (`nb_features`), and attention head dimensions (`dim_heads`).
- Configures the kernel function (`self.kernel_fn`) based on the selected sketched kernel type.

**Uniform Sketching**:
```python
@torch.no_grad()
def uniform_sketching(self, n, nb_rows, nb_columns, non_padding_num):
```
- Randomly selects rows and columns for sketching matrices.
- Applies a random sign for efficient kernel approximation.

**Forward Pass**:
```python
def forward(self, q, k, v, mask):
```
1. **Preprocessing**:
   - Normalizes `q`, `k`, and `v` using the mask and a scaling factor.

2. **Sketching**:
   - Generates sketching matrices (`self.sketching_matrix`) and computes sketched kernel representations (`AS`).

3. **Inverse Normalization**:
   - Constructs a sketch-based self-attention approximation using `STAS` (sketched kernel matrix).
   - Normalizes and computes its inverse (`STAS_inv`) via `iterative_inv`.

4. **Attention Context**:
   - Computes the context vector (`context`) by combining the sketched `K` matrix and the value vectors (`v`).
   - Outputs the final attention (`out`).

### Key Features
1. **Kernel Approximation**:
   - Reduces computational complexity using sketching and kernel-based approximations.
   - Supports various kernels (softmax, RBF).

2. **Efficient Inverse Computation**:
   - Utilizes iterative approximations for inverting matrices to avoid direct inversion.

3. **Memory-Efficient Sketching**:
   - Randomized sketching reduces memory usage while preserving essential information.

### Use Cases
- Designed for long-sequence attention tasks where computational cost and memory usage are bottlenecks.
- Suitable for scenarios where approximations (e.g., kernel methods) are acceptable without significant loss in accuracy.


```python
import math
import time
import torch
from torch import nn
import torch.linalg as la

from functools import partial
from models.attention import SoftmaxAttention as SelfAttention
from config import Config
from torch.autograd import Function, gradcheck

def linear_attention(q, k, v): # for SM kernel
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out



def kernel_SM(X1, X2=None, X2_accu=False):
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...mnd', X1, X2)
        # product = torch.matmul(X1.unsqueeze(dim=2), torch.transpose(X2, 3, 4))
        result = torch.exp(product)

        result = result.sum(dim=2)
        # print(result.shape)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)

    return result
    # return result, product

def kernel_RS_SM(X1, X2=None, X2_accu=False, random_sign=None):
    # RS for random sign
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...mnd', X1, X2)
        # product = torch.matmul(X1.unsqueeze(dim=2), torch.transpose(X2, 3, 4))
        result = torch.exp(product)
        result = torch.transpose(result, 2, 3) # nmd
        result = result * random_sign
        result = result.sum(dim=3)

    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)

    return result
    
def kernel_RS_SM1(X1, X2=None, X2_accu=False, random_sign=None):
    if X2 is None:
        X2 = X1
        X2_accu = False
    if X2_accu:
        product = torch.einsum('...np,...mdp->...nmd', X1, X2)
        result = torch.exp(product)
        result = torch.einsum('bhnmd,...bmd->...bhnd', result, random_sign)
        # result = (result.transpose(0, 2) * random_sign).sum(-2).transpose(0, 2) # nhbmd -> nhbd -> bhnd

    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2)
        result = torch.exp(product)

    return result
    # return result, product

def kernel_RS_RBF(X1, X2=None, X2_accu=False, random_sign=None):

    # todo

    if X2 is None:
        X2 = X1
        X2_accu = False

    diag_X1 = (X1 * X1).sum(-1) * 0.5
    diag_X1 = diag_X1.unsqueeze(dim=-1)
    diag_X2 = (X2 * X2).sum(-1) * 0.5
    diag_X2 = diag_X2.unsqueeze(dim=-2)

    if X2_accu:
        diag_X1 = diag_X1.unsqueeze(dim=-3)
        product = torch.einsum('...np,...mdp->...mnd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)
        result = torch.transpose(result, 2, 3) # nmd
        result = torch.einsum('bhnmd,bmd->bhnd', result, random_sign)
    else:
        product = torch.einsum('...np,...dp->...nd', X1, X2) - diag_X1 - diag_X2
        result = torch.exp(product)

    return result

def rbf_attention(q, k, v): # for rbf kernel
    # todo
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

def kernel_sketch(q, k, *, kernel_fn, sketching_matrix, random_sign, normalize_data=False, eps=1e-4, device = None):
    # sketching_matrix: (self.M, self.d) tensor
    # sketching_matrix

    b, h, n, p = q.shape

    # data_normalizer = (p ** -0.25) if normalize_data else 1.
    # X = torch.cat([q, k], dim=2) * data_normalizer
    X = torch.cat([q, k], dim=2)
    
    XS = X.transpose(1, 2)[torch.arange(b)[:, None, None], sketching_matrix].permute(0,3,1,2,4) # bmdhp -> bhmdp
    AS = kernel_fn(X, XS, True, random_sign)
    # AS, p = kernel_fn(X, X[:,:,sketching_matrix], True)
    # if True in torch.isinf(AS):
    # print()
    # if True in torch.isnan(AS):
    # print()

    return AS.type_as(q)
        
def iterative_inv(mat, n_iter = 6, init_option = "original"):
    

    I = torch.eye(mat.size(-1), device = mat.device)
    K = mat
    
    if init_option == "original":
        V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
    else:
        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
    
    for _ in range(n_iter):
        # print(V)
    
        KV = torch.matmul(K, V)
        V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
    return V

 

# class SketchedAttentionRBF(nn.Module):
class Skyformer(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.device = config["device"] if "device" in config else "cuda"
        n = config["max_seq_len"]
        
        self.accumulation = config["accumulation"]
        sampling_factor = config["sampling_factor"]

        # nb_features = default(nb_features, int(math.sqrt(n)))
        # nb_features = default(nb_features, int(3*n ** (1.0/4)))
        nb_features = config["nb_features"] if "nb_features" in config else int(sampling_factor  * math.log(n))

        # self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')


        self.dim_heads = config["head_dim"]
        self.nb_features = nb_features

        if config["sketched_kernel"] == "kernel_RS_RBF":
            self.kernel_fn = kernel_RS_RBF

        self.no_projection = config["no_projection"]


    @torch.no_grad()
    def uniform_sketching(self, n, nb_rows, nb_columns, non_padding_num):
        
        total = nb_rows * nb_columns
        S = torch.rand(total, device=self.device)
        S = torch.einsum("b,d->bd", non_padding_num, S).long()
        S[:, total//2:] = S[:, total//2:] + n
        S = S.reshape(-1, nb_rows, nb_columns)
        # random_sign = (torch.randint(2, S.shape, device=self.device) * 2 - 1) * (math.sqrt(2 * n) / nb_rows / nb_columns)
        random_sign = torch.ones(S.shape, device=self.device)

        return S, random_sign

    def forward(self, q, k, v, mask):

        
        device = q.device
        b, h, n, d = q.shape
        
        data_normalizer = (32 ** -0.25)
        

        q = q * (mask[:, None, :, None] * data_normalizer)
        k = k * (mask[:, None, :, None] * data_normalizer)
        v = v * mask[:, None, :, None]
        

        
        non_padding_num = mask.sum(-1) # b


        self.sketching_matrix, self.random_sign = self.uniform_sketching(
            n, self.accumulation, self.nb_features, non_padding_num) # bmd
        

        create_kernel_sketch = partial(kernel_sketch, kernel_fn = self.kernel_fn,
           sketching_matrix = self.sketching_matrix, random_sign=self.random_sign, device = device)
        AS = create_kernel_sketch(q, k)  # b,h,2n, nb_feat
        Q = AS[:,:,:n] # b, h, n, nb_feat
        

        STAS = AS.transpose(1, 2)[torch.arange(b)[:, None, None], self.sketching_matrix] # bnhd -> bmdhd
        STAS = torch.einsum('bmdhe,bmd->bhde', STAS, self.random_sign) # bmdhd -> bhdd

        STAS = STAS + 1e-1*torch.eye(STAS.shape[-1], device=self.device)
 
        K = AS[:,:,n:]
        
        
        ##################################################################
        D_STAS_inv = 1 / STAS.sum(-1)
        D_STAS_inv = torch.sqrt( D_STAS_inv)
        STAS = torch.einsum("...d,...de,...e->...de", D_STAS_inv, STAS, D_STAS_inv)
        
        
        
        STAS_inv = iterative_inv(STAS, 6)
        
        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv) @ STAS_inv
        

        
        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv) * mask[:, None, :, None]
        
        ##################################################################

        context = torch.einsum('...nd,...ne->...de', K, v)
        out = torch.matmul(Q, context)    
        
        return out   

```