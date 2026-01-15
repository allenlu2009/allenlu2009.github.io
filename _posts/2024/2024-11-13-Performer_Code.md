---
title: Performer Pytorch Code Analysis
date: 2024-11-13 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1111-Attention_Kernel_Code]]"
categories:
  - GenAI
tags:
  - Attention
  - Kernel
  - Transformer
---

[[2024-10-11-Linear_Attention]]
[[2024-10-10-Attention_Math]]

## Source
Code base for this article. https://github.com/lucidrains/performer-pytorch
Code base for the Attention Kernel Code article:  https://teddykoker.com/2020/11/performers/  the same thing but simpler for illustration.


## Performance = FAVOR+

1. Fast-Attention (FA), V is Via
2. **P**ositive Random Feature (**+/P**RF),
3. Orthogonal Random features (ORF).


## **Fast Attention Code** 

![[Pasted image 20241111233206.png]]

下圖 $L_L$是 typo, 應該是$I_L$.
![[Pasted image 20241111233229.png]]


### FastAttention Class

A class implementing fast attention with customizable kernels, causal settings, and optional projections.

### ProjectionUpdater Class (New to Fast Attention)

白話文就是上式的 $\phi(\cdot)$ 因為是 random feature, 需要 periodically update 得到更好的 performance (Question: 可以用於 adaption during testing 類似人體骨頭的 piezoelectric effect to optimize the bone structure?)

Tracks when to update projection matrices in **FastAttention** to improve training efficiency. It only **redraws projections** at a specified interval.

這個 class (ProjectionUpdater) and function (redraw_projection_matrix) 只有在 training 使用，inferencing 不會 update.

### Transformer Layers

- **ReZero, PreScaleNorm, PreLayerNorm**: Variants of normalization layers that adapt to different normalization strategies.  
	- ReZero is for training stability.
	- Scale Norm is a simpler norm.
- **Chunk**: Divides tensors into chunks, processes each chunk independently, and recombines them, r**educing memory usage.**
- **FeedForward**: A standard feedforward network layer with optional GLU (Gated Linear Unit) activation.
- **(New 重點) Attention**: Implements a multi-headed attention mechanism that can include local and global heads, rotary or absolute positional embeddings, and dropout.


```python
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out

# a module for keeping track of when to update the projections

class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)

            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented

# classes

class ReZero(nn.Module):
    ... ignore

class PreScaleNorm(nn.Module):
    ... ignore

class PreLayerNorm(nn.Module):
    ... ignore

class Chunk(nn.Module):
    ... ignore

class FeedForward(nn.Module):
    ... ignore

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)

class SelfAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)

class CrossAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context = context, **kwargs)


```

The `partial` function in Python, from the `functools` module, is used to **create a new function with some of its arguments pre-filled (partially applied)**. This is useful when you want to fix certain parameters of a function and leave others open to be specified later. Essentially, it allows you to create a simpler function from an existing one by pre-setting some of the parameters.

### Why Use `partial`?

1. **Simplify Function Calls**: If you frequently call a function with certain parameters fixed, `partial` lets you create a new function with those parameters pre-set, reducing redundancy.
2. **Function Customization**: In complex code, `partial` enables more modular and customizable code by allowing functions to be adapted with specific parameters for different contexts.
3. **Improve Readability**: It makes code more readable by reducing the need for repeated arguments and creating functions that are easier to understand at a glance.

### How `partial` Works

`partial` takes a function as its first argument and then any arguments or keyword arguments you want to pre-set. It returns a new function where those pre-set arguments are "locked in," while the remaining arguments can still be supplied when calling the new function.

Here’s an example:

```python
from functools import partial

def multiply(x, y):
    return x * y

# Create a new function that always multiplies by 2
double = partial(multiply, 2)

# Now you can use double with just one argument
print(double(5))  # Output: 10
```

In this case:

- `partial(multiply, 2)` creates a new function `double` that is equivalent to `multiply(2, y)`.
- Calling `double(5)` is the same as calling `multiply(2, 5)`.

### Example in Context of `FastAttention`

In the `FastAttention` class, `partial` is used to create functions with specific parameters pre-set. For instance:

```python
self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=nb_features, nb_columns=dim_heads, scaling=ortho_scaling)
```
In this example:

- `partial` is used to create `self.create_projection`, a version of `gaussian_orthogonal_random_matrix` with `nb_rows`, `nb_columns`, and `scaling` pre-set.
- `self.create_projection` can now be called with just a `device` argument, simplifying function calls and making the code more modular.


### Fast Attention

原文 FA 算法如下。Bidirectional = Non-causal linear attention.  Unidirectional = Causal linear atttention. 
![[Pasted image 20241113193123.png]]

### 核心 code

- **linear_attention**: Non-causal linear attention for efficient matrix multiplications.
- **causal_linear_attention**: Causal linear attention, preventing future tokens from attending to past ones.
- **causal_linear_attention_noncuda**: Inefficient CPU-based implementation of causal linear attention for reference.  沒有使用，只是用於解釋。

### Non-causal Linear Attention (直接單純)

This function computes **non-causal linear attention** for transformer models. Linear attention mechanisms approximate the standard softmax attention to reduce computational complexity from $O(n^2)$ to $O(n)$, where $n$ is the sequence length.

只有四行 code.

```python
# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out
```

#### **Inputs**

- **`q`**: Query tensor of shape `(..., n, d)`
- **`k`**: Key tensor of shape `(..., n, d)`
- **`v`**: Value tensor of shape `(..., n, e)`

Here, `...` represents any number of batch dimensions, `n` is the sequence length, `d` is the embedding dimension, and `e` is the value dimension.

#### **Outputs**

- **`out`**: Output tensor of shape `(..., n, e)`

#### **Mathematical Interpretation**

This function approximates the standard softmax attention using linear operations:

$$\text{Attention}(Q, K, V) \approx D^{-1}\phi(Q)(\phi(K)^\top V)$$

where $\phi(\cdot)$ is a feature map (e.g., exponential function) that linearizes the softmax kernel.

### Causal Linear Attention (有點燒腦)

正常的 attention 先計算 $QK^{\top}$,  再直接加上 causal mask $M$ (strict upper triangular matrix with element = $-\infty$),  或是 $\text{softmax} (QK^{\top}) \otimes M$.   $\otimes$ 是 element-wise 乘法，此處 $M$ 是 lower triangular matrix with element = +1.    最後再乘 $V$.  

但是 Fast Attention (linear attention) 要先計算 $K^{\top} V$.  如何處理 causal mask 就非常燒腦。原文解釋如下。需要變成 iterative process (index *i*)，其實可以視為另一種 RNN.   但是這個 iterative process 也可以平行計算時間可以 $O(L)\to O(\log L)$，只是要多計算和存儲 (11)。

![[Pasted image 20241113190707.png]]


我們看比較慢 (without cuda code), 使用 iterative method 類似 RNN  implementation 如下。

```python
# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size = 128, eps = 1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
        k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)

        last_k_cumsum = k_cumsum[:, :, -1:]
        last_context_cumsum = context_cumsum[:, :, -1:]
        outs.append(out)

    return torch.cat(outs, dim = -2)

```

#### `causal_linear_attention_noncuda` Function

The `causal_linear_attention_noncuda` function is an implementation of causal linear attention without using CUDA acceleration. Although it's labeled as "inefficient" and "not being used," it serves as a reference implementation to understand how causal linear attention can be computed using cumulative sums. This function processes queries (`q`), keys (`k`), and values (`v`) to compute the attention output in a causal (autoregressive) manner, ensuring that each position in the sequence only attends to previous positions.

Below, we will break down the code step by step, explain each component, and relate it to the underlying mathematical operations.

## Function Signature

```python
def causal_linear_attention_noncuda(q, k, v, chunk_size=128, eps=1e-6):
```

- **Inputs**:
  - `q`: Query tensor of shape `(..., N, D)`.
  - `k`: Key tensor of shape `(..., N, D)`.
  - `v`: Value tensor of shape `(..., N, E)`.
  - `chunk_size`: Size of chunks to process at a time along the sequence dimension (default: 128).
  - `eps`: Small epsilon value to prevent division by zero (default: 1e-6).

- **Outputs**:
  - Returns the attention output tensor of shape `(..., N, E)`.

## Initialization

```python
last_k_cumsum = 0
last_context_cumsum = 0
outs = []
```

- **Purpose**:
  - `last_k_cumsum`: Stores the cumulative sum of keys up to the previous chunk.
  - `last_context_cumsum`: Stores the cumulative sum of the context (key-value products) up to the previous chunk.
  - `outs`: List to collect output chunks.

## Processing Chunks

The function processes the input tensors in chunks along the sequence dimension to manage memory usage and simulate causal computation.

### Chunking the Inputs

```python
for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):
```

- **Explanation**:
  - `t.chunk(chunk_size, dim=-2)`: Splits each tensor (`q`, `k`, `v`) into chunks along the sequence dimension (`dim=-2`).
  - `zip(...)`: Iterates over the corresponding chunks of `q`, `k`, and `v`.

### Iterating Over Chunks

For each chunk, the function performs the following steps:

1. **Cumulative Sum of Keys**

   ```python
   k_cumsum = last_k_cumsum + k.cumsum(dim=-2)
   ```

   - **Explanation**:
     - `k.cumsum(dim=-2)`: Computes the cumulative sum of `k` along the sequence dimension within the current chunk.
     - `last_k_cumsum`: Adds the cumulative sum from the previous chunks to account for all previous positions.
     - **Result**: `k_cumsum` contains the cumulative sum of keys up to the current position.

2. **Computing the Normalization Factor $D_{\text{inv}}$**

   ```python
   D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
   ```

   - **Explanation**:
     - `torch.einsum('...nd,...nd->...n', q, k_cumsum + eps)`: Computes the dot product between `q` and `k_cumsum` along the feature dimension (`D`), resulting in a tensor of shape `(..., N)`.
     - `1. / (...)`: Takes the reciprocal to obtain $D_{\text{inv}}$, the inverse of the normalization factor.
     - `eps`: A small value added for numerical stability to prevent division by zero.
     - **Mathematical Representation**:
       $$
       D_{\text{inv, t}} = \frac{1}{\sum_{i=1}^{t} q_t^\top k_i + \epsilon} = \frac{1}{q_t^\top\sum_{i=1}^{t}  k_i + \epsilon}
       $$
       where $t$ indexes the positions in the sequence.

3. **Computing the Context Tensor**

   ```python
   context = torch.einsum('...nd,...ne->...nde', k, v)
   ```

   - **Explanation**:
     - `torch.einsum('...nd,...ne->...nde', k, v)`: Computes the outer product between `k` and `v` for each position, resulting in a tensor of shape `(..., N, D, E)`.
     - **Result**: `context` contains the key-value products for the current chunk.

4. **Cumulative Sum of Context**

   ```python
   context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
   ```

   - **Explanation**:
     - `context.cumsum(dim=-3)`: Computes the cumulative sum of the context along the sequence dimension (now at `dim=-3` due to the additional dimension from the outer product).
     - `last_context_cumsum`: Adds the cumulative sum from the previous chunks.
     - **Result**: `context_cumsum` contains the cumulative sum of key-value products up to the current position.

5. **Computing the Output**

   ```python
   out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)
   ```

   - **Explanation**:
     - `torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)`: Performs a tensor contraction that:
       - Multiplies `context_cumsum` with `q` along the `D` dimension.
       - Scales the result by `D_inv`.
     - **Result**: `out` is the attention output for the current chunk, of shape `(..., N, E)`.

   - **Mathematical Representation**:
     $$
     \text{out}_t = D_{\text{inv}, t} \left( \sum_{i=1}^{t} \left( q_t^\top k_i \right) v_i \right) = D_{\text{inv}, t} \left( q_t^{\top} \sum_{i=1}^{t}   k_i  v_i \right)
     $$

6. **Updating Cumulative Sums**

   ```python
   last_k_cumsum = k_cumsum[..., -1:, :]
   last_context_cumsum = context_cumsum[..., -1:, :, :]
   ```

   - **Explanation**:
     - Extracts the last cumulative sum from the current chunk to carry over to the next chunk.
     - Slices the tensors to keep the last position (`-1:`) along the sequence dimension.

7. **Collecting the Output**

   ```python
   outs.append(out)
   ```

   - **Explanation**:
     - Appends the output `out` of the current chunk to the list `outs`.

## Concatenating the Outputs

After processing all chunks, the outputs are concatenated along the sequence dimension:

```python
return torch.cat(outs, dim=-2)
```

- **Explanation**:
  - `torch.cat(outs, dim=-2)`: Concatenates the list of output tensors along the sequence dimension to form the final output.

## Mathematical Understanding

### Causal Linear Attention

The causal linear attention mechanism aims to compute the attention output efficiently by leveraging the associativity of matrix multiplication and avoiding explicit computation of the attention matrix, which is quadratic in sequence length.

The standard attention mechanism is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}(Q K^\top) V
$$

In linear attention, we approximate the softmax function with a feature mapping $\phi(x)$such that:

$$
\text{Attention}(Q, K, V) \approx \left( \phi(Q) \left( \phi(K)^\top V \right) \right)
$$

For causal attention, we need to ensure that each position $t$only attends to positions $\leq t $.

### Implementing Causality with Cumulative Sums

By using cumulative sums, we can efficiently compute the causal attention outputs:

- **Cumulative Sum of Keys**:
  - $S_t = \sum_{i=1}^{t} k_i$
- **Cumulative Sum of Context**:
  - $Z_t = \sum_{i=1}^{t} k_i v_i^\top$
- **Normalization Factor**:
  - $D_t = q_t^\top S_t + \epsilon$
- **Attention Output**:
  - $\text{out}_t = \frac{1}{D_t} q_t^\top Z_t$

This ensures that at each position $t$, the output depends only on keys and values from positions $\leq t$.

## Relating Code to Mathematical Equations

1. **Computing $S_t$ (Cumulative Sum of Keys)**:

   ```python
   k_cumsum = last_k_cumsum + k.cumsum(dim=-2)
   ```

   - Corresponds to:
     $$
     S_t = S_{t-1} + k_t
     $$

2. **Computing $Z_t$ (Cumulative Sum of Context)**:

   ```python
   context = torch.einsum('...nd,...ne->...nde', k, v)
   context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
   ```

   - Corresponds to:
     $$
     Z_t = Z_{t-1} + k_t v_t^\top
     $$

3. **Computing $D_t$ (Normalization Factor)**:

   ```python
   D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
   ```

   - Corresponds to:
     $$
     D_t = q_t^\top S_t + \epsilon
     $$

4. **Computing the Output $\text{out}_t$**:

   ```python
   out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)
   ```

   - Corresponds to:
     $$
     \text{out}_t = \frac{1}{D_t} q_t^\top Z_t
     $$

## Recap

Since we don't have the actual figure, let's conceptually understand how the cumulative sums implement causal attention.

- **Sequence Processing**:
  - The sequence is processed in chunks to simulate streaming data or to handle long sequences without exceeding memory limits.
- **Causality Enforcement**:
  - By using cumulative sums, each position's output only depends on current and past positions.
  - No information from future positions is included in the computation.
- **Chunking Mechanism**:
  - The cumulative sums (`k_cumsum` and `context_cumsum`) are updated incrementally.
  - The last cumulative sums from the previous chunk are used to initialize the current chunk's cumulative sums.
- **Attention Computation**:
  - The attention output at each position is computed using the cumulative sums up to that position.
  - The normalization factor ensures proper scaling of the outputs.

## Practical Implications

- **Inefficiency**:
  - The code is labeled as "inefficient" because it does not leverage optimized GPU operations or memory-efficient algorithms.
  - It serves as a reference implementation to illustrate the concept.
- **Not Using CUDA**:
  - The absence of CUDA-specific code means that it may not run efficiently on GPUs.
  - In practice, optimized implementations would use custom CUDA kernels or libraries designed for performance.

## Summary

The `causal_linear_attention_noncuda` function:

- Implements causal linear attention by computing cumulative sums of keys and context.
- Ensures that each output position only depends on current and past positions (causality).
- Processes the sequence in chunks to manage memory usage.
- Uses Einstein summation (`torch.einsum`) for tensor contractions.
- Is an illustrative implementation for understanding causal linear attention, albeit not optimized for performance.


CUDA 有一個優化的版本如下。不知道 `causal_dot_product_fn` 是如何實現的。
```python
# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps = 1e-6):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))

        out = causal_dot_product_fn(q, k, v)

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out

```

## Kernel Functions:  ORF (Orthogonal Random Feature +)

These are optimized implementations of softmax and generalized kernels:

- **softmax_kernel**: Computes a softmax-based kernel using projections. Used to replace standard softmax attention.
- **generalized_kernel**: Implements a general kernel function, defaulting to `ReLU`, with optional projection.
- **orthogonal_matrix_chunk & gaussian_orthogonal_random_matrix**: Create random matrices with orthogonal constraints, useful for efficient kernel approximations.

QR decomposition 產生正交的 basis.

```python
# kernel functions

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# linear attention classes with softmax kernel

```


## Key Points

- **Linear Attention**: Replaces quadratic complexity attention with linear complexity attention for faster and memory-efficient computations.
- **Projection Updater**: Manages periodic redraws of projection matrices to improve performance without recalculating projections every step.
- **Flexible Configuration**: Various options for normalization, token shifting, cross-attention, and dropout make it highly customizable.

### Performer Class

The main **Performer** class is an efficient variant of the transformer. Key features:

- Supports multiple heads and local attention windows.
- **Reversible** layers allow for memory-efficient training.  這是 Feedforward layer, 最早從 reformer model 引入。
- **Feature redraw interval**: Redraws projection matrices periodically.
- **ProjectionUpdater** is used to manage feature projection updates automatically.

### PerformerLM Class

This is the full transformer language model using the Performer attention mechanism. Key components:

- **Embedding layers**: For tokens and positional encodings.
- **Performer**: Uses the efficient Performer attention mechanism for self-attention.
- **Layer normalization** and **linear output layer**: Applied to the final output.


```python
# performer

class Performer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        no_projection = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):

            attn = SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(CrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)

class PerformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        dim,
        depth,
        heads,
        dim_head = 64,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,
        rotary_position_emb = True,
        axial_position_emb = False,
        axial_position_shape = None,
        auto_check_redraw = True,
        qkv_bias = False,
        attn_out_bias = False,
        shift_tokens = False
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(dim, depth, heads, dim_head, local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, no_projection, auto_check_redraw, qkv_bias, attn_out_bias, shift_tokens)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embeddings
        x = self.token_emb(x)
        x += self.pos_emb(x)

        x = self.dropout(x)

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb = layer_pos_emb, **kwargs)

        # norm and to logits
        x = self.norm(x)

        if return_encodings:
            return x

        if exists(self.to_out):
            return self.to_out(x)

        return x @ self.token_emb.weight.t()

```


The main differences between **PerformerLM** and **Performer** in the code are related to their specific roles within the transformer architecture. Here’s a breakdown of each:

### 1. **Performer**
The `Performer` class implements the core attention mechanism and feedforward layers of the Performer architecture. It's the main building block that defines how tokens interact with each other through **attention** and **nonlinear transformations**. Key points about `Performer`:

- **Attention Layers**: uses `FastAttention` can be set to either causal or non-causal mode
- **Local Attention Support**: `Performer` includes support for both local and global attention heads. This allows for a mixture of local attention (attention to a limited neighborhood) and global attention (full context).
- **Feedforward Layers**: chunked for memory efficiency.  Support **Reversible Sequence Support** allowing for memory savings during training by not storing intermediate activations.

### 2. **PerformerLM**
The `PerformerLM` class is a **language model** that builds on top of `Performer`. It wraps the `Performer` class and adds elements specific to language modeling tasks. Key points about `PerformerLM`:

- **Token Embeddings**
- **Positional Embeddings**: Adds positional information to the tokens to maintain the sequence order. It supports multiple types of positional embeddings, such as fixed sinusoidal, rotary, or absolute positional embeddings.
- **Output Layer**: After processing through the `Performer` layers, `PerformerLM` includes a final layer to map the transformed embeddings back to vocabulary logits for language modeling tasks. If `tie_embed` is set to `True`, it ties the input and output embeddings for parameter efficiency.
- **Language Model-Specific Configurations**: `PerformerLM` includes specific configurations for language modeling, like the ability to handle maximum sequence lengths and efficient generation through causal attention.



