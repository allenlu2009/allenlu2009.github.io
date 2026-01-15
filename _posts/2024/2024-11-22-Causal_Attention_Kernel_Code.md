---
title: Causal Attention Kernel Code
date: 2024-11-22 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-1117-Causal_Attention_Kernel]]"
categories:
  - GenAI
tags:
  - Attention
  - Causal
  - Cumsum
  - Kernel
  - LLM
  - Transformer
---



## Source

Excellent paper: GAU (Gated Attention Unit) and local (transformer-like)+global attention (RNN):    https://arxiv.org/pdf/2202.10447  --> **check with cat(local, global attn) instead of (local+ global attn)**

Gated linear attention + Chunk!  Songlin Yang:  https://arxiv.org/pdf/2312.06635


![[Pasted image 20241120004206.png]]



## Flash Attention Code

### Layers
https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/layers/gla.py

```python
class GatedLinearAttention(nn.Module):
    r"""
    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_kv_heads (int, Optional):
            The number of key/value heads, used for MQA. Default: None.
        feature_map (str, Optional):
            Feature map function applied to queries/keys. Default: None.
		...
    """

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False
            )
```



## 先看 chunk.py,  是正常 attention 順序: (QK)V; 但是 (1) 切 spatial and temporal chunks!  (2) 每個 chunk 做 gate!! 
Code: https://github.com/sustcsonglin/flash-linear-attention/tree/main/fla/ops/gla

# Mathematical Explanation of `chunk.py`

The `chunk.py` module implements a **Gated Linear Attention (GLA)** mechanism optimized for GPU execution using [Triton](https://triton-lang.org/). 

## Table of Contents

1. [Overview](#overview)
2. [Definitions and Notations](#definitions-and-notations)
3. [Forward Pass](#forward-pass)
    - [Forward Kernel: `chunk_gla_fwd_A_kernel_intra_sub_inter`](#forward-kernel-chunk_gla_fwd_A_kernel_intra_sub_inter)
    - [Forward Kernel: `chunk_gla_fwd_A_kernel_intra_sub_intra`](#forward-kernel-chunk_gla_fwd_A_kernel_intra_sub_intra)
    - [Forward Kernel: `chunk_gla_fwd_kernel_o`](#forward-kernel-chunk_gla_fwd_kernel_o)
    - [Orchestrating Forward Pass: `chunk_gla_fwd`](#orchestrating-forward-pass-chunk_gla_fwd)
    - [Final Output Computation: `chunk_gla_fwd_o_gk`](#final-output-computation-chunk_gla_fwd_o_gk)
6. [User-Facing Function: `chunk_gla`](#user-facing-function-chunk_gla)
7. [Conclusion](#conclusion)

---

## Overview

The `chunk.py` module defines a series of **Triton kernels** and orchestrating functions to perform the forward and backward passes of the Gated Linear Attention mechanism. The implementation leverages **chunking** to handle large sequences efficiently, dividing them into smaller blocks that fit into GPU memory constraints. This approach enhances performance by maximizing parallelism and minimizing memory bottlenecks.

## Definitions and Notations

Before diving into the code, let's define the key dimensions and tensors involved in the GLA mechanism:

- **Dimensions**:
  - $B$: Batch size
  - $H$: Number of attention heads
  - $T$: Sequence length (number of time steps)
  - $K$: Dimension of query and key vectors
  - $V$: Dimension of value vectors
  - $BT$: Block size for time dimension
  - $BC$: Block size for chunks within a block
  - $BK$: Block size for the key dimension
  - $BV$: Block size for the value dimension
  - $NC$: Number of chunks per block
  - $NT$: Number of time blocks
  - $NK$: Number of key blocks

- **Tensors**:
  - **Queries**: $\mathbf{Q} \in \mathbb{R}^{B \times H \times T \times K}$ or $\mathbf{Q} \in \mathbb{R}^{B \times T \times H \times K}$ depending on `head_first`.
  - **Keys**: $\mathbf{K} \in \mathbb{R}^{B \times H \times T \times K}$ or $\mathbf{K} \in \mathbb{R}^{B \times T \times H \times K}$.
  - **Values**: $\mathbf{V} \in \mathbb{R}^{B \times H \times T \times V}$ or $\mathbf{V} \in \mathbb{R}^{B \times T \times H \times V}$.
  - **Gate Keys**: $\mathbf{G} \in \mathbb{R}^{B \times H \times T \times K}$ or $\mathbf{G} \in \mathbb{R}^{B \times T \times H \times K}$.
  - **Attention Matrix**: $\mathbf{A} \in \mathbb{R}^{B \times H \times T \times BT}$ or $\mathbf{A} \in \mathbb{R}^{B \times T \times H \times BT}$.
  - **Hidden State**: $\mathbf{h}$, various shapes depending on context.
  - **Outputs**: $\mathbf{O} \in \mathbb{R}^{B \times H \times T \times V}$ or $\mathbf{O} \in \mathbb{R}^{B \times T \times H \times V}$.

- **Scaling Factor**:
 $$
  \text{scale} = \frac{1}{\sqrt{K}}
 $$

## Forward Pass

The forward pass computes the attention outputs and updates the hidden state based on the queries, keys, values, and gate keys. It involves several Triton kernels optimized for different chunking strategies.

### Forward Kernel: `chunk_gla_fwd_A_kernel_intra_sub_inter`

The code you provided is a Triton kernel implementation of a **forward pass for gated linear attention** in a chunk-wise manner, with both **intra-chunk** and **inter-chunk** computations involved. Below, I'll explain it mathematically in terms of the matrix operations being carried out, along with the relevant formulas.

### Mathematical Breakdown of the Code

The forward kernel implementation here computes **attention weights** and applies them to **queries** and **keys** using gated linear attention (GLA) principles. Specifically, this is a **two-pass approach**: it computes **intra-chunk** and **inter-chunk** attention contributions in a fused manner.

#### **Kernel Variables and Parameters:**

- **Input Variables:**
  - $Q \in \mathbb{R}^{B \times H \times T \times K}$: The query matrix.
  - $K \in \mathbb{R}^{B \times H \times T \times K}$: The key matrix.
  - $G \in \mathbb{R}^{B \times H \times T \times K}$: Gating factors applied to the keys/queries.
  - $A \in \mathbb{R}^{B \times H \times T \times T}$: Output accumulated matrix (i.e., the "attention" scores).
  - **Scale** ($\text{scale}$): Scaling factor, typically $\frac{1}{\sqrt{K}}$, where $K$ is the dimension of keys/queries.

- **Chunk Parameters:**
  - $T$: Length of the sequence.
  - $H$: Number of heads.
  - $K$: Dimension of each key/query.
  - $BT$, $BC$, $BK$: Block size in different dimensions.
  - $NC$: Number of chunks.
  - **HEAD_FIRST**: Indicates whether the head dimension is processed first.

### **Steps in the Code and Corresponding Mathematical Representation**

#### **1. Intra-Chunk and Inter-Chunk Processing**
The kernel processes intra-chunk and inter-chunk contributions in a loop.

- **Intra-chunk contribution**: Within a given chunk, attention weights are computed for all time steps in that chunk.
- **Inter-chunk contribution**: Attention is computed between different chunks, ensuring cross-chunk dependencies are handled.

The kernel iterates over a chunk of size $BC$, and for each chunk, it computes the pairwise interactions between queries and keys.

#### **2. Gating Mechanism**
In GLA, a **gating mechanism** is introduced to modulate the contributions of different time steps. This is done via **gating factors** $G$.

The gating factor $G$ for the $i$-th chunk modifies the queries and keys as follows:

$$
Q_g = Q_i \cdot \exp(G_i - G_{\text{norm}})
$$

Where:
- $Q_g$ is the gated query.
- $G_{\text{norm}}$ is a normalization term (loaded in `b_gn`).

Similarly, for the keys:

$$
K_g = K_j \cdot \exp(G_{\text{norm}} - G_j)
$$

Where:
- $K_g$ is the gated key.

The gating mechanism ensures that the contribution of each chunk is weighted appropriately, potentially to improve gradient flow and stability during training.

#### **3. Attention Weight Calculation**
Once the gated queries and keys are computed, the **attention weights** are calculated via a dot product:

$$
A_{ij} = Q_g \cdot K_g^\top
$$

This dot product is computed in blocks to avoid excessive memory usage. The kernel stores these results in a **block-wise matrix** $A$.

The code uses **tensor cores** (TF32) to improve precision for these operations, which might otherwise suffer from numerical instability when dealing with large sequences.

#### **4. Accumulating Attention Weights**
The **attention score** for the chunk, $A$, is accumulated:

$$
A_{ij} += Q_g \cdot K_g^\top
$$

This means that for each chunk, all attention contributions are added to form the final matrix of scores for each sequence.

### **Mathematical Summary**

The overall goal of the kernel is to compute:

$$
A = \sum_{i=1}^{N} \sum_{j=1}^{N} Q_g^{(i)} K_g^{(j) \top}
$$

Where $N$ is the number of chunks, and:

1. **Gated Query**:
   $$
   Q_g^{(i)} = Q^{(i)} \cdot \exp(G^{(i)} - G_{\text{norm}})
   $$

2. **Gated Key**:
   $$
   K_g^{(j)} = K^{(j)} \cdot \exp(G_{\text{norm}} - G^{(j)})
   $$

3. **Attention Scores**:
   $$
   A_{ij} = Q_g^{(i)} \cdot K_g^{(j) \top}
   $$

These attention scores are computed for **both intra-chunk and inter-chunk** interactions, allowing the model to capture both **local dependencies** (within chunks) and **global dependencies** (between chunks).

### **Key Differences from Standard Attention**

- **Linear Attention:**  No softmax. 
- **Chunking**: Instead of processing the entire sequence at once, the kernel processes it in chunks of size $BC$.
- **Gated Linear Attention**: Introduces gating to scale the contributions of keys/queries, potentially improving efficiency and stability.
- **Precision Control**: Uses TF32 to balance precision and performance, especially for large-scale tensor operations.

### Forward Kernel: `chunk_gla_fwd_A_kernel_intra_sub_intra`

The code you've provided implements a **forward kernel for intra-chunk gated linear attention (GLA)**, where intra-chunk means the attention calculation is restricted within a chunk of data rather than involving inter-chunk interactions. Below is a mathematical breakdown and representation of what this kernel does.

### **Mathematical Representation and Explanation**

#### **Inputs and Parameters:**

- **Inputs:**
  - $Q \in \mathbb{R}^{B \times H \times T \times K}$: The query matrix.
  - $K \in \mathbb{R}^{B \times H \times T \times K}$: The key matrix.
  - $G \in \mathbb{R}^{B \times H \times T \times K}$: Gating factors used to modulate keys/queries.
  - $A \in \mathbb{R}^{B \times H \times T \times T}$: Accumulated output attention scores.
  - **Scale** ($\text{scale}$): Typically $\frac{1}{\sqrt{K}}$, where $K$ is the dimension of keys/queries, used for stabilizing gradients.

- **Parameters:**
  - $T$: Sequence length.
  - $H$: Number of attention heads.
  - $K$: Dimension of each key/query.
  - $BT$, $BC$, $BK$: Block sizes in various dimensions (time, chunk, key dimension).
  - **HEAD_FIRST**: Flag indicating if head dimension is processed first.

### **Steps in the Kernel and Corresponding Mathematics**

#### **1. Intra-Chunk Computation:**
The kernel is responsible for computing **intra-chunk attention** within chunks of size $BC$ for queries and keys, with **intra-chunk gating** applied.

For simplicity, consider that we have queries $Q$, keys $K$, and gating factors $G$, and the forward kernel computes:

$$
A_{ij} = Q_i \cdot K_j^\top \cdot e^{G_i - G_j}
$$

#### **2. Gating Mechanism in Intra-Chunk Attention**
In **gated linear attention**, a gating factor is applied to modify the attention weights dynamically.

- The gating factor for **queries** and **keys** influences the contribution each part of the sequence makes to the overall attention.

Mathematically, the **gated queries** and **keys** can be expressed as:

$$
Q_g^{(i)} = Q^{(i)} \cdot e^{G^{(i)}}
$$

$$
K_g^{(j)} = K^{(j)} \cdot e^{-G^{(j)}}
$$

Where:
- $Q_g^{(i)}$ is the gated version of the $i$-th chunk of queries.
- $K_g^{(j)}$ is the gated version of the $j$-th chunk of keys.
- $G^{(i)}$ and $G^{(j)}$ are the gating factors for each chunk.

The **gated version of the attention score** between the $i$-th and $j$-th chunk, within the same chunk, is given by:

$$
A_{ij} = Q_g^{(i)} \cdot K_g^{(j) \top} = (Q^{(i)} \cdot e^{G^{(i)}}) \cdot (K^{(j)} \cdot e^{-G^{(j)}})^\top
$$

$$
A_{ij} = Q^{(i)} K^{(j) \top} \cdot e^{G^{(i)} - G^{(j)}}
$$

This formula shows that the gating factor helps in **scaling** the interactions between different elements of the sequence.

#### **3. Block-Wise Computation**
The kernel computes **block-wise dot products** to improve memory efficiency.

For each chunk:

- Load a **block of queries** ($Q$).
- Load the corresponding **gating factors** ($G$).
- Perform element-wise multiplication and **scaling** based on gating factors.
  
The **attention weight** between two positions $i$ and $j$ is computed as:

$$
A_{ij} = \sum_{k=0}^{K-1} Q^{(i)}_k \cdot K^{(j)}_k \cdot \exp(G^{(i)} - G^{(j)})
$$

Where:
- $Q^{(i)}_k$ and $K^{(j)}_k$ represent the $k$-th component of the $i$-th query vector and the $j$-th key vector, respectively.

#### **4. Accumulating Results in $A$**
- The computed **attention score** ($b_A$) is accumulated and stored into the corresponding position in matrix $A$.
  
The intra-chunk attention accumulation follows:

$$
A[i, j] = A[i, j] + b_A
$$

Where $b_A$ represents the attention scores computed in the block-wise manner.

#### **5. Scaling Factor**
The kernel also uses a **scaling factor** ($\text{scale}$):

$$
A_{ij} = A_{ij} \cdot \text{scale}
$$

This scaling factor, typically $\frac{1}{\sqrt{K}}$, is used to prevent the gradients from becoming too large, thus improving the stability of training.

### **Mathematical Summary of the Kernel's Operation**

The kernel computes **attention scores** using a gated linear approach with intra-chunk restrictions. The key steps can be represented mathematically as:

1. **Gating Application**:
   $$
   Q_g^{(i)} = Q^{(i)} \cdot e^{G^{(i)}}, \quad K_g^{(j)} = K^{(j)} \cdot e^{-G^{(j)}}
   $$

2. **Attention Weight Calculation**:
   $$
   A_{ij} = Q_g^{(i)} \cdot K_g^{(j) \top} = Q^{(i)} K^{(j) \top} \cdot e^{G^{(i)} - G^{(j)}}
   $$

3. **Scaling**:
   $$
   A_{ij} = A_{ij} \cdot \text{scale}
   $$

4. **Accumulate and Store**:
   The resulting attention values are accumulated in the output matrix $A$.

This approach is **memory-efficient** and works well for **long sequences**, where the computation is broken into smaller chunks, and only intra-chunk interactions are processed by this specific kernel. The gating mechanism helps **modulate** the contributions dynamically, enhancing model performance by incorporating gating information.

### Forward Kernel: `chunk_gla_fwd_kernel_o`

This kernel computes the final output $\mathbf{O}$ by combining the attention matrix $\mathbf{A}$ with the values $\mathbf{V}$.

**Mathematical Operations**:

1. **Attention-Value Multiplication**:
  $$
   \mathbf{O} = \mathbf{A} \cdot \mathbf{V}
  $$

2. **Masking**:
   A mask $\mathbf{M}$ ensures causality or other constraints:
  $$
   \mathbf{O}_{t,v} = \sum_{k} \mathbf{A}_{t,k} \times \mathbf{V}_{k,v} \times \mathbf{M}_{t,k}
  $$

### Orchestrating Forward Pass: `chunk_gla_fwd`

The `chunk_gla_fwd` function orchestrates the forward pass by:

1. **Chunking the Gate Keys**:
  $$
   \mathbf{G}_{\text{cumsum}} = \text{cumsum}(\mathbf{G}, \text{chunked by } BT)
  $$

2. **Computing Hidden States**:
  $$
   \mathbf{h}, \mathbf{h}_t = \text{chunk\_fwd\_h}(\mathbf{K}, \mathbf{V}, \mathbf{G}_{\text{cumsum}}, \mathbf{h}_0)
  $$

3. **Computing Attention Matrix**:
  $$
   \mathbf{A} = \text{chunk\_gla\_fwd\_intra\_gk}(\mathbf{Q}, \mathbf{K}, \mathbf{G}_{\text{cumsum}}, \text{scale})
  $$

4. **Computing Outputs**:
  $$
   \mathbf{O} = \text{chunk\_gla\_fwd\_o\_gk}(\mathbf{Q}, \mathbf{V}, \mathbf{G}_{\text{cumsum}}, \mathbf{A}, \mathbf{h}, \text{scale})
  $$

### Final Output Computation: `chunk_gla_fwd_o_gk`

This function finalizes the computation of the output tensor $\mathbf{O}$ by performing the matrix multiplication between the attention matrix $\mathbf{A}$ and the value tensor $\mathbf{V}$, scaled appropriately.

**Mathematical Operations**:

$$
\mathbf{O} = \mathbf{A} \times \mathbf{V} \times \text{scale}
$$

This operation aggregates the weighted values based on the attention scores, producing the final output.

## User-Facing Function: `chunk_gla`

The `chunk_gla` function serves as the primary interface for users to perform GLA operations. It seamlessly integrates the forward and backward passes, ensuring that gradients are correctly propagated during training.

### Function Signature

```python
def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
```

### Arguments

- **q** ($\mathbf{Q}$): Queries tensor of shape `[B, H, T, K]` if `head_first=True`, else `[B, T, H, K]`.
- **k** ($\mathbf{K}$): Keys tensor of shape `[B, H, T, K]` if `head_first=True`, else `[B, T, H, K]`.
- **v** ($\mathbf{V}$): Values tensor of shape `[B, H, T, V]` if `head_first=True`, else `[B, T, H, V]`.
- **g** ($\mathbf{G}$): Gate keys tensor of shape `[B, H, T, K]` if `head_first=True`, else `[B, T, H, K]`.
- **scale**: Scaling factor for attention scores. Defaults to $\frac{1}{\sqrt{K}}$ if not provided.
- **initial_state**: Initial hidden state tensor of shape `[B, H, K, V]`. Defaults to zero if not provided.
- **output_final_state**: Boolean flag to output the final hidden state. Defaults to `False`.
- **head_first**: Boolean flag indicating the format of input tensors. Defaults to `True`.

### Returns

- **o** ($\mathbf{O}$): Output tensor of shape `[B, H, T, V]` if `head_first=True`, else `[B, T, H, V]`.
- **final_state** ($\mathbf{h}_t$): Final hidden state tensor of shape `[B, H, K, V]` if `output_final_state=True`, else `None`.

### Mathematical Operations

1. **Scaling Factor**:
  $$
   \text{scale} = 
   \begin{cases}
     \frac{1}{\sqrt{K}}, & \text{if } \text{scale is None} \\
     \text{scale}, & \text{otherwise}
   \end{cases}
  $$

2. **Forward Pass**:
  $$
   \mathbf{O}, \mathbf{h}_t = \text{ChunkGLAFunction.apply}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{G}, \text{scale}, \mathbf{h}_0, \text{output\_final\_state}, \text{head\_first})
  $$

3. **Return**:
   The function returns the output $\mathbf{O}$ and the final hidden state $\mathbf{h}_t$ based on the `output_final_state` flag.


## 再來是 fused_chunk.py

#### Gated Linear Attention
The Gated Linear Attention mechanism allows the computation of attention with a **gating mechanism** (`g`) applied to the key (`k`) and value (`v`) pairs.

Mathematically:
- Queries ($q$), Keys ($k$), and Values ($v$) are processed as:
  $$
  A = q \cdot k^T
  $$
  where $A$ is the attention score matrix.

- Gated Linear Attention involves modifying these values using a **gate** (`g`) to control the impact of the keys and values.

### Forward Propagation (`forward`)
The `forward` method implements the fused GLA in the following steps:

1. **Initialization and Setup**:
   - Determine tensor shapes (`B, H, T, K, V`), which refer to Batch size, number of Heads, sequence Length, Key dimension, and Value dimension respectively.
   - Define the **block sizes** (`BT`, `BK`, `BV`) for chunking the sequence, key, and value dimensions.
   - Use `chunk_local_cumsum` to compute the **cumulative decay** for the gate (`g`) along chunks.
   - Allocate memory for intermediate outputs (`o`, `q_g`, `k_g`, `A`).

2. **Prepare Intermediate Gates**:
   - **prepare_qg_kg** is called to compute intermediate gated queries and keys:
     $$
     q_g = q \times g
     $$
     $$
     k_g = k \times g
     $$
   - Here, `g` serves as a gating factor to modify the key and query contributions based on context.

3. **Fused GLA Computation (`fused_chunk_gla_fwd_kernel`)**:
   - Uses Triton to run the forward computation in a highly parallelized manner.
   - The kernel performs the chunk-wise gated attention:
     - Compute the **decayed contribution** from the previous time steps using `g`.
     - Compute the new **hidden state** as a weighted sum of the gated queries, keys, and values.

   Mathematically, for each chunk:
   $$
   o_t = q_t \cdot \left( h_t \times \exp(g_t) \right)
   $$
   where:
   - $h_t$ is the hidden state.
   - $g_t$ controls the decay or influence of previous chunks.

4. **Compute Intra-Chunk Contributions (`fwd_inner_chunk`)**:
   - `fwd_inner_chunk` is used to calculate intra-chunk attention scores.
   - This is done by calculating attention weights for individual segments within a chunk, making the method highly scalable for large sequences.
   
   Mathematically:
   $$
   A = Q \cdot K^T
   $$

5. **Aggregate Results**:
   - **Combine intra-chunk** (`o2`) and **inter-chunk** (`o`) contributions:
     $$
     o = o + o2
     $$

6. **Save for Backward**:
   - Save relevant tensors and information (`q, k, v, g`) for use during the backward pass.


4. **Cumulative Summation for Gradients**:
   - The **reverse cumulative summation** is used to account for the sequential dependency of the gated contributions:
     $$
     dg_{\text{rev}} = \text{reverse\_cumsum\_exclusive}(dg)
     $$
   - Add `dg` and its cumulative version to fully propagate the influence of the gating factor across chunks.

### Efficiency: Why Fused?
- The term **fused** means that multiple tensor operations (like **dot products**, **element-wise multiplications**, and **gating mechanisms**) are **combined into a single GPU kernel**.
- This approach reduces **memory overhead** and increases computational efficiency by:
  - Minimizing the **number of times data is loaded and stored** from/to memory.
  - **Reducing kernel launch overhead**, which is especially important for GPUs.
  - Utilizing **fast on-chip memory** for intermediate computations, allowing for faster operations compared to global memory.


This fused approach is useful in scenarios where the **sequence length** (`T`) is large and directly computing attention would be computationally prohibitive, making it particularly suited for **Transformer-based architectures**.


### Naive.py Implementation:  就是 RNN-like implementation, not in the above layers
https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/ops/gla/naive.py

```python
# -*- coding: utf-8 -*-
def ceildiv(a, b):
    return -(a // -b)


def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False
):
    dtype = q.dtype
    q, k, v, gk = map(lambda x: x.float(), (q, k, v, gk))
    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    scale = K ** -0.5

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        h += initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o[:, :, i] = (q_i[..., None] * h).sum(-2)

    if not output_final_state:
        h = None
    return o.to(dtype), h

```


# Mathematical Explanation of the `naive_recurrent_gla` Function

The `naive_recurrent_gla` function implements a gated recurrent mechanism that processes sequences of queries, keys, values, and gate keys to produce an output tensor over time. This explanation translates the code into precise mathematical formulations using LaTeX, which are rendered via MathJax for clarity.

## Definitions

### Dimensions

- $B$: **Batch size**
- $H$: **Number of attention heads**
- $T$: **Sequence length** (number of time steps)
- $K$: **Dimension of query and key vectors**
- $V$: **Dimension of value vectors**

### Input Tensors

- **Queries**: $\mathbf{Q} \in \mathbb{R}^{B \times H \times T \times K}$
- **Keys**: $\mathbf{K} \in \mathbb{R}^{B \times H \times T \times K}$
- **Values**: $\mathbf{V} \in \mathbb{R}^{B \times H \times T \times V}$
- **Gate Keys**: $\mathbf{G} \in \mathbb{R}^{B \times H \times T \times K}$

### Initial State (Optional)

- **Hidden State**: $\mathbf{h}^{(0)} \in \mathbb{R}^{B \times H \times K \times V}$ (if provided)

### Output Tensors

- **Output**: $\mathbf{O} \in \mathbb{R}^{B \times H \times T \times V}$
- **Final Hidden State**: $\mathbf{h}^{(T)} \in \mathbb{R}^{B \times H \times K \times V}$ (if `output_final_state` is `True`)

## Computation Steps

### 1. Initialization

- **Scaling Factor**:
  $$
  \text{scale} = \frac{1}{\sqrt{K}}
  $$

- **Initialize Hidden State**:
  $$
  \mathbf{h}^{(0)} = 
  \begin{cases}
    \text{initial\_state}, & \text{if provided} \\
    \mathbf{0}, & \text{otherwise}
  \end{cases}
  $$
  where $\mathbf{h}^{(0)} \in \mathbb{R}^{B \times H \times K \times V}$.

- **Initialize Output Tensor**:
  $$
  \mathbf{O} = \mathbf{0} \in \mathbb{R}^{B \times H \times T \times V}
  $$

### 2. Iterative Computation Over Time Steps

For each time step $t = 0$ to $T-1$, perform the following operations:

#### a. Extract Slices

- **Scaled Queries**:
  $$
  \mathbf{q}_t = \text{scale} \times \mathbf{Q}[:, :, t, :] \in \mathbb{R}^{B \times H \times K}
  $$

- **Keys**:
  $$
  \mathbf{k}_t = \mathbf{K}[:, :, t, :] \in \mathbb{R}^{B \times H \times K}
  $$

- **Values**:
  $$
  \mathbf{v}_t = \mathbf{V}[:, :, t, :] \in \mathbb{R}^{B \times H \times V}
  $$

- **Gate Keys (Exponentiated)**:
  $$
  \boldsymbol{\gamma}_t = \exp\left(\mathbf{G}[:, :, t, :]\right) \in \mathbb{R}^{B \times H \times K}
  $$

#### b. Compute Outer Product of Keys and Values

$$
\mathbf{kv}_t = \mathbf{k}_t \otimes \mathbf{v}_t \in \mathbb{R}^{B \times H \times K \times V}
$$

**Element-wise Representation**:
$$
\mathbf{kv}_t[b, h, k, v] = \mathbf{k}_t[b, h, k] \times \mathbf{v}_t[b, h, v]
$$

#### c. Update Hidden State

$$
\mathbf{h}^{(t+1)} = \mathbf{h}^{(t)} \odot \boldsymbol{\gamma}_t[..., \text{None}] + \mathbf{kv}_t
$$

**Component-wise**:
$$
\mathbf{h}^{(t+1)}[b, h, k, v] = \mathbf{h}^{(t)}[b, h, k, v] \times \boldsymbol{\gamma}_t[b, h, k] + \mathbf{kv}_t[b, h, k, v]
$$

- **Explanation**:
  - $\odot$ denotes element-wise multiplication.
  - $\boldsymbol{\gamma}_t[..., \text{None}]$ adds an extra dimension to $\boldsymbol{\gamma}_t$ for broadcasting purposes, matching the shape of $\mathbf{h}^{(t)}$.

#### d. Compute Output

$$
\mathbf{O}[:, :, t, :] = \sum_{k=1}^{K} \mathbf{q}_t[..., k] \times \mathbf{h}^{(t+1)}[..., k, :]
$$

**Element-wise Representation**:
$$
\mathbf{O}[b, h, t, v] = \sum_{k=1}^{K} \mathbf{q}_t[b, h, k] \times \mathbf{h}^{(t+1)}[b, h, k, v]
$$

- **Explanation**:
  - The output at time step $t$ is the sum of the products of the scaled queries and the updated hidden state across the key dimension $K$.

### 3. Final Output

- **Return Values**:
  $$
  \begin{cases}
    (\mathbf{O}, \mathbf{h}^{(T)}), & \text{if } \text{output\_final\_state} = \text{True} \\
    (\mathbf{O}, \text{None}), & \text{otherwise}
  \end{cases}
  $$

## Summary of Equations

1. **Initialization**:
   $$
   \mathbf{h}^{(0)} = 
   \begin{cases}
     \text{initial\_state}, & \text{if provided} \\
     \mathbf{0}, & \text{otherwise}
   \end{cases}
   $$
   $$
   \mathbf{O} = \mathbf{0} \in \mathbb{R}^{B \times H \times T \times V}
   $$

2. **For Each Time Step $t$**:
   - **Scaled Query**:
     $$
     \mathbf{q}_t = \frac{\mathbf{Q}[:, :, t, :]}{\sqrt{K}}
     $$
   - **Gated Hidden State Update**:
     $$
     \mathbf{h}^{(t+1)} = \mathbf{h}^{(t)} \odot \exp\left(\mathbf{G}[:, :, t, :]\right)[..., \text{None}] + \mathbf{k}_t \otimes \mathbf{v}_t
     $$
   - **Output Computation**:
     $$
     \mathbf{O}[:, :, t, :] = \sum_{k=1}^{K} \mathbf{q}_t[..., k] \times \mathbf{h}^{(t+1)}[..., k, :]
     $$

3. **Final Output**:
   $$
   \begin{cases}
     (\mathbf{O}, \mathbf{h}^{(T)}), & \text{if } \text{output\_final\_state} = \text{True} \\
     (\mathbf{O}, \text{None}), & \text{otherwise}
   \end{cases}
   $$

## Detailed Explanation

### Scaling Factor

The scaling factor 
$$
\text{scale} = \frac{1}{\sqrt{K}}
$$ 
is used to normalize the query vectors. This prevents the dot products from growing too large, which can destabilize the learning process, especially in attention mechanisms.

### Hidden State Update

The hidden state $\mathbf{h}^{(t+1)}$ is updated using a gating mechanism controlled by $\boldsymbol{\gamma}_t$:
$$
\mathbf{h}^{(t+1)} = \mathbf{h}^{(t)} \odot \boldsymbol{\gamma}_t[..., \text{None}] + \mathbf{kv}_t
$$

- **Gating Mechanism**: 
  $$
  \boldsymbol{\gamma}_t = \exp(\mathbf{G}[:, :, t, :])
  $$
  ensures that the gating factors are positive, allowing the model to regulate the contribution of the previous hidden state effectively.

- **Outer Product $\mathbf{kv}_t$**: Captures the interaction between keys and values at the current time step, enriching the hidden state with new information.

### Output Computation

The output $\mathbf{O}$ at each time step is a weighted sum of the updated hidden state, where the weights are determined by the scaled queries:
$$
\mathbf{O}[:, :, t, :] = \sum_{k=1}^{K} \mathbf{q}_t[..., k] \times \mathbf{h}^{(t+1)}[..., k, :]
$$

This operation aggregates information across the key dimension, producing a context-aware output for each time step.

### Broadcasting and Dimensions

- **Broadcasting**: The notation $\boldsymbol{\gamma}_t[..., \text{None}]$ adds an extra dimension to $\boldsymbol{\gamma}_t$ to facilitate element-wise multiplication with $\mathbf{h}^{(t)}$, aligning their dimensions appropriately.

- **Dimension Alignment**: Ensuring that tensor dimensions match is crucial for valid mathematical operations, such as element-wise multiplication and summation.

## Interpretation and Significance

- **Recurrent Computation**: The function embodies a recurrent neural network (RNN) structure, where the hidden state $\mathbf{h}$ carries information across time steps, enabling the model to capture temporal dependencies in the data.
  
- **Gating Mechanism**: Similar to gates in Long Short-Term Memory (LSTM) networks, $\boldsymbol{\gamma}_t$ controls the flow of information, allowing the model to retain or forget information as needed.
  
- **Attention-Like Mechanism**: Although not a traditional attention mechanism, the projection of the hidden state onto the queries resembles attention by weighting different components based on their relevance.
  
- **Efficiency**: The recurrent update avoids storing large attention matrices, making the model more memory-efficient, especially beneficial for processing long sequences.



## 再來是 fused_recurrent.py

## fused_recurrent code 暫時不顯示。數學如下：和 naive recurrent 好像非常接近。

Fused 並不是把 recurrent and chunk fused.  而是把幾個 operations (outer product and element-wise multiplication) fuse 在一起。
### **1. Hidden State Update**

At each timestep tt, the hidden state hth_t is updated as follows:

$$
h_t = \left( h_t \odot \exp(\text{gates}) \right) + \left( k_t \otimes v_t \right)
$$

Where:

- $h_t$ is the hidden state at timestep tt, with shape [K,V][K, V].
- $\odot$ denotes **element-wise (Hadamard) multiplication**.
- $\exp(\text{gates})$ applies the exponential function element-wise to the gating tensors (if provided).
- $k_t$ (key vector) has shape [K][K], and $v_t$ (value vector) has shape [V][V].
- $k_t \otimes v_t$ denotes the **outer product**, resulting in a matrix of shape [K,V][K, V].

---











### **2. Output Computation**

The output oto_t at timestep tt is computed using **matrix multiplication** between the transposed hidden state $h_t^\top$ and the query $q_t$:

$$
o_t = h_t^\top \cdot q_t
$$

Where:

- $h_t^\top$ is the transpose of hth_t, with shape [V,K][V, K].
- $q_t$ is the query vector at timestep tt, with shape [K][K].
- The result oto_t is a vector with shape [V][V].



### Fused Loss function

https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/modules/fused_cross_entropy.py

Fused chunks loss.

```python

def fused_cross_entropy_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignore_index: int = -100,
    process_group=None,
):

    n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
    losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)

    with torch.cuda.device(logits.device.index):
        cross_entropy_fwd_kernel[(n_rows, n_splits)](
            losses,  # data ptrs
            lse,
            z_losses,
            logits,
            target,
            label_smoothing,
            logit_scale,
            lse_square_scale,
            ignore_index,
            total_classes,
            class_start_idx,
            n_cols,  # shapes
            n_rows,
            logits.stride(0),  # strides
            BLOCK_SIZE=BLOCK_SIZE,  # constants
            num_warps=num_warps,
            SPLIT=split
        )

    if split:
        # If there's no label_smoothing, if target are in the vocab of this partition, losses contains
        # - predicted logit, and 0 otherwise.
        # If there's label_smoothing=0.1, for target in the vocab of this partition, losses contains
        # -0.9 * predicted logit - 0.1 * sum logit / total_classes.
        # For target not in the vocab of this partition, losses contains
        # -0.1 * sum logit / total_classes.
        if n_splits > 1:
            lse = torch.logsumexp(lse, dim=0)
            losses = losses.sum(dim=0)
            
        if world_size > 1:  # materialization
            lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
            handle_losses = torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
            )
            lse = torch.logsumexp(lse_allgather, dim=0)
            handle_losses.wait()
        # After the allreduce, if there's no label_smoothing, the total losses are - predicted_logit,
        # we just have to add the (global) lse.
        # If there's label_smoothing=0.1, the total losses are
        # -0.9 * predicted_logit - 0.1 * sum logit / total_classes.
        # Again, we just have to add the (global) lse.
        losses += lse
        if lse_square_scale != 0.0:
            z_losses = lse_square_scale * lse.square()
            z_losses.masked_fill_(target == ignore_index, 0.0)
            losses += z_losses
        else:
            z_losses = torch.zeros_like(losses)
        losses.masked_fill_(target == ignore_index, 0.0)

    return losses, z_losses, lse, total_classes, class_start_idx

```

The chunk procedure in the provided code appears in the `fused_cross_entropy_forward` function, where it processes logits in smaller blocks or chunks for efficiency, especially when working with large vocabulary sizes or distributed systems.

Here’s how the chunking process is implemented and managed:

### Key Elements of the Chunk Procedure

1. **Block Size Determination**: (這裏的 block size 就是 chunk size) and **Splitting the Vocabulary into Chunks**:
   
    - The `BLOCK_SIZE` is determined based on the vocabulary size (`n_cols`) and is set to a power of 2 (via `triton.next_power_of_2`), ensuring it aligns with hardware efficiency requirements
    - The vocabulary (`n_cols`) is divided into chunks of size `BLOCK_SIZE`. The total number of splits (`n_splits`) is calculated
      ```python
      MAX_BLOCK_SIZE = 64 * 1024
      BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
      n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
      ```
      This ensures that each chunk can be processed separately, even for large vocabularies.

1. **Loss, LSE, and Z-Loss Allocation (爲什麽需要三種 loss, 因爲 chunks?)**:
    - Storage tensors (`losses`, `lse`, `z_losses`) are initialized to hold intermediate results for each chunk:
    - losses: cross-entropy loss,  lse: log-sum-exp,  z_losses: average lse?

復習一下：
### `loss`: The Core Cross-Entropy Loss

- **Purpose**:
    - This is the primary objective function for classification tasks (只是非常大的分類，因爲有几萬或几十萬的 vocab). It measures how well the predicted probability distribution matches the true label distribution.
- **Formula**:
$$
	loss = - \sum_{i} p_i \cdot \log(\hat{p}_i)
$$

​	where:
​    	$p_i$: True probability of class $i$ (from labels).
​    	$\hat{p}_i$: Predicted probability of class $i$ (from the model’s logits).

- **Key Characteristics**:
    - Guides the model to minimize the difference between true and predicted probabilities.
    - Is computed across all classes in the vocabulary and summed up.
- **Why it’s Needed**:
    - It’s the main optimization target in supervised learning, ensuring the model improves its predictions over time.

### `lse`: Log-Sum-Exp

- **Purpose**:
    - The **log-sum-exp (LSE)** value is used for numerical stability in the softmax operation, especially when handling very large or small logits.
- **Formula**:
$$
    lse=\log \left( \sum_{i} \exp(\text{logit}_i) \right)
$$
- **Key Role**:
    - In cross-entropy loss computation, $\text{lse}$ appears in the denominator of the softmax: 
$$
    \hat{p}_i = \frac{\exp(\text{logit}_i)}{\sum_{j} \exp(\text{logit}_j)} 
$$

​	- Taking the log of the softmax for numerical stability reduces to: 
$$
    -\log(\hat{p}_i) = \text{lse} - \text{logit}_i
$$
- **Why it’s Needed**:
    - **Numerical Stability**:
        - Directly exponentiating logits can cause overflow for large values or underflow for small values. The LSE reformulates this operation in a more stable way.
    - **Efficiency**:
        - The LSE is reused across multiple calculations, reducing redundant computation.
    - In distributed setups, $\text{lse}$ must be aggregated across processes to ensure consistency of probabilities.

### `z_loss`: Regularization Term

- **Purpose**:    
    - The **z-loss** is an optional regularization term added to the loss to stabilize training, particularly for models with large vocabularies.
- **Formula**:

$$
\text{z-loss} = \text{lse}^2 \cdot \text{lse-square-scale}
$$

- The squared LSE penalizes extremely confident predictions or very large logits, which can destabilize training.
- **Key Characteristics**:
    - $\text{lse square scale}$ controls the strength of the z-loss.
    - Encourages the model to avoid overconfidence in its predictions, which can lead to poor generalization.
- **Why it’s Needed**:
    - **Training Stability**:
        - Large logits can make gradients unstable. The z-loss acts as a regularizer, ensuring logits don’t grow unbounded.
    - **Improved Generalization**:
        - Penalizing overconfident predictions encourages the model to remain calibrated and prevents overfitting.
    - **Practical Use**:
        - Often helpful in large-scale training tasks or when working with very large vocabularies (e.g., language models).

### Summary: Why All Three Losses are Needed

|Loss Component|Purpose|Benefit|
|---|---|---|
|**`loss`**|Measures prediction accuracy.|Core metric for training and evaluating classification tasks.|
|**`lse`**|Handles numerical stability.|Prevents numerical overflow/underflow during softmax and ensures reliable probability computations.|
|**`z_loss`**|Adds regularization to logits.|Improves training stability and prevents overconfidence, especially in large-scale models.|

Together, these three components ensure:

1. Accurate optimization (`loss`).
2. Stable computation (`lse`).
3. Better generalization and training robustness (`z_loss`).

```python
        loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
        losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
```

4. **Processing Each Chunk**:
   
    - The `cross_entropy_fwd_kernel` is invoked for each chunk, processing rows of the logits matrix split across chunks in parallel:
      
        ```python
        cross_entropy_fwd_kernel[(n_rows, n_splits)](
            losses,
            lse,
            z_losses,
            logits,
            target,
            label_smoothing,
            logit_scale,
            lse_square_scale,
            ignore_index,
            total_classes,
            class_start_idx,
            n_cols,
            n_rows,
            logits.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            SPLIT=split
        )
        ```
        
        - `col_block_idx`: Identifies which chunk of the vocabulary is being processed.
        - The Triton kernel processes each chunk independently and stores the intermediate results in `losses`, `lse`, and `z_losses`.
5. **Combining Results Across Chunks**:
   
    - If the vocabulary is split across chunks (`n_splits > 1`), the results from all chunks are combined using operations like `torch.logsumexp` for the log-sum-exponential values:
      
        ```python
        if n_splits > 1:
            lse = torch.logsumexp(lse, dim=0)
            losses = losses.sum(dim=0)
        ```
    
6. **Distributed Aggregation (Tensor Parallelism)**:
   
    - **If the vocabulary is distributed across multiple processes (e.g., in a Tensor Parallel setup), additional communication steps aggregate results across processes:**  這是對應 materialization 的情況，因爲分成多個 GPUs.
      
        ```python
        if world_size > 1:
            lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
            handle_losses = torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
            )
            lse = torch.logsumexp(lse_allgather, dim=0)
            handle_losses.wait()
        ```
    
7. **Final Adjustments**:
   
    - After combining across chunks or processes, final adjustments are made to the loss, including adding the z-loss term if applicable:
      
        ```python
        if lse_square_scale != 0.0:
            z_losses = lse_square_scale * lse.square()
            z_losses.masked_fill_(target == ignore_index, 0.0)
            losses += z_losses
        losses.masked_fill_(target == ignore_index, 0.0)
        ```
        


## Causal Linear Attention in Training and Inferencing?

Linear attention 的好處是在 n >> d 的時候。
因此一個想法就是, n <= d 作為一個 chunk,  仍然採用一般 transformer (quadratic attention) 方法。沒有 training causal 的問題。
但是在 n > d 時候，分成一個個 chunk, 採用 linear attention 方法！因為是以 chunk 為單位。不會需要每個 q.  

答案是 mixed linear attention and quadratic attention using chunk processing.

1. Pure transformer mode: 問題是 long context 的 training 和 inferencing 時 computation 和 memory 都和長度平方正比的問題。
2. Pure sequential mode: 基本和 RNN 一樣。問題是對於 training 太慢。對於 inferencing 的 pre-fill (prompt mode) stage 也太慢。但是對於 inferencing 的 generation mode 非常好。因為 NO KV cache 固定 memory and token time。

Implementing Causality with Cumulative Sums
By using cumulative sums, we can efficiently compute the causal attention outputs:

- **Cumulative Sum of Keys**:
  - $S_t = \sum_{i=1}^{t} k_i = S_{t-1} + k_t$
- **Cumulative Sum of Context**:
  - $Z_t = \sum_{i=1}^{t} k_i v_i^\top = Z_{t-1} + k_t v_t^\top$
- **Normalization Factor**:
  - $D_t = q_t^\top S_t + \epsilon$
- **Attention Output**:
  - $\text{out}_t = \frac{1}{D_t} q_t^\top Z_t$

This ensures that at each position $t$, the output depends only on keys and values from positions $\leq t$




3. (Non-materialization) Chunk sequential mode (下圖 a):  分成 chunks

$$
\begin{aligned}
& \mathbf{S}_{[i+1]}=\mathbf{S}_{[i]}+\underbrace{\sum_{j=i C+1}^{(i+1) C} \boldsymbol{k}_j^{\top} \boldsymbol{v}_j}_{\mathbf{K}_{[i]}^{\top} \mathbf{V}_{[i]}} \in \mathbb{R}^{d \times d} . \\
& \mathbf{O}_{[i+1]}=\underbrace{\mathbf{Q}_{[i+1]} \mathbf{S}_{[i]}}_{\text {inter-chunk: } \mathbf{O}_{[i+1]}^{\text {inexe }}}+\underbrace{\left(\left(\mathbf{Q}_{[i+1]} \mathbf{K}_{[i+1]}^{\top}\right) \odot \mathbf{M}\right) \mathbf{V}_{[i+1]}}_{\text {intra-chunk: } \mathbf{O}_{[i+1]}^{\text {intra }}},\\
&\text { where } \mathbf{O}_{[i+1]} \in \mathbb{R}^{C \times d} 
\end{aligned}
$$

 - 在 inter-check 用 linear attention 計算 $S_n$.  
 - 在 intra-chunk 使用 pure transformer mode $O_{intra}$ 加上 initial $O_{inter}$, 因為一次可以產生所有的 outputs (利用上式).  也只需要存一個 chunk 的 (Q)KV cache (cxdx3) 和一個 S (dxd) 
 - 好處是節省 memory.  壞處是雖然速度比 pure sequential mode，但因為整個過程仍然是 sequential, 並非是最快的方法。

![[Pasted image 20241120175328.png]]

- 
4. (Materialization) Mixed sequential (上圖 b) + chunk-wise parallel mode (上圖 c):  基本結構和 3 完全一樣。差異是順序。
- 先計算所有 $S_n$  in sequence.  注意這是 in token chunk (c=128 or higher), 所以比 2 的 sequence in token 要快的多。
- 有了所有的 $S_n$, 接下來就是平行計算所有的 outputs.  因為這些 chunks 之間都沒有直接關係 (都是經過 $S_n$),  可以放在 batch dimension (類似 FlashAttention).   


![[Pasted image 20241120181249.png]]


## Causal Attention Training and Inferencing

|                        | Mode1 | Mode2 | Mode3                     | Mode4 |
| ---------------------- | ----- | ----- | ------------------------- | ----- |
| Training-Foward        |       |       |                           | V     |
| Training-Backward      |       |       |                           |       |
| Inferencing-Prompt     |       |       | V.....(No, 只要半套，就是 $S_n$) |       |
| Inferencing-Generation |       | V     |                           |       |

### 注意在 inferencing prompt mode, 其實只要做 mode 3 的半套，就是 $S_n$.  在最後一個 block 計算 output.  因為 prompt mode 只 care 產生 output token 的哪一個 output 即可。不在乎之前的 o.

也可以視為 mode 2.  就是先 compute 所有的 kv sum, 可以平行計算 (例如用 binary tree)，最後 sum.  然後一次得到最後的 output.   因為 prompt (prefill) mode 不在意之前的 output.  只有 training 才在乎。

$$
\begin{aligned}
& \mathbf{S}_{[i+1]}=\mathbf{S}_{[i]}+\underbrace{\sum_{j=i C+1}^{(i+1) C} \boldsymbol{k}_j^{\top} \boldsymbol{v}_j}_{\mathbf{K}_{[i]}^{\top} \mathbf{V}_{[i]}} \in \mathbb{R}^{d \times d} . \\
& \mathbf{O}_{last}=\underbrace{\mathbf{Q}_{last} \mathbf{S}_{last-1}}_{\text {inter-chunk: } \mathbf{O}_{last}^{\text {inexe }}}+\underbrace{\left(\left(\mathbf{Q}_{last} \mathbf{K}_{last}^{\top}\right) \odot \mathbf{M}\right) \mathbf{V}_{last}}_{\text {intra-chunk: } \mathbf{O}_{[i+1]}^{\text {intra }}},
\end{aligned}
$$




## Causal Linear Attention

以下是 performer 的 training 部分的説明。注意的是 uni-directional 也就是 causal 的部分。
**我是看不懂以下 unidirectional (causal) 部分的 G matrix。直接看 code 還比較清楚。就是 iterative 的算法**
![[Pasted image 20241117222056.png]]


### 白話文 Cumsum

### Implementing Causality with Cumulative Sums

By using cumulative sums, we can efficiently compute the causal attention outputs:

- **Cumulative Sum of Keys**:
  - $S_t = \sum_{i=1}^{t} k_i = S_{t-1} + k_t$
- **Cumulative Sum of Context**:
  - $Z_t = \sum_{i=1}^{t} k_i v_i^\top = Z_{t-1} + k_t v_t^\top$
- **Normalization Factor**:
  - $D_t = q_t^\top S_t + \epsilon$
- **Attention Output**:
  - $\text{out}_t = \frac{1}{D_t} q_t^\top Z_t$

This ensures that at each position $t$, the output depends only on keys and values from positions $\leq t$

有兩種模型：
1. Recursive 類似 RNN 方法。好處是 inference generation 節省算力和 storage
2. parallel 方法:  用於 training 和 prefill.  不過兩者還是不同。 
	- Prefill 只需要 prompt 最後的 $out_t$.
	- Training 則需要所有的 $out_1, out_2, ..., out_t$

不止考慮算力，還要考慮 memory access.   計算 $out_1, out_2, .., out_t$ 需要大量 memory access.  而 non-causal 只需要一次。  

問題不在於 inferencing, 不論是 pre-fill 或是 generation.  因爲 $S_t$ 和 $Z_t$ 都可以一次運算，或是平行運算。注意
- Prefill 只需要 $out_t$: 也就是 $Z_t$, $q_t$, 和 $S_t$.  當然要得到 $Z_t$ 還是需要 $k_1, k_2, ..., k_t$ 和 $v_1, v_2, ..., v_t$.
- Generation 更簡單，只需要之前的  $Z_t$, $q_t$, 和 $S_t$. 
- 問題是 training,  需要所有的 $out_1, out_2, ..., out_t$ , 也就是所有的 $Z_1, Z_2, ..., Z_t$  和 $S_1, S_2, ..., S_t$
	- $S_1, S_2, ..., S_t$  沒有問題，就是 $k_1, k_2, .., k_t \in R^{1\times d}$  乘以 lower triangular matrix with all-1 elements.
	- $Z_1, Z_2, ..., Z_t$ 應該要特別主要,  $k_1 v_1, k_2 v_2, ..., k_t v_t \in R^{d\times d}$  也是可以乘 乘以 lower triangular matrix with all-1 elements?


## Source

- DiJiang Q&A:  https://github.com/YuchuanTian/DiJiang/issues/6
- Lightning attention:  https://arxiv.org/pdf/2401.04658
- Transformer in linear time, FLASH (not Flash Attention):  https://arxiv.org/pdf/2202.10447
- [[2023-03-26-Transformer_LLM]] , [[2023-02-18-Attn_All_U_Need_Visual]]
- https://www.kexue.fm/archives/7546/comment-page-1: check this article
- FAST algorithm: https://arxiv.org/pdf/2009.14794
- https://teddykoker.com/2020/11/performers/
- https://teddykoker.com/2020/11/performers/
- Linear Attention 打改变 Transformer 大模型结构垄断 : https://www.bilibili.com/video/BV1V7s9etEmQ/?spm_id_from=333.999.0.0&vd_source=a99fc6374f18662fe559d32fdc3a80cd
- Transformer are RNNs: https://arxiv.org/pdf/2006.16236
- TransNormerLLM:  https://arxiv.org/pdf/2307.14995
- Universal Transformer:  https://arxiv.org/pdf/1807.03819
- Transformer Quality in Linear Time: https://arxiv.org/pdf/2202.10447

