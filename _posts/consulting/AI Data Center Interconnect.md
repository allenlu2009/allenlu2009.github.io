
# 幾個主角

CPU:  負責所有軟體執行 including hypervisor, virtual machine, linux OS, application software. 
* CPU increasingly becomes **control plane**, not **data plane**, in AI systems.

GPU:  AI scalar, vector, matrix 執行。
* GPU also owns **DMA engines** and **memory endpoints**, which is crucial for scale-up/out/across.

HBM 是被封裝在 GPU 用 interposer, 但是分開的 dies。可以視爲是 GPU 的私有財產。

DDR 是連在 CPU 的 external system memory.  BW/latency fundamentally different from HBM.

**(New) Memory pooling / disaggregation:**
- **DDR pooling:** CPU 與 GPU 可透過 **CXL fabric** 存取共享或分區的 DDR 記憶體（具備 coherency semantics）。
- **HBM pooling:** 透過 **proprietary high-bandwidth fabrics（例如 Celestial AI photonic interconnect）**，GPU 可對遠端 HBM 進行 **DMA 存取**，但 **不具備 cache-coherent shared memory semantics**。

# AI Data Center Interconnect

**AI server interconnect 一般分成兩塊：
Data plane: GPU interconnect (scale-up/out/across)，這是 peer-communication 的關係。 
Control plane: CPU-to-CPU or CPU-to-GPU or CPU to direct memory or shared memory.  這算是 host-device 的關係。**

**傳統的 CPU server 只有 host-device 的關係。但是 GPU interconnect 的 peer-communication 是 AI data center 最重要的基石。**

# GPU/TPU/NPU Interconnect, scale-up/out/across.   這是 Peer communication, 和 CPU 完全無關！

scale-up, scale-out, and scale-across.  具體就是
- Scale-up:  GPU to GPU on the same board, the same rack, example: NVLink or UALink.
- Scale-out:  GPU to GPU on different rack, example: InfiniteBand or Ethernet.
- Scale-across: GPU to GPU on different data center, example: Ethernet-like, e.g. Spectrum Etherenet

這是從 GPU 角度出發的遠近關係。

另外是 CPU-GPU 一般是用 PCIe，雖然在用一個 board, 也不會稱爲 scale-up, 有時稱爲 scale-out? NO!

## Clean comparison table: Scale-up vs Scale-out vs Scale-across

### GPU Interconnect Taxonomy (Architectural View)

| Dimension                 | **Scale-up**                                      | **Scale-out**                                 | **Scale-across**                                    |
| ------------------------- | ------------------------------------------------- | --------------------------------------------- | --------------------------------------------------- |
| **Primary goal**          | Increase GPU performance as _one big accelerator_ | Increase total GPU count within a data center | Extend a single AI job across multiple data centers |
| **GPU scope**             | Same board / same rack / same pod                 | Multiple racks, same data center              | Multiple data centers                               |
| **Canonical examples**    | NVLink, NVSwitch, UALink                          | InfiniBand, Ethernet (intra-DC)               | NVIDIA Spectrum-XGS Ethernet                        |
| **Data path**             | GPU HBM → DMA → fabric → DMA → GPU HBM            | GPU HBM → DMA → NIC → network → NIC → GPU HBM | Same as scale-out, but over long-haul links         |
| **CPU involvement**       | ❌ None (control only)                             | ❌ None (control only)                         | ❌ None (control only)                               |
| **HBM involved?**         | ✅ Directly (HBM↔HBM)                              | ✅ Directly (HBM↔HBM)                          | ✅ Directly (HBM↔HBM)                                |
| **Cache coherency**       | ❌ No                                              | ❌ No                                          | ❌ No                                                |
| **Latency sensitivity**   | Extremely high                                    | High                                          | Very high but physics-limited                       |
| **Bandwidth requirement** | TB/s class                                        | 100s of GB/s aggregate                        | Optimized for predictability, not raw BW            |
| **Communication style**   | Fine-grained, frequent collectives                | Bulk collectives, hierarchical                | Hierarchical, latency-tolerant collectives          |
| **Typical software**      | NCCL, CUDA collectives                            | NCCL + RDMA                                   | NCCL with distance-aware algorithms                 |
| **Physical distance**     | cm → meters                                       | meters → hundreds of meters                   | km → 100s of km                                     |
| **Key constraint**        | Switch radix, power, packaging                    | Network congestion, topology                  | Speed of light, jitter                              |
| **What it is NOT**        | Not CPU attach                                    | Not shared memory                             | Not coherent memory                                 |

# 幾個誤區

### **If the path goes through the CPU, it is not scale-up/out/across.**

GPU scale-up/out/across 到底是什麽意思。**是 GPU 内的 HBM to 另一個 GPU 的 HBM 用 DMA 交換資料。因爲非常大量資料，不會進入 CPU 或是 GPU 内部的 register or computing pipeline.  而是直接 GPU 的 HBM to GPU 的 HBM!  這是反直覺的！**

Scale-up:  GPU↔GPU = HBM → DMA → fabric → DMA → HBM
Scale-out/across: GPU HBM → DMA → NIC → network → NIC → GPU HBM

#### CPU↔GPU is not scale-out！
CPU-GPU … even on one board, sometimes called scale-out?
❌ This is **not correct** in strict architecture language.

- CPU↔GPU is **host-device attach**
- It is **neither scale-up nor scale-out**
- It is a **heterogeneous attach domain**
    
> ✔ Correct wording:
> - **Host–device interconnect**
> - **Accelerator attach**
> - **Heterogeneous memory hierarchy**

### Host Interconnect: PCIe and CXL

CPU - GPU:  不但是 data 還要有 coherency.  傳統的 PCIe or PCIe switch 只管 data movement, 不管 coherency.  CXL 則是 PCIe switch + coherency.  
- PCIe:
    - DMA semantics
    - No cache coherency
- CXL:
    - Adds **cache & memory coherency protocols**
    - Still rides on PCIe PHY
    - Switch ≠ coherency by itself; protocol matters

Correct restatement:
> **CXL = PCIe physical layer + coherent memory semantics**

This matters for:
- Unified CPU↔GPU address space
- Memory pooling
- Load/store access


### GPU - GPU 是否也要 cache coherent? NO!

### 🔥 This is subtle and important

**Short answer:**
❌ **GPU↔GPU scale-up does NOT require cache coherency**

**Why?**
* AI workloads use:
  * Bulk DMA
  * Explicit synchronization
  * Collective ops (AllReduce, AllGather)
  
* GPUs already manage memory consistency via:
  * Barriers
  * Kernel launches
  * Stream synchronization

**Coherency would:**
* Add latency
* Add protocol overhead
* Reduce scalability

> NVLink, UALink, InfiniBand **are not cache-coherent fabrics**

### When *would* GPU coherency matter?

* Fine-grained pointer sharing
* CPU-style shared memory programming
* General-purpose workloads

👉 **Not AI training**


### A) Interconnect taxonomy

| Category         | What scales | Fabric          | Memory semantics   |
| ---------------- | ----------- | --------------- | ------------------ |
| **Host attach**  | CPU↔GPU     | PCIe / CXL      | Optional coherency |
| **Scale-up**     | GPU↔GPU     | NVLink / UALink | DMA, non-coherent  |
| **Scale-out**    | Node↔Node   | InfiniBand      | Message passing    |
| **Scale-across** | DC↔DC       | Ethernet        | Message passing    |

### Key semantic points (text explanation)

|Interconnect|Primary domain|Typical use|Coherency|Best performance for|
|---|---|---|---|---|
|**NVLink / NVSwitch**|On-board GPU fabric|GPU↔GPU HBM DMA|No|Tight GPU scale-up|
|**UALink**|Multi-GPU fabric|GPU↔GPU HBM DMA|No|Open scale-up|
|**PCIe**|CPU↔device, general I/O|DMA|No|GPU attach, general I/O|
|**CXL**|PCIe + coherency|CPU↔GPU shared memory, pools|Yes|Host ↔ accelerator memory|
|**Ethernet / Spectrum-XGS**|Data center fabric|Long-distance NCCL|No (protocols over it)|Scale-across distributed jobs|

This clarifies their **positioning**:
- NVLink / UALink = **true GPU scale-up fabrics**
- PCIe = **host attach layer**
- CXL = **coherent attach + memory pooling layer**
- Ethernet / Spectrum-XGS = **long-distance connectivity** across data centers.



# 幾個不同的 Scale-Up (and out) 比較
## NVSwitch vs Google TPU OCS vs Photonic Fabric (Celestial AI)

| Dimension                             | **NVSwitch (NVLink)**                                | **Google TPU OCS**                                | **Photonic Fabric (Celestial AI)**                                 |
| ------------------------------------- | ---------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| **Primary owner / ecosystem**         | **NVIDIA**                                           | **Google**                                        | **Celestial AI**                                                   |
| **Main goal**                         | Ultra-low-latency **scale-up** inside a tight domain | Massive **scale-out** TPU pods with optical reach | Scale-up **+ memory pooling** beyond copper limits                 |
| **Switching medium**                  | Electrical (copper)(optical NVLink exists, niche)    | Optical (fiber)                                   | Optical core + E/O/E at endpoints                                  |
| **Switching type**                    | Electrical ASIC crossbar                             | Optical **circuit switching** (MEMS)              | Optical switching plane (crossbar / mesh)                          |
| **Topology (logical)**                | Full all-to-all mesh                                 | Reconfigurable mesh / torus                       | Mesh / all-to-all (may be staged)                                  |
| **Hop semantics (XPU→XPU)**           | **Always 1-hop** (within domain)                     | **Not 1-hop** (topology-dependent)                | **Often 1-hop**, not guaranteed at large scale                     |
| **Typical cluster size (one domain)** | 8–72 GPUs (hard boundary)                            | 1K–4K+ TPUs (soft boundary)                       | 64–512 XPUs (1-hop)1K+ XPUs (multi-hop)                            |
| **Reconfiguration granularity**       | Fine-grain, cycle-level                              | **Coarse-grain** (job / phase / fault)            | Medium-grain (fabric-level, more dynamic than OCS)                 |
| **On critical data path?**            | Yes (always)                                         | No (only topology setup)                          | Yes (fabric participates in traffic)                               |
| **Memory model**                      | Local HBM per GPU only                               | Local HBM per TPU only                            | **Hierarchical**: local HBM + **shared fabric HBM / memory pools** |
| **HBM inside switch/fabric?**         | ❌ No                                                 | ❌ No                                              | ✅ Yes (HBM co-packaged in fabric appliances)                       |
| **Shared / pooled memory**            | ❌                                                    | ❌                                                 | ✅ Core design goal                                                 |
| **Latency characteristics**           | Lowest, uniform                                      | Low, but topology-dependent                       | Low; may increase with multi-hop staging                           |
| **Software assumption**               | Tight coherence / collectives                        | Bulk-synchronous training                         | New memory + collective abstractions                               |
| **Maturity / deployment**             | Shipping at scale                                    | Shipping at hyperscale (Google only)              | Emerging / early deployments                                       |
| **Key strength**                      | Deterministic, fastest scale-up                      | Massive optical scale                             | Breaks HBM-per-XPU constraint                                      |
| **Key limitation**                    | Hard scale ceiling                                   | No fine-grain memory semantics                    | Ecosystem & integration maturity                                   |

---

## One-line positioning (very useful in exec discussions)

- **NVSwitch**: _“The fastest possible 1-hop electrical scale-up — but capped in size.”_
- **Google TPU OCS**: _“Optical wiring for massive TPU pods — topology, not memory.”_
- **Photonic Fabric**: _“Optical scale-up with shared memory — trades strict 1-hop for elasticity.”_
    

---

## Why this table matters strategically

- NVSwitch answers **latency first**
- OCS answers **scale first**
- Photonic fabric answers **memory-system first**
    
