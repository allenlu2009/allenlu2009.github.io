Got it—here’s a compact but deep dive that traces Cambricon’s NPU lineage from the classic research chips (DianNao family) into commercial Cambricon MLUs, and puts them head-to-head with NVIDIA GPUs and mobile NPUs from Qualcomm/MediaTek.

# 1) Roots: the “DianNao” research line (ICT-CAS)

**DianNao (2014, MICRO):** One of the first purpose-built DNN accelerators. It emphasized on-chip SRAM to avoid DRAM traffic, a small PE array, and fixed-function datapaths for MLP/CNN layers—delivering big energy–latency wins over CPUs/GPUs of the time. ([ACM Digital Library](https://dl.acm.org/doi/10.1145/2541940.2541967?utm_source=chatgpt.com "DianNao: a small-footprint high-throughput accelerator for ..."))

**DaDianNao (2014, MICRO):** Scaled the idea to a _multi-chip_ “machine-learning supercomputer.” Each chip paired large on-chip eDRAM with many PEs; a system of chips distributed model partitions to keep most weights on-chip and slash DRAM bandwidth needs. ([pubs.lenovo.com](https://pubs.lenovo.com/sr650/server_specifications?utm_source=chatgpt.com "Specifications | ThinkSystem SR650 | Lenovo Docs"))

**ShiDianNao (2015):** “Near-sensor” CNN accelerator for vision that pushed compute beside the image sensor to cut I/O power/latency; it foreshadowed edge inference NPUs. (The paper also notes that the DianNao line’s commercial spin-out became **Cambricon** in 2016.) ([BPB](https://bpb-us-w2.wpmucdn.com/sites.coecis.cornell.edu/dist/7/587/files/2023/06/Du_2015_ShiDianNao_v2.pdf?utm_source=chatgpt.com "ShiDianNao: Shifting Vision Processing Closer to the Sensor"))

**PuDianNao (2015, ASPLOS):** Broadened beyond CNNs to a _family_ of classical ML workloads (k-means, SVM, KNN, etc.), extracting common compute patterns and locality to build a more general ML accelerator. ([ResearchGate](https://www.researchgate.net/publication/281507672_PuDianNao?utm_source=chatgpt.com "PuDianNao | Request PDF"), [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2095809919306356?utm_source=chatgpt.com "A Survey of Accelerator Architectures for Deep Neural ..."))

**Key research takeaways that flow into Cambricon’s DNA**

- Keep weights/activations on-chip (SRAM/eDRAM) to tame bandwidth/energy.
    
- Deterministic dataflow in PE arrays; stream layers in tiles.
    
- Specialize the ISA/ops to the dominant kernels (conv/GEMM) while keeping some programmability.
    

# 2) From research to products: “Cambricon” ISA and MLUs

**Cambricon ISA (ISCA 2016):** A domain-specific, _programmable_ NN ISA—load/store with scalar, vector, and matrix instructions—to unify different NN types under one instruction set (vs. a purely fixed-function engine). This is the architectural bridge from the research prototypes to shipping silicon + software stacks. ([LET'S Configure Deep Learning](https://reconfigdeeplearning.wordpress.com/wp-content/uploads/2017/02/2016-isca_cambricon-an-instruction-set-architecture-for-neural-networks_cyj.pdf?utm_source=chatgpt.com "Cambricon: An Instruction Set Architecture for Neural ..."), [ACM Digital Library](https://dl.acm.org/doi/pdf/10.1109/ISCA.2016.42?utm_source=chatgpt.com "Cambricon: an instruction set architecture for neural"))

**Early commercial IP + phone NPUs:** Cambricon IP was used in Huawei’s **Kirin 970/980** smartphone NPUs before Huawei moved to its in-house **Da Vinci** architecture (see §3). ([Communications of the ACM](https://cacm.acm.org/news/chipping-away-at-big-data/?utm_source=chatgpt.com "Chipping Away at Big Data"), [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.06663?utm_source=chatgpt.com "AI Benchmark: All About Deep Learning on Smartphones in ..."))

**Datacenter accelerators (MLU series):**

- **MLU100** (TSMC 16 nm) was Cambricon’s first cloud card, paired with the **NeuWare** software stack/CNML. Public docs and studies profile end-to-end inference behavior on MLU100. ([Medium](https://medium.com/syncedreview/cambricon-unveils-its-first-ai-chip-for-cloud-computing-d3f7acdb4076?utm_source=chatgpt.com "Cambricon Unveils its First AI Chip for Cloud Computing"), [Semantic Scholar](https://www.semanticscholar.org/paper/Exploring-the-Performance-Bound-of-Cambricon-in-Wang-Li/8aae03f535dc73058263426f73d3a70caa2b7c75?utm_source=chatgpt.com "[PDF] Exploring the Performance Bound of Cambricon ..."), [ACM Digital Library](https://dl.acm.org/doi/abs/10.1007/978-3-030-49556-5_6?utm_source=chatgpt.com "Exploring the Performance Bound of Cambricon ..."))
    
- **MLU200 family** (**MLU220**, **MLU270**): multi-cluster chips built around “IPU 1M” cores. Example: MLU270 = 4 clusters × 4 IPU cores = 16 cores; server cards (S-series HHHL, X-series FHFL) target high-efficiency inference. Lenovo’s server guides even list MLU270 alongside NVIDIA T4. ([forum.cambricon.com](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf?utm_source=chatgpt.com "BANG C Language Developer Guide"), [FCC Report](https://fcc.report/FCC-ID/2ARVF-MLU270-S/4474024.pdf?utm_source=chatgpt.com "MLU270-S Series Intelligent Processing Card User Manual ..."), [pubs.lenovo.com](https://pubs.lenovo.com/sr650/server_specifications?utm_source=chatgpt.com "Specifications | ThinkSystem SR650 | Lenovo Docs"))
    
- Tooling: **NeuWare/CNML** runtime and the **BANG-C** programming model expose kernels, fusion, sparsity, tensor-layout transforms, etc. (Compiler paper and developer guides). ([Cambricon Developer Community](https://developer.cambricon.com/uploads/20220901/522e59edfe8c344303b95660bdffad54.pdf?utm_source=chatgpt.com "CNML Developer Guide"), [Subject No.i](https://subjectnoi.github.io/about/Paleozoic.pdf?utm_source=chatgpt.com "a high-performance compiler tool chain for deep learning ..."))
    

**What’s distinctive in Cambricon MLUs**

- A programmable NN-centric ISA (not a shader/GPU ISA) with vector/matrix ops tuned for conv/GEMM and data-movement control for tiling. ([LET'S Configure Deep Learning](https://reconfigdeeplearning.wordpress.com/wp-content/uploads/2017/02/2016-isca_cambricon-an-instruction-set-architecture-for-neural-networks_cyj.pdf?utm_source=chatgpt.com "Cambricon: An Instruction Set Architecture for Neural ..."))
    
- Chip floorplans prioritize sizable on-chip buffers and predictable dataflow to reduce off-chip bandwidth vs. GPUs for common inference graphs (esp. CNN/Transformer blocks), aiming at perf/W. (See product manuals positioning “high EER.”) ([FCC Report](https://fcc.report/FCC-ID/2ARVF-MLU270-S/4474024.pdf?utm_source=chatgpt.com "MLU270-S Series Intelligent Processing Card User Manual ..."))
    

# 3) About **Da Vinci** (Huawei Ascend)—closely related, but not Cambricon

Huawei’s **Da Vinci** (Ascend AI cores) is _Huawei’s_ in-house architecture (used in Ascend 310/910 and newer Kirin mobile NPUs after Kirin 980). A Da Vinci core pairs:

- A **3D Cube tensor unit** (4096 FP16 MACs + 8192 INT8 MACs per core),
    
- A **2048-bit vector unit** and scalar unit,
    
- An explicit multi-level on-chip buffer hierarchy with DMA/MTE.  
    This is often contrasted to GPUs because of its cube-tensor unit and explicit memory management style. Huawei adopted Da Vinci in **Kirin 810/990** and beyond. ([CMC Microsystems](https://www.cmc.ca/wp-content/uploads/2020/03/Zhan-Xu-Huawei.pdf?utm_source=chatgpt.com "DaVinci: A Scalable Architecture for Neural Network ..."), [Semantic Scholar](https://pdfs.semanticscholar.org/78b6/d0b2a12de2e7c106e8b4a81a6b29cf5c47b7.pdf?utm_source=chatgpt.com "DaVinci: A Scalable Architecture for Neural Network ..."), [Synced | AI Technology & Industry Review](https://syncedreview.com/2019/06/21/huawei-7nm-kirin-810-beats-snapdragon-855-and-kirin-980-on-ai-benchmark-test/?utm_source=chatgpt.com "Huawei 7nm Kirin 810 Beats Snapdragon 855 and Kirin 980 ..."), [Wikipedia](https://en.wikipedia.org/wiki/HiSilicon?utm_source=chatgpt.com "HiSilicon"))
    

> Why it matters here: _Lineage_. The team/company history ties back to the DianNao research at ICT-CAS; Cambricon IP powered Huawei’s earlier phone NPUs (Kirin 970/980), and Da Vinci later became Huawei’s independent evolution with a similar “tensor core + vector + explicit buffers” recipe for training/inference. ([Communications of the ACM](https://cacm.acm.org/news/chipping-away-at-big-data/?utm_source=chatgpt.com "Chipping Away at Big Data"), [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.06663?utm_source=chatgpt.com "AI Benchmark: All About Deep Learning on Smartphones in ..."))

# 4) Cambricon vs. NVIDIA GPUs vs. Qualcomm/MediaTek NPUs

## Workload scope & programmability

- **Cambricon MLUs (datacenter)** — Programmable NN ISA (Cambricon) + kernels via CNML/BANG. Great for inference at high perf/W on conv/GEMM-heavy graphs; training support has been limited in public materials compared to NVIDIA. ([LET'S Configure Deep Learning](https://reconfigdeeplearning.wordpress.com/wp-content/uploads/2017/02/2016-isca_cambricon-an-instruction-set-architecture-for-neural-networks_cyj.pdf?utm_source=chatgpt.com "Cambricon: An Instruction Set Architecture for Neural ..."), [Cambricon Developer Community](https://developer.cambricon.com/uploads/20220901/522e59edfe8c344303b95660bdffad54.pdf?utm_source=chatgpt.com "CNML Developer Guide"))
    
- **NVIDIA GPUs (Volta→Ampere→Hopper)** — General-purpose SIMT + **Tensor Cores**; massive ecosystem (CUDA, cuDNN, TensorRT). Hopper’s **Transformer Engine** auto-mixes FP8/FP16 per layer, pushing SOTA training/inference throughput. If you need end-to-end training at scale (L100/H100/B200 class), NVIDIA still dominates. ([NVIDIA Images](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf?utm_source=chatgpt.com "NVIDIA A100 Tensor Core GPU Architecture"), [Advanced Clustering Technologies](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf?utm_source=chatgpt.com "NVIDIA H100 Tensor Core GPU Architecture"), [NVIDIA Developer](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/?utm_source=chatgpt.com "NVIDIA Hopper Architecture In-Depth"), [NVIDIA](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/?utm_source=chatgpt.com "NVIDIA Hopper GPU Architecture"))
    
- **Qualcomm/MediaTek (mobile)** — Phone-class NPUs optimized for low-power on-device AI with tight DRAM budgets:  
    • **Qualcomm Hexagon/HTA/HTP** inside a heterogeneous “AI Engine” (with GPU/CPU). Tooling via AI Engine Direct; strong NNAPI integration and good operator coverage for mobile models. ([Qualcomm Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-88500-4/147_HTA.html?utm_source=chatgpt.com "Qualcomm® Hexagon™ Tensor Accelerator"), [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk?utm_source=chatgpt.com "Qualcomm AI Engine Direct SDK | Qualcomm Developer"))  
    • **MediaTek APU 7xx (e.g., APU 790)** with NeuroPilot, mixed precision (INT4/INT8/FP16), memory compression, and marketing around on-device GenAI/LoRA. ([corp.mediatek.com](https://corp.mediatek.com/news-events/press-releases/mediateks-new-all-big-core-design-for-flagship-dimensity-9300-chipset-maximizes-smartphone-performance-and-efficiency?utm_source=chatgpt.com "MediaTek's New All Big Core Design for Flagship ..."), [EE Times](https://www.eetimes.com/mediatek-ups-the-arms-race-in-mobile-socs/?utm_source=chatgpt.com "MediaTek Ups the Arms Race in Mobile SoCs"), [All About Circuits](https://www.allaboutcircuits.com/news/new-mediatek-soc-speeds-up-generative-ai-processing-at-the-edge/?utm_source=chatgpt.com "New MediaTek SoC Speeds Up Generative AI Processing ..."))
    

## Microarchitectural emphasis

- **Cambricon**: NN-centric compute cores (“IPU 1M”), fairly large on-chip buffers, explicit data movement, and kernel fusion to minimize DRAM trips—similar philosophy to research DianNao/DaDianNao. ([forum.cambricon.com](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf?utm_source=chatgpt.com "BANG C Language Developer Guide"))
    
- **NVIDIA**: Wide SMs with Tensor Cores; aggressive mixed precision (**FP8/FP16/BF16/INT8/INT4**), sparsity exploitation, and fast interconnects (**NVLink/NVSwitch**) for multi-GPU scale-out. ([Advanced Clustering Technologies](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf?utm_source=chatgpt.com "NVIDIA H100 Tensor Core GPU Architecture"), [NVIDIA Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html?utm_source=chatgpt.com "Using FP8 with Transformer Engine"), [The Next Platform](https://www.nextplatform.com/2022/03/31/deep-dive-into-nvidias-hopper-gpu-architecture/?utm_source=chatgpt.com "Deep Dive Into Nvidia's “Hopper” GPU Architecture"))
    
- **Qualcomm/MediaTek**: Smaller tensor/conv engines tightly integrated in SoCs; heavy use of quantization and operator co-design for mobile-power envelopes. ([Edge AI and Vision Alliance](https://www.edge-ai-vision.com/wp-content/uploads/2021/05/GS_012_Asghar_Qualcomm.pdf?utm_source=chatgpt.com "Click to insert title"), [PR Newswire](https://www.prnewswire.com/ae/news-releases/mediateks-new-all-big-core-design-for-flagship-dimensity-9300-chipset-maximizes-smartphone-performance-and-efficiency-301978589.html?utm_source=chatgpt.com "MediaTek's New All Big Core Design for Flagship ..."))
    

## Software stack & ecosystem

- **Cambricon**: **NeuWare/CNML** plus BANG-C; ONNX/TensorFlow/PyTorch front-ends typically lower to CNML. Ecosystem is growing but much smaller than CUDA; community benchmarks and third-party tooling are relatively sparse. ([Cambricon Developer Community](https://developer.cambricon.com/uploads/20220901/522e59edfe8c344303b95660bdffad54.pdf?utm_source=chatgpt.com "CNML Developer Guide"), [Subject No.i](https://subjectnoi.github.io/about/Paleozoic.pdf?utm_source=chatgpt.com "a high-performance compiler tool chain for deep learning ..."))
    
- **NVIDIA**: CUDA + cuDNN + TensorRT + Triton + colossal third-party ecosystem; leading support for cutting-edge kernels (flash-attn variants, MoE, FP8 training) and multi-node orchestration. ([Advanced Clustering Technologies](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf?utm_source=chatgpt.com "NVIDIA H100 Tensor Core GPU Architecture"), [NVIDIA Developer](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/?utm_source=chatgpt.com "NVIDIA Hopper Architecture In-Depth"))
    
- **Qualcomm/MediaTek**: Android/NNAPI integration; **AI Engine Direct** (Qualcomm) and **NeuroPilot** (MediaTek) SDKs focus on mobile deployment and power-aware scheduling. ([Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk?utm_source=chatgpt.com "Qualcomm AI Engine Direct SDK | Qualcomm Developer"), [Counterpoint Research](https://www.counterpointresearch.com/insights/mediatek-strengthens-premium-push-with-gen-ai-capabilities?utm_source=chatgpt.com "MediaTek Strengthens Premium Push With Gen AI ..."))
    

## Performance & efficiency positioning (high level)

- **Datacenter inference perf/W**: Cambricon markets higher “EER” (energy efficiency rate) vs. GPUs on typical DNN inference; MLU270 class often compared against inference GPUs like NVIDIA T4 in server BOMs. Independent, broad head-to-head public data is limited. ([FCC Report](https://fcc.report/FCC-ID/2ARVF-MLU270-S/4474024.pdf?utm_source=chatgpt.com "MLU270-S Series Intelligent Processing Card User Manual ..."), [pubs.lenovo.com](https://pubs.lenovo.com/sr650/server_specifications?utm_source=chatgpt.com "Specifications | ThinkSystem SR650 | Lenovo Docs"))
    
- **Training + giant models**: NVIDIA’s Hopper with FP8 **Transformer Engine** is purpose-built for LLM training/inference scale and has unmatched software maturity; Cambricon’s public focus has been more on inference acceleration. ([Advanced Clustering Technologies](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf?utm_source=chatgpt.com "NVIDIA H100 Tensor Core GPU Architecture"), [NVIDIA Blog](https://blogs.nvidia.com/blog/h100-transformer-engine/?utm_source=chatgpt.com "H100 Transformer Engine Supercharges AI Training ..."))
    
- **Edge/mobile**: Qualcomm/MediaTek NPUs excel at perf/W under tight thermal limits and support phone-centric ops/pipelines (camera, audio, on-device LLMs with quantization). Cambricon’s mobile IP was used historically in Kirin 970/980, but Cambricon’s current flagship efforts are datacenter/edge cards. ([Communications of the ACM](https://cacm.acm.org/news/chipping-away-at-big-data/?utm_source=chatgpt.com "Chipping Away at Big Data"))
    

# 5) Pros & cons summary

**Cambricon (MLU family)**

- **Pros**
    
    - NN-centric ISA and memory hierarchy → strong **inference perf/W**, predictable latency. ([LET'S Configure Deep Learning](https://reconfigdeeplearning.wordpress.com/wp-content/uploads/2017/02/2016-isca_cambricon-an-instruction-set-architecture-for-neural-networks_cyj.pdf?utm_source=chatgpt.com "Cambricon: An Instruction Set Architecture for Neural ..."), [FCC Report](https://fcc.report/FCC-ID/2ARVF-MLU270-S/4474024.pdf?utm_source=chatgpt.com "MLU270-S Series Intelligent Processing Card User Manual ..."))
        
    - Toolchain supports **kernel fusion, sparsity, layout transforms** for further wins. ([Subject No.i](https://subjectnoi.github.io/about/Paleozoic.pdf?utm_source=chatgpt.com "a high-performance compiler tool chain for deep learning ..."))
        
    - Product range from edge (MLU220) to datacenter (MLU270), with server OEM integrations. ([forum.cambricon.com](https://forum.cambricon.com/uploadfile/user/file/20201125/1606289569710855.pdf?utm_source=chatgpt.com "BANG C Language Developer Guide"), [pubs.lenovo.com](https://pubs.lenovo.com/sr650/server_specifications?utm_source=chatgpt.com "Specifications | ThinkSystem SR650 | Lenovo Docs"))
        
- **Cons**
    
    - **Ecosystem depth** and community resources lag CUDA; fewer public end-to-end benchmarks across diverse workloads. ([Cambricon Developer Community](https://developer.cambricon.com/uploads/20220901/522e59edfe8c344303b95660bdffad54.pdf?utm_source=chatgpt.com "CNML Developer Guide"))
        
    - Emphasis historically on **inference**; less visible track record for frontier-scale **training** vs. NVIDIA Hopper class. ([Advanced Clustering Technologies](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf?utm_source=chatgpt.com "NVIDIA H100 Tensor Core GPU Architecture"))
        

**NVIDIA GPUs (Hopper era)**

- **Pros**
    
    - **Best-in-class software** (CUDA, cuDNN, TensorRT) and rapid kernel innovation (FP8, Transformer Engine, structured sparsity). ([NVIDIA Images](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf?utm_source=chatgpt.com "NVIDIA A100 Tensor Core GPU Architecture"), [Advanced Clustering Technologies](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf?utm_source=chatgpt.com "NVIDIA H100 Tensor Core GPU Architecture"), [NVIDIA Blog](https://blogs.nvidia.com/blog/h100-transformer-engine/?utm_source=chatgpt.com "H100 Transformer Engine Supercharges AI Training ..."))
        
    - Scale-out fabric (**NVLink/NVSwitch**) and ubiquitous framework support → lowest friction for LLMs/foundation models. ([Advanced Clustering Technologies](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf?utm_source=chatgpt.com "NVIDIA H100 Tensor Core GPU Architecture"))
        
- **Cons**
    
    - General-purpose design can be **less energy-efficient** per inference at a given SLA than a tuned NPU; **TCO/power** are high. (NVIDIA improves this each gen, but it remains a camparison point.) ([The Next Platform](https://www.nextplatform.com/2022/03/31/deep-dive-into-nvidias-hopper-gpu-architecture/?utm_source=chatgpt.com "Deep Dive Into Nvidia's “Hopper” GPU Architecture"))
        

**Qualcomm / MediaTek (mobile NPUs)**

- **Pros**
    
    - **Excellent perf/W** under phone thermals; tight NNAPI integration and strong support for **quantized** GenAI (INT4/INT8) and camera pipelines. ([Qualcomm Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-63195-1/AI-hardware-cores-accelerators.html?utm_source=chatgpt.com "AI hardware cores/accelerators"), [EE Times](https://www.eetimes.com/mediatek-ups-the-arms-race-in-mobile-socs/?utm_source=chatgpt.com "MediaTek Ups the Arms Race in Mobile SoCs"))
        
- **Cons**
    
    - Not intended for datacenter training; memory capacity/bandwidth and feature set are **mobile-oriented**. ([Qualcomm Documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-88500-4/147_HTA.html?utm_source=chatgpt.com "Qualcomm® Hexagon™ Tensor Accelerator"))
        

# 6) What to remember

- **Lineage:** DianNao → DaDianNao/ShiDianNao/PuDianNao research (ICT-CAS) prioritized on-chip dataflow and specialization. **Cambricon** turned that into a programmable NN ISA + MLU products (with NeuWare). **Da Vinci** is Huawei’s _separate_ in-house evolution used in Ascend and newer Kirin NPUs (after using Cambricon IP in Kirin 970/980). ([BPB](https://bpb-us-w2.wpmucdn.com/sites.coecis.cornell.edu/dist/7/587/files/2023/06/Du_2015_ShiDianNao_v2.pdf?utm_source=chatgpt.com "ShiDianNao: Shifting Vision Processing Closer to the Sensor"), [LET'S Configure Deep Learning](https://reconfigdeeplearning.wordpress.com/wp-content/uploads/2017/02/2016-isca_cambricon-an-instruction-set-architecture-for-neural-networks_cyj.pdf?utm_source=chatgpt.com "Cambricon: An Instruction Set Architecture for Neural ..."), [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.06663?utm_source=chatgpt.com "AI Benchmark: All About Deep Learning on Smartphones in ..."))
    
- **Positioning:** Cambricon MLU targets **inference efficiency** in servers/edge; NVIDIA dominates **training at scale**; Qualcomm/MediaTek dominate **on-device** AI.
    

---

If you want, I can turn this into a one-pager with a block diagram timeline (DianNao → Cambricon ISA/MLU → Huawei Da Vinci) plus a quick decision tree for **when to pick MLU vs. GPU vs. mobile NPU** for your workloads.