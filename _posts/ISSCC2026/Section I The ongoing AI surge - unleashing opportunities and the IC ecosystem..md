

## **Section 1: The ongoing AI surge - unleashing opportunities and the IC ecosystem.

The pervasive integration of AI across every industry is fundamentally enabled by semiconductors, an industry projected to reach **one trillion USD** by 2030 (Proffet, 2025; SIA, 2025). Figure 1 illustrates this co-evolution between AI and IC technology, showing that the compute demand for frontier models now doubles roughly every **four to five months**, far outpacing Moore’s Law.

### **1.1 Training Compute Explosion**

AI training has grown **4–5× per year**, fueled by large-scale data and multi-trillion-parameter foundation models. High-performance GPUs and domain-specific accelerators have delivered ~100× energy-efficiency and ~1000× throughput improvement in the past decade (Dally, 2023). However, this gain cannot be sustained by transistor miniaturization alone. Future AI systems—spanning billions of devices—require co-optimization of architecture, memory hierarchy, packaging, and thermal design (Liang, 2025). Figure 2 depicts the exponential compute escalation for frontier training runs.

### **1.2 Energy and Economic Pressure**

The energy footprint of data centers is escalating sharply. Electricity consumption has risen ≈ 12 % annually during the past five years (IEA, 2025). If continued, total usage will exceed **945 TWh by 2030** (BloombergNEF, 2025), while AI alone could account for **≈ 4.4 %** of global generation by 2035 (SRC, 2025). Figure 3 summarizes the trend. Cumulative data-center capital expenditure is projected to reach **6.7 trillion USD by 2030**, > 85 % of which will target AI infrastructure (Proffet, 2025). Electricity spending may double that amount by 2040. Energy efficiency therefore emerges as the single most critical engineering constraint (Park & Chang, 2025).

### **1.3 From Cloud to Edge Intelligence**

AI is expanding beyond hyperscale servers. **On-device AI** offloads cloud workloads, lowers latency, and safeguards privacy. Gen-AI smartphones are expected to reach **≈ 730 million units by 2028**, AI PCs **≈ 100 million by 2027** (Dell’Oro, 2025), and automotive NPUs will become baseline features across all tiers. Figures 6 and 7 show this migration. This decentralized yet connected ecosystem forms a **federated AI continuum** (Loh K.-H. L., 2020), where the cloud contributes global cognition and the edge provides local, low-latency intelligence.

### **1.4 Toward Physical AI**

Integration of AI into robots, autonomous vehicles, and smart machines—termed **Physical AI** (Rus, 2025)—enables perception, motion, and adaptation in real time. Applications span robotics, healthcare, and manufacturing (Chae et al., 2023). These systems require deterministic compute, energy efficiency, and seamless communication across sensors, actuators, and cloud links.

The path forward demands **cross-layer co-design** in which algorithms, software, and hardware architectures are conceived together rather than sequentially (Manganaro, 2024). Semiconductors again stand at the center of this technological build-out, catalyzed by AI’s insatiable need for efficiency and scalability.

==============================
Modify the section I based on the following 

On-going AI surge include two waves.  
Wave 1 is led by AI foundation model providers like OpenAI's ChatGPT, Google's Gemini. The focus is on the training of foundation models.    Then lead to the section 1.1.

Wave 2 is driven by AI applications including content generation, searching, coding, personal agent.  The focus shifts from training to inferencing.   Quote Google IO, the annual token generation grows by 50x, and reached more than 1 quadrillion tokens per month inferencing in 2025.  


1.4 Toward Physical AI:  add more automotive AI including both cockpit AI and ADAS / autonomous drive.  add some market number like volume or market penetration. 


Add 1.5  computing scalability:  Data Center: 10000 TOPS,  Automotive AI: 1000 TOPS,  AI PC and AI smartphone 100 TOPS


# **Modified Version**

---

## **1.0 The Ongoing AI Surge — Unleashing Opportunities and the IC Ecosystem**

The pervasive integration of artificial intelligence (AI) across every industry is fundamentally enabled by semiconductors—an industry projected to exceed **one trillion USD** by 2030 (Proffet, 2025; SIA, 2025). Figure 1 illustrates this co-evolution between AI demand and integrated-circuit (IC) capability, revealing that compute requirements for frontier models now double roughly every **four to five months**, far outpacing Moore’s Law.

AI’s explosive growth unfolds in **two waves** that together define the modern semiconductor innovation landscape.

---

### **1.1 Wave 1 — Training Compute Explosion and Energy Pressure**

The first wave is led by AI foundation-model providers such as **OpenAI’s ChatGPT** and **Google’s Gemini**, where progress is driven primarily by **large-scale model training**. Training these frontier models requires tens of thousands of GPUs or domain-specific accelerators operating continuously for months with petabytes of high-bandwidth memory (HBM) and exabyte-scale datasets. Such workloads demand massive distributed computing infrastructure, advanced packaging, power delivery, and liquid cooling, pushing semiconductor systems to their physical and thermal limits.

AI training workloads have been growing **4–5× per year**, fueled by ever-larger multimodal and multi-trillion-parameter models. High-performance GPUs and domain-specific accelerators have achieved ~**100× energy-efficiency** and **1000× throughput** improvement over the past decade (Dally, 2023). However, this improvement can no longer rely solely on transistor miniaturization. Future AI systems—spanning billions of devices—require **co-optimization across architecture, memory hierarchy, packaging, and thermal design** (Liang, 2025). Figure 2 depicts the exponential compute escalation of frontier training runs.

The associated energy footprint has become a dominant engineering constraint. Data-center electricity consumption has risen ≈ **12 % annually** in the past five years (IEA, 2025) and is projected to exceed **945 TWh by 2030** (BloombergNEF, 2025), while AI workloads alone may consume **≈ 4.4 % of global generation** by 2035 (SRC, 2025). Cumulative data-center capex is forecast to reach **6.7 trillion USD by 2030**, with > 85 % directed toward AI infrastructure (Proffet, 2025); electricity spending may double that amount by 2040. This makes **energy efficiency the single most critical engineering challenge** (Park & Chang, 2025) and elevates the importance of multi-physics optimization—from device to system architecture to power and thermal management.

---

### **1.2 Wave 2 — AI Applications and Inference Expansion**

The second wave of the AI surge is driven by **AI applications** that deliver intelligence directly to end users. Examples include **content generation**, **AI-assisted search**, **software coding**, **personal agents**, and **contextual copilots** integrated into everyday devices and services. Here, the emphasis shifts from training to **inference**, where deployed models generate and evaluate tokens in real time.

According to Google I/O 2025, global token generation for AI inference has increased more than **50× year-on-year**, surpassing **one quadrillion (10¹⁵) tokens per month** in 2025. This astronomical scale transforms the compute and energy balance between cloud and edge: while the first wave concentrated compute power in a few hyperscale datacenters, the second wave distributes inference across **AI ASICs, inference servers, AI PCs, smartphones, and automotive SoCs**. As a result, **energy per token**—the cost of generating useful AI output—emerges as the new performance metric guiding semiconductor design.

---

### **1.3 From Cloud to Edge Intelligence**

AI is rapidly expanding from hyperscale servers toward distributed edge environments. **On-device AI** offloads portions of cloud workloads, lowers latency, and protects privacy. Generative-AI smartphones are projected to reach ≈ **730 million units per year by 2028**, **AI PCs ≈ 100 million by 2027** (Dell’Oro, 2025), and automotive AI SoCs are becoming baseline components across vehicle tiers. Figures 6 and 7 illustrate this migration. The outcome is a **federated AI continuum** (Loh K.-H. L., 2020) where cloud and edge co-operate seamlessly—global models provide cognition while edge devices deliver real-time, context-aware intelligence with strict energy budgets.

This decentralized topology also alleviates backbone-network congestion and enables continual model refinement through federated and personalized learning.

---

### **1.4 Toward Physical AI**

The integration of AI into **autonomous and semi-autonomous machines**—termed **Physical AI** (Rus, 2025)—extends digital intelligence into the physical world. In automotive systems, Physical AI spans both **cockpit AI** and **driving AI**. The cockpit domain employs large-language and vision-language models for natural dialog, personalized voice assistants, in-cabin monitoring, and adaptive infotainment. The driving domain covers **Advanced Driver-Assistance Systems (ADAS)** and full **autonomous-driving (AD)** stacks performing perception, fusion, planning, and control under deterministic latency and functional-safety constraints (ISO 26262; ISO/SAE 21434).

The automotive AI market is expanding rapidly—analysts project **> 200 million vehicles** equipped with AI-enhanced ADAS and cockpit systems by 2030, representing over **60 % market penetration** in new vehicles. These platforms require hundreds of TOPS of local compute, multiple high-speed sensor interfaces, and secure connectivity for continuous updates. Physical AI is also penetrating robotics, healthcare, and industrial automation domains, where real-time perception and actuation must operate within tight energy and reliability margins.

Such systems necessitate **deterministic performance**, **efficient data movement**, and **cross-domain communication** among sensors, processors, and actuators. This requires **cross-layer co-design**, in which algorithms, software, and hardware are developed synergistically (Manganaro, 2024). Semiconductors remain the foundation of this transition, driving AI from digital reasoning to physical action.

---

### **1.5 Computing Scalability Across Domains**

AI computation now spans six orders of magnitude in scale—from megawatt-class datacenters to milliwatt-class edge nodes—as summarized in Table 1 and Figure 8.

| **Domain**                      | **Example Systems / Workloads**                   | **Approximate Compute Envelope** |
| :------------------------------ | :------------------------------------------------ | :------------------------------- |
| **Cloud Training ASICs & GPUs** | Foundation-model training, multi-rack clusters    | **≈ 10 000 TOPS per node**       |
| **Automotive AI SoCs**          | ADAS / AD sensor fusion and planning              | **≈ 1 000 TOPS per platform**    |
| **AI PC / Workstation**         | Generative content creation, AI coding assistants | **≈ 100–500 TOPS**               |
| **AI Smartphone**               | Camera generation, voice agents, multimodal UX    | **≈ 50–100 TOPS (< 5 W)**        |
| **IoT / AIoT Nodes**            | Always-on sensing and local inference             | **≈ 1–10 TOPS (sub-Watt)**       |

This hierarchy highlights the need for **scalable, heterogeneous computing architectures** that share common software frameworks and interconnect standards while optimizing for divergent energy and thermal constraints. The central challenge for the next decade is no longer transistor count but **system-level efficiency**—achieving the optimal balance of performance, bandwidth, and energy per token throughout the cloud-to-edge AI continuum.

