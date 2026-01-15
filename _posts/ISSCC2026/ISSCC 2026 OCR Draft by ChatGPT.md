
**Advancing Horizons for AI: Perspectives on Semiconductor Innovations**  

**Abstract**  
AI is enhancing all aspects of our lives. Emerging autonomous AI further accelerates the extreme growth of computing and communication capability requiring unimaginable energy consumption. A shift from improving individual semiconductor components to Design/System/Application-Technology Co-Optimization is fundamental for future silicon chips. We provide a glimpse of what future IC/SoC breakthroughs may look like including high energy efficiency, open ecosystems and strategic partnerships.

**1.0 The growth of AI, unleashing opportunities and the IC ecosystem.**  
The pervasive integration of AI across every industry and aspect of daily life is fundamentally enabled by the semiconductor sector, which is projected to reach a trillion-dollar valuation by 2030 (Proffet, 2025) (SIA, 2025) (Buturac, Dragan, & Lehmann, 2022) (SRC, 2025) as shown in Fig. 1. This co-evolution is forcing a complete redesign of the entire computational stack, from foundational hardware like advanced packaging (Hung C.-M., 2023) and high-bandwidth memory (HBM) (Song, 2025) to the distributed compute fabric spanning cloud data centers and the robot edge (Loh K.-H. L., 2020) (Su & Nafziger, 2023) (Park & Park, 2024) (Shehariari, 2025). As these hardware and software technologies converge into a transformative, intelligent ecosystem (Tan, 2024), new bottlenecks continually emerge, presenting phenomenal engineering challenges that must be overcome to sustain the next wave of innovation.

AI’s demand for computing power, particularly in data centers, is exploding, outpacing Moore’s Law. While the AI’s demand used to double every two years, it now doubles every four to five months. Training compute for top-tier AI models is growing 4–5 times per year, as shown in Fig. 2. High-performance infrastructure GPUs implementing several architectural improvements have delivered a 100 times breakthrough in energy efficiency and a 1000 times boost in throughput over a decade (Dally, 2023). But this pace of progress can’t be sustained by improving silicon design in isolation and established IC technologies as future AI systems will scale from a billion to a billion+ more devices (Daly, 2023) (Liang, 2025).

Meeting this challenge requires a fundamental shift in engineering strategy. The solution isn’t limited to better chips; it’s a comprehensive co-evolution of hardware, software, and algorithms all optimized at the system level.

Cloud providers demand different classes of Domain-Specific processors and accelerators (Xu & Ramakrishnan, 2024) (Coburn & al., 2025) (Smith & al., 2024) (Prabhakar, 2024), high-bandwidth memory (HBM), and a combination with other conventional counterparts to be integrated together through advanced heterogeneous packaging (Hi) (Yang & Hung, 2025). It’s an engineering challenge also confronting an unsustainable energy consumption trend. Data centers’ electricity use has already grown by about 12% annually over the past five years as shown in Fig. 3. If this trend continues, the global data-center electricity use is set to more than double to ~945 TWh by 2030 (IEA, 2025) (BloombergNEF, 2025) as shown in Fig. 4. AI’s electricity demand alone could reach around 4.4% of the world’s total supply by 2035 (SRC, 2025), as shown in Fig. 5. This reality makes energy efficiency not just an important aspect, but the single most significant semiconductor engineering challenge (Park & Chang, 2025).

From a capital investment perspective, the scale of AI infrastructure growth is staggering: Projections show a cumulative $6.7 trillion in data-center capital expenditure (capex) by 2030, with the vast majority of it, over 85%, still being AI-oriented (Proffet, 2025) (SIA, 2025). Independent analysts further anticipate cumulative electricity spending 2× this huge capex by 2040. The economic gravity is speeding directly fuels innovation across the entire hardware stack, including AI accelerators, advanced packaging, ultra-high-speed networking (ranging beyond 1.6 T opt ics), advanced memory, and power solutions (Hung & C.-M., 2023). These areas hold the highest leverage for AI semiconductor breakthroughs, representing both today’s limitations and tomorrow’s renewable innovation value. The core challenge is to improve critical metrics like efficiency, latency, and bandwidth.

The next frontier for AI isn’t just in the cloud; it’s moving directly to edge devices as shown in Fig. 6 and 7. On-device AI is scaling fast for three key reasons: it offloads some of the cloud’s computational load (real experience), inference latency, and data privacy—all of which are driving a shift in focus from vastly “energy-limited” server experience (UX). The market for this segment is expected to grow dramatically. GenAI-capable smartphones are expected to hit ~730 million annual units by 2028 and all AI PCs are forecast to reach 100 million units by 2027 (Dell’Oro, 2025), with automotive neural network-driven solutions (NPUs) becoming standard features across all vehicle tiers.

This shift will enable trillions of local AI inferences every day, empowering everything from vision and personal copilots that seamlessly sync with cloud data when needed (MediaTek, 2025) to generative content creation, task planning, and intelligent control. This movement into personal and large-scale intelligence, reaching from handhelds to cars and homes, forms the basis of a new AI ecosystem that’s decentralized yet interconnected, as illustrated in Fig. 8. The cloud no longer owns all intelligence; now it’s just one node in a broader, federated AI continuum (Moore, 2025).

Additionally, integration of AI in physical devices, such as robots, autonomous vehicles, or smart machines, often referred to as Physical AI (Rus, 2025), is enabling them to act on their surroundings using sensors and motor skills, process real-world data, make decisions, and adapt their behavior in real time. This results in intelligent machines that can learn from experience, perform complex tasks, and navigate unpredictable environments in fields such as robotics, healthcare, and manufacturing (Chae, Lee, Jang, Hong, & Park, 2023).

Overall, the path forward is through cross-layer co-design: a holistic strategy where algorithms, software, and hardware architecture are built together from the ground up, rather than separately. It’s about optimizing the entire system cohesively (Liang, 2025) (Manganaro, 2024), resulting in a balanced fabric: global intelligence in the cloud, low-latency adaptation at the edge, and private, instantaneous response on personal devices (Loh K.-H. L., 2020) (Su & Nafziger, 2023) (Loh K.-H. L., 2025).

Once again, semiconductors are central to such a significant technological build-out with Artificial Intelligence serving as the primary catalyst for system-level innovation and investment. The engineering challenge is significant, but the payoff is tangible: a trustworthy assistant for every worker and in every robot, with latency, privacy, and cost steadily improving as the cloud-to-edge fabric matures. This is not a distant vision; it is a healthy, hopeful technology cycle that is already well underway (Nailampu, 2025) (Tsai, 2025).

---

### 2.0 Integrating AI into applications, enriching life through technology innovation

Two contemporary overview examples are discussed in this section, providing a glimpse not only at the application benefits but also at the distinct engineering challenges emerging as AI takes a central role in information and communication technology. Previously untied high-level services such as task planning with physical AI and computational load and its energy demand, as well as embedding complex intelligence deeply within edge, are elaborated in the context of bridging technology to assisting and enriching everyone’s life.

---

#### 2.1 Agentic AI and Physical AI augmenting human intelligence

AI isn’t just something that happens behind the scenes in massive cloud data centers; it also increasingly stays with you, from the things you carry to the things that carry you. The vast variety of AI advancements made so far — from generative models to multimodal agents — are reshaping daily life. Agentic AI and Physical AI are complementary pathways to augmenting human capability. For instance, from increasing autonomous intelligence in vehicles to robots acting as real-time companions, these advances such as tactile-rich physical AI performing adaptive human-machine interface (HMI) in the real world merge model physical AI supports emotional control, actuation and care for elder people (Hung & al., 2019), as well as hybrid intelligence that blends cognitive and physical skills.

With the emergence of such multi-agentic systems, the boundary between AI reasoning and embodiment is becoming blurred. As AI blends with non-physical AI, i.e., robot and apps, respectively, in human life, the form becomes more social and emotional. This transformation continues with multimodal understanding, context comprehension upon specific use cases and operational feedback, leading to systems that can perceive, interpret, and act. For example, AI copilots that converse with occupants showing natural and appropriate emotions, going beyond single-turn voice commands.


OEM pilots that embed large-model assistants into In-Vehicle Infotainment (IVI) stacks (e.g., Mercedes-Benz integrating ChatGPT via Azure OpenAI in MB’s UX) show the near-term trajectory toward richer in-car agents. Physical AI in vehicle is the embodied counterpart: from perception (from a rapidly growing number of cameras, LiDAR, to all types of sensors) to planning to control loops that actuate steering, braking, powertrain, and comfort functions under deterministic timing and functional safety governed by ISO 26262. Further strengthened by cybersecurity (ISO/SAE 21434), OTA/software-update governance (UNECE R155/R156), safety of the intended function (SOTIF), and the overarching framework of IEC 61508.

Projection of navigation commands and traffic-aware Advanced Driver Assistance System (ADAS) cues into the driver’s view (e.g., Continental’s AR-HUD waveguide systems), crash avoidance, Drowsiness and Attention Warning (DAW) and in-cabin monitoring of wellness will be a safety baseline. Managing conversation length between the agent and the driver, visual load, and AI core density are critical factors.

To support adequate data traffic within the vehicle’s units, achieving multi-hundred GB/p bandwidth with automotive-grade 3D packaging for performance, energy efficiency, cost, and reliability while adhering to deterministic data will require new functional safety design challenges before mass deployment. To support a cloud-centric paradigm, a centralized training and distributed-deployment methodology can simultaneously reduce communication data bandwidth, lower compute latency and improve scalability. The local-handling of features, inference caching and spatio-temporal memory management, integrated data deployment, bandwidth-aware mapping, and system coherence compression and quantization of features, and life-long learning of edge devices can substantially reduce required data traffic (Kim, Phinyavatapan, Kim, Saad, & Cain, 2025).

Physical AI delivering safe, precise assistance pushes compute to the edge under bounded-latency constraints. The IEEE Time-Sensitive Networking (TSN) suite provides time synchronization (802.1AS) and reliability (802.1CB) for real-time automotive Ethernet, which has become the backbone of zonal vehicle design. The ISO 26262 for E/E functional safety and ISO 21434 regulate this local deterministic network. Future vehicle generations (2030+) will integrate RISC-V-based domain controllers and compute clusters running hybrid CPUs, GPUs, and NPUs across mixed-criticality zones. Redundant, fail-operational runtime monitors. These measures together make local AI accelerators for perception and decision an essential complement to cloud offload.

Meanwhile, cars have evolved far beyond simple transportation. They’re now fully connected mobile environments for work, business, and entertainment. Today’s vehicles are packed with high-performance multimedia systems that offer real-time online interaction and conferencing, immersive entertainment and gaming, seamless connectivity through both terrestrial (5G, 6G) and non-terrestrial networks (NTN) (Fu et al., 2023). This includes the car’s own system as well as the many independent mobile devices that passengers would bring with them into the car, all able to communicate and cooperate, as part of a local device cloud, directly among each other (Terny, Kim, Shanait, Hsu, & Cain, 2025). Delivering this smooth, integrated user experience is a perfect blend of Agentic AI and Physical AI (Shih & al., 2024) (Hsieh & al., 2025). This performance and hardware combination runs on a high-performance, energy-efficient computing architecture, powered by cutting-edge chips built using advanced manufacturing processes, following a smart system-level design as those previously discussed (Varma & al., 2024) (MediaTek, 2025) (Tsai, 2025).

Where current physical AI systems face significant security and privacy challenges, especially as they transition from digital agents to embodied platforms. Enforcement via human feedback (RLHF-trained models) meets early reward hacking, social misalignment, and power-seeking behavior (Ng, 2024). Strengthening governance of training data is essential to cover privacy, consent, and transparency (Kang & al., 2025). Standardized and applied AI development methods, such as the TRI Risk Management Framework (RMF) provide the tools, defining, assessing, and managing AI risks; the EU AI Act as well as ISO/IEC 42001 offers guidance for organizations to structure policies and controls to manage AI. To further bridge the security gap, especially—

Act sets legal requirements for deploying AI in the EU, and ISO/IEC 42001 offers guidance for organizations to structure policies and controls to manage AI. To further bridge the security gap, existing frameworks must be extended beyond digital risk management to include cyber-physical safeguards.

---

### 2.2 Data centers expansion, and the transition from IC design to Multiphysics Design/System-Technology Co-Optimization (DTCO and STCO)

As mentioned, multiple inter-related themes emerge in the context of data centers, requiring DTCO and STCO of distinct engineering disciplines to tackle the growing demands for energy and the physical limits of materials, thermal management, the complexity of network connectivity, security and resilience. In this area, MediaTek is enabling the rapid infrastructure compute growth (“overwhelming compute”) by solving difficult technical obstacles with multiple leading technologies and through key technology partnerships (NVLink, 2025), reducing system power, total cost of ownership, while increasing performance (Nailampu, 2025) (Hu, 2025).

The large electrical power consumed by high-density servers leads to difficult thermal management challenges, pushing traditional air-cooling technologies past their physical limits, and leading to much more effective liquid-based solutions. These include direct-to-chip (DLC) cooling, where a coolant, typically water, circulates through cold plates that extract heat directly from hot components (such as CPUs/GPUs) or, even more among them in a 3-D heterogeneous structure for submersion cooling, where boards and components are fully immersed for passive and fan-free efficiency (Wu & al., 2025). Additionally, with the more aggressive immersion cooling, entire servers or other components are submerged into a tank filled with a dielectric fluid. Finally, to enhance efficiency and sustainability, AI-based adaptive thermal management is used to dynamically allocate cooling resources and optimize IT workload patterns for performance and energy savings (Zhang & al., 2025). These solutions are critical for maintaining real-time thermal management as the density and temperature rise of the AI load (Liu, Alpin, Song, & Hu, 2022).

At a “processor level,” heterogeneous integration of chiplets and multi-chip modules (MCMs) or system-in-package (SiP) technologies have become mainstream, while systems-on-wafer (SoW) are emerging. 3-D heterogeneous electrical and optical interconnects including fine-pitch through-silicon vias (3DICs), custom HBM interposer/substrate aim to minimize resistive copper losses and communication between the chips. Within the data center hierarchy, a server’s distributed architecture is evolving toward a “Rack-Scale Integration” (RSI) architecture that supports resource disaggregation, dynamically connecting resources following IEEE 2401 standards.

Beyond the data center, at a higher level, cloud disaggregation is evolving toward the “Data Center as a Die” concept, where compute, aggregation, and edge layers, suited for traditional client-server computing separation, are now serving in a unified disaggregated structure (Zhou, 2025) (Montalbano, 2025). Modern AI workloads have shown strong needs to 448Gb/s serial networking requiring DSP-based serial links beyond 200Gb/s have demonstrated (Chen & al., 2025) (Motoki & al., 2025), which pushes the interconnect and memory bandwidth even further through silicon-photonics interconnects and co-packaged optics (CPO) or energy-efficient photonic chiplets integrated with advanced materials. As CMOS scaling approaches physical and economic limits, new heterogeneous integration combining photonics, memory, and compute becomes an essential vector for future AI scaling (Yang & Hung, 2023).

Another approach to reduce total energy use is the co-optimization of packaging, power delivery, and system design, dynamically varying and fine-grained power requirements impact power management and distribution similarly to the above thermal management. At board level, a hierarchical power management structure, starting at a high voltage (e.g., 48 to 54V) distributes through the servers, where it then DC/DC converted down to 12V by intermediate bus converters (IBC). In turn the IBCs distribute supplies to smart power stages (SPS) on the backside of the packages. While at substrate and die level, complex power delivery networks (PDNs) are developed to manage varying loads and voltage drops in 3DICs by through-silicon vias (TSVs) and backside power delivery, coupled with dedicated integrated voltage regulation (IVR) and possibly power gating when units are temporarily powered down) for the individual processing units and interface circuitry, hierarchically fed to top level supply sources (Veloso et al., 2023) (Loh K.-H., 2025) (Zhang, 2024) (Shahriari, 2025) as shown in Fig. 8.

Finally, in a data-based economy, the resilience of data centers is critical. It requires built-in physical security. Zero Trust architectures are widely being adopted integrating physical, cyber, and operational controls.

---

### 3.0 Enabling technologies

As we look closer at the underlying enabling technologies it is possible to identify some key goals and trends that make this vision an engineering reality and to draw a path for a future development.

Process technology scaling is expected to continue to be an important component of computing and communication growth. However, performance scaling will require an optimal blend of strategy between traditional device miniaturization and multi-chip heterogeneous integration (HI). Device miniaturization involves shrinking transistors on individual dies, supported by new materials (IMEC, 2025), advanced transistor architectures like eFETs, HI assembles multiple chiplets using 2.5D, 3D, and 3.5D integration. These chipsets combine potentially different wafers (organic, silicon or glass) and substrates to stack them vertically. Prominent examples include TSMC’s CoWoS™ and SoIC™. Intel’s EMIB™ and Foveros™, and substrate-less alternatives like Nvidia’s Chip-on-Wafer-on-Package (Jansen & al., 2025).

Stacked systems remain extremely data-dense and complex, demanding advances in design methodology, modeling, and simulation, where DTCO/STCO co-design frameworks are required to address cross-domain optimization challenges across mechanical, electrical, and thermal boundaries. The physical constraints on interconnect distances, parasitic resistance/capacitance (RC) delay, and bandwidth efficiency motivate the emergence of new EDA tools that support multi-physics analysis of these tradeoffs.

Demand for high-speed data interfaces from cloud to edge devices continues to rise, and overcoming their massive bandwidth requirements and signal integrity issues demands new solutions. MediaTek leads in high-speed data interfaces through the co-design of circuits, systems, and signal processing. By employing higher-order modulation schemes like PAM4, bandwidth efficiency of copper channels is maximized (Chen & al., 2025). For long-reach, ultra-high data rate links, however, optical solutions are a superior alternative due to the vast bandwidth and physical channel isolation of fiber. Established solutions like Active Optical Cables (AOCs) offer high bandwidth and lower energy consumption for offboard connections. Pushing this trend further, emerging technologies are integrating optical links closer to the silicon processing units to reduce energy and latency. These include near-package optics (NPO) and Co-Packaged Optics (CPO). In the latter, MediaTek has demonstrated excellent results in CPO, showcasing its leadership in next-generation interconnects (Trendforce, 2024).

As wireless communication transitions from 5G to 6G (Varma & al., 2024), MediaTek has focused on three imperatives: achieving ultra-low power (ULP) consumption, embedding AI-native intelligence, and enabling ubiquitous coverage. Among these, energy efficiency is essential for managing the thermal constraints and battery life in handheld systems, which in turn enable higher performance AI inference at the edge. Wearable and form factors for emerging materials like smart glasses, for example, have far more stringent limitations on size, weight, and heat dissipation than smartphones. This challenge is compounded by the fact that higher receiver sensitivity levels raise power consumption. To keep the average power within acceptable limits, systems must be significantly more energy proportional at low data rates, such as when the device is performing routine housekeeping tasks (e.g., control channel monitoring, beam management, or idle paging).

A defining feature of 6G will be its AI-native architecture, enabling agentic AI on edge devices to collaborate seamlessly across the entire network, from the device to the network edge and the cloud, as shown in Fig. 9. A good example of edge AI application is in high-resolution video processing, effectively reducing the large data volume streamed without loss of quality. This allows one to intelligently optimize the trade-off between bandwidth loads and the local energy required for computation. To handle these demanding workloads, energy efficiency and throughput of AI accelerators with adaptable Digital Compute-in-Memory (DCIM) engines have proven highly effective. Built on the latest lithography nodes, their architectures are optimized to operate across both generic and model-specific tasks like weight switching. This approach achieves synergistic energy efficiency by unifying AI tasks across device nodes (Hsieh & al., 2025).

Looking at the future, innovative approaches to improve edge computing for physical AI at a higher abstraction level with massive data creation are explored. For instance, introducing a cognitive architecture, such as the human brain’s dual-cognition system architecture, can help self-organize and adaptively manage different levels of perception and planning both consciously and unconsciously, while system two is responsible for complex problem-solving (Kahneman, 2013). A working-memory-based conscious planning approach has been proposed for robots (Yuan & al., 2025). To achieve brain-like energy efficiency, neuromorphic computing is also a promising path, linking key inferencing tasks (e.g., high-speed sensor integration) and enhancing broad sensor abstraction for programming neuromorphic hardware, along with proper compilation to mapping spiking neural networks (SNNs) to hardware architectures. A general modern standard for compatibility should be considered when designing a neuromorphic system (Kuhdjura & al., 2025).

Additionally, intelligent devices will be moving seamlessly between different networks (cellular, WiFi, satellite/NTN) without losing connection or experiencing service degradation. AI can predict when a user is about to move out of WiFi range and preemptively switch to cellular without user intervention. MediaTek’s intelligent device edge cloud (device and Radio Access Network (RAN)) concept leverages both data-plane and control-plane management, which improves collaboration and allows secure analytics crossing devices and network edge (device cloud). Support becomes possible for a new generation of applications that require ultra-high data rates, stringent latencies and intense computing and caching resources, without mandating massive processing, storage or power resources in the devices themselves.

MediaTek has taken a leading role in the system design, standardization and ongoing evolution of 5G NTN (satellite), both for NR NTN and IoT NTN, to 6G NTN as shown in Fig. 10. Integrating satellite and terrestrial mobile networks to offer pervasive connectivity across the world will not only enable a new era of innovative digital services, but also significantly contribute to the United Nations Sustainable Development Goals. Compared with proprietary satellite communication technologies, 6G non-terrestrial network (NTN) technology based on a 3GPP open standard can leverage the economies of scale from the existing global mobile cellular ecosystem to bring satellite communication from a niche market to mainstream consumer and business markets leveraging common devices that switch between satellite and cellular networks for an always connected user experience (Fu I.-K., 2025) (Yang & al., 2025).

---

### 4.0 Ecosystem and partnerships

Addressing large-scale technological challenges requires productive ecosystems that lower barriers to contribution and foster strategic partnerships. Historically, openness has been a powerful catalyst for innovation. In software, Linux became the backbone of the internet, while Android created a massive mobile ecosystem. In hardware, Tesla’s open patents spurred the growth of the EV industry, and in content, Wikipedia’s open license made it the world’s largest encyclopedia.

This principle of open innovation is a driving force in AI’s rapid ascent. Many foundational tools (like TensorFlow and PyTorch) and influential models (such as LLaMA and Mistral) are open source. Moreover, the industry would benefit from moving beyond the core open model weights and training data. Similarly, open systems and standardization will be even more critically important to design and build high performance systems in areas such as heterogeneous integration, wired and wireless communication and so on in order to accelerate the innovative solutions required to address the remaining technology bottlenecks described above. Looking further ahead, the AI landscape will likely consist of a strategic mix of open and closed approaches as players balance the speed of innovation with capturing commercial value.

To power this dynamic and hybrid AI ecosystem, MediaTek delivers the essential computing foundation, scaling from the edge to the cloud as shown in Fig. 11. We meet diverse needs, from small-scale on-device AI (MDLA) to large-scale acceleration in data center ASICs. Supported by flexible business models, a resilient supply chain, and the transformative power of wireless technologies like Wi-Fi and 6G, MediaTek is a pivotal partner in AI’s development and the hardware implementation that brings it into our daily lives.

---

### 5.0 Conclusions

The semiconductor industry is transitioning from an era of performance-driven scaling to an era of system-level, efficiency-driven innovation. The immense challenges posed by AI’s energy consumption and computational intensity are met in three major obstacles: the energy cost of training and inference, the massive distributed data centers to private, responsive AI on personal devices.

The future of compute and communication is not monolithic as before; it is heterogeneous and hierarchical, where disaggregated data centers to private, responsive AI on personal devices are connected through an open ecosystem. AI-driven design methodologies, new materials, and multi-physics co-optimization are unleashing a new generation of hardware that is both powerful and profoundly efficient.

Achieving this vision is contingent on a fundamental shift in industry mindset: from designing individual chips to architecting integrated systems, and from proprietary development to open, ecosystem-driven collaboration. The ultimate goal is to deliver a future of ubiquitous, trustworthy AI that enhances human productivity, safety, and daily life, marking the successful navigation of this critical technological inflection point.

---

## **推進人工智慧的前沿：半導體創新的觀點**

### **摘要**

AI 正在加速改善我們生活的方方面面。新興的 autonomous AI 進一步推動了運算與通訊能力的極端成長，同時伴隨著難以想像的能耗需求。未來的矽晶創新將不再僅是單一元件的改進，而是必須採取 Design/System/Application-Technology Co-Optimization 的整合性方法。本文將展望未來 IC/SoC 的突破方向，包括高能源效率、開放式生態系統與策略性夥伴關係。

---

### **1.0 AI 的成長，釋放機會與 IC 生態系統**

AI 在各行各業與日常生活中的普及應用，主要是由半導體產業所支撐。根據 (Proffet, 2025)、(SIA, 2025)、(Buturac, Dragan, & Lehmann, 2022)、(SRC, 2025)，半導體產業預計在 2030 年達到一兆美元規模，如圖 1 所示。此種共演化迫使整個運算堆疊重新設計，從基礎硬體如 advanced packaging (Hung C.-M., 2023)、high-bandwidth memory (HBM) (Song, 2025)，到橫跨雲端資料中心與 robot edge 的 distributed compute fabric (Loh K.-H. L., 2020) (Su & Nafziger, 2023) (Park & Park, 2024) (Shehariari, 2025)。隨著硬體與軟體技術融合成智慧生態系統 (Tan, 2024)，新的瓶頸不斷出現，形成推動下一波創新的工程挑戰。

AI 對運算能力的需求急遽上升，遠超過 Moore’s Law。過去 AI 的運算需求約每兩年翻倍，現在則是每四到五個月翻倍。頂尖 AI 模型的訓練運算需求每年成長 4–5 倍，如圖 2 所示。雖然高效能 GPU 透過架構改良在十年間達成 100 倍能效與 1000 倍效能提升 (Dally, 2023)，但單靠晶片設計已難以維持這樣的速度。未來 AI 系統的規模將從十億級擴展至更多 (Liang, 2025)。

解決之道不僅是更好的晶片，而是整體硬體、軟體與演算法的系統級共同演化。  
Cloud provider 需要不同類型的 Domain-Specific Processor、Accelerator、HBM，以及異質整合的 heterogeneous packaging (Yang & Hung, 2025)。同時，能源消耗問題日益嚴重，資料中心的耗電量近五年平均年增約 12%，如圖 3 所示。若此趨勢持續，至 2030 年全球資料中心耗電量將超過 945 TWh (IEA, 2025) (BloombergNEF, 2025)，如圖 4。AI 的電力需求到 2035 年可能佔全球電力總量的 4.4% (SRC, 2025)，如圖 5。能效因此成為半導體產業最關鍵的工程挑戰 (Park & Chang, 2025)。

從資本投入角度看，AI 基礎設施的擴張規模驚人，至 2030 年累計 Data Center CapEx 預估達 6.7 兆美元，其中超過 85% 為 AI 專用 (Proffet, 2025)。分析師預期至 2040 年電力支出將達此金額兩倍。此龐大經濟規模推動整個硬體堆疊創新，包括 AI Accelerator、Advanced Packaging、Ultra-High-Speed Networking (>1.6T optics)、Memory 與 Power Solutions (Hung & C.-M., 2023)。能效、延遲與頻寬將是關鍵指標。

AI 的下一個前沿不僅在雲端，也正快速移向 edge device（見圖 6、7）。On-device AI 崛起的三大原因為：減少雲端運算負載、降低 inference latency、強化資料隱私。此市場預計快速成長，GenAI-capable smartphone 於 2028 年將達 7.3 億台，AI PC 於 2027 年達 1 億台 (Dell’Oro, 2025)，automotive NPU 也將成為標準配備。

這將使每日產生數兆次 local inference，涵蓋 vision、personal copilot、generative content creation、task planning、intelligent control。此去中心化但互聯的 AI 生態系統 (Fig. 8) 將形成新的 computing paradigm，雲端只是其中一個節點 (Moore, 2025)。

進一步地，AI 的整合正延伸至 physical devices，如 robot、autonomous vehicle、smart machine 等，形成所謂的 Physical AI (Rus, 2025)。這些系統具備感知、決策與動作能力，能即時學習與適應環境，應用於 robotics、healthcare、manufacturing (Chae, Lee, Jang, Hong, & Park, 2023)。

未來發展將仰賴 cross-layer co-design 策略，從演算法、軟體、硬體共同設計出平衡的架構 (Liang, 2025) (Manganaro, 2024)：雲端負責 global intelligence、edge 提供低延遲適應、個人裝置負責即時私密回應 (Loh K.-H. L., 2025)。

半導體再次成為此科技革命的核心，AI 是推動系統級創新的主要催化劑。最終目標是讓 AI 成為可靠的助手，降低延遲、強化隱私與成本效率，並促成更智慧的人機共存循環 (Nailampu, 2025)。

---

### **2.0 AI 與應用的融合：以創新豐富生活**

本節提供兩個當代範例，展示 AI 如何在應用層面帶來價值與工程挑戰：一是將 AI 嵌入實體世界的 Physical AI；二是資料中心的多物理共設計 (DTCO/STCO) 如何支撐其運行。

---

#### **2.1 Agentic AI 與 Physical AI：增強人類智能**

AI 不僅存在於雲端資料中心，也無所不在於個人裝置與車輛。從 Generative Model 到 Multimodal Agent，AI 正改變日常體驗。Agentic AI 與 Physical AI 形成互補：前者增強認知推理，後者擴展行動與感知。例如，車用 AI Co-pilot、具觸覺反饋的 Physical AI 為長者提供照護 (Hung & al., 2019)，展現 Hybrid Intelligence 的應用。

多 Agent 系統的出現，使 reasoning 與 embodiment 的界線模糊。AI 正從純軟體代理延伸到社會性、情感性互動。例如，IVI (In-Vehicle Infotainment) 中的大型語言模型，如 Mercedes-Benz 透過 Azure OpenAI 將 ChatGPT 整合進 MB UX，展現即將到來的 in-car agent 趨勢。

Physical AI 在車輛中的體現包括感知（camera、LiDAR、sensor）、規劃與控制迴圈，受 ISO 26262、ISO/SAE 21434、UNECE R155/R156、SOTIF 與 IEC 61508 規範。未來車輛需多百 GB/s 帶寬，採用 automotive-grade 3D packaging，以兼顧能效、成本與可靠性。IEEE TSN (802.1AS、802.1CB) 與 zonal architecture 將成為 backbone。RISC-V Domain Controller、Hybrid CPU/GPU/NPU Cluster、Fail-Operational Runtime Monitor 也將成為主流。

同時，車輛成為行動辦公與娛樂空間。結合 5G/6G、NTN 通訊，乘客裝置間形成 local device cloud (Fu et al., 2023) (Terny et al., 2025)。這種互聯體驗正是 Agentic AI 與 Physical AI 的結合 (Shih & al., 2024) (Hsieh & al., 2025)。

然而，Physical AI 仍面臨安全與隱私風險，包括 reward hacking、social misalignment、power-seeking 行為 (Ng, 2024)。因此，AI 治理框架如 TRI Risk Management Framework、EU AI Act、ISO/IEC 42001，將是未來標準，以確保 cyber-physical safety。

---

#### **2.2 Data Center 擴張與 DTCO/STCO 的轉變**

Data Center 需跨學科整合 DTCO/STCO，以因應能耗、材料極限、網路複雜度與安全性挑戰。MediaTek 透過關鍵技術（如 NVLink, 2025）與夥伴合作，提升效能、降低 TCO (Nailampu, 2025)。

高密度伺服器造成嚴重熱管理挑戰。傳統風冷逐漸被液冷取代，包括 direct-to-chip (DLC)、submersion cooling (Wu & al., 2025)。AI-based thermal management 動態分配冷卻資源以節能 (Zhang & al., 2025)。

在 Processor 層面，chiplet-based MCM/SiP 成為主流，3DIC、HBM interposer、optical interconnect 成為能效關鍵。未來將邁向 Rack-Scale Integration (RSI) 與 Data Center-as-a-Die (Zhou, 2025)，並採用 448Gb/s serial link、Co-Packaged Optics (CPO) (Motoki & al., 2025)。

電源管理亦須層級化設計：從 48–54V bus → 12V IBC → smart power stage → backside PDN、TSV、IVR、Power Gating (Veloso et al., 2023) (Zhang, 2024)。資料中心的物理安全與 Zero Trust 架構將成為核心。

---

### **3.0 關鍵使能技術**

未來的成長將結合 device scaling 與 heterogeneous integration (HI)。先進製程（IMEC, 2025）、eFET、2.5D/3D/3.5D integration（如 TSMC CoWoS™, SoIC™, Intel EMIB™, Foveros™）將持續演進。  
EDA 工具需支援多物理 (multi-physics) 模擬與 DTCO/STCO 框架。

高速介面需求不斷提升，MediaTek 在高速數據介面透過 PAM4、Signal Processing、CPO 展現領導力 (Trendforce, 2024)。

隨 5G 邁向 6G (Varma & al., 2024)，重點為 Ultra-Low Power、AI-Native Intelligence、Ubiquitous Coverage。為維持 edge 裝置效能與續航，系統需具 Energy-Proportional 設計。

6G 將實現 AI-Native 架構，Agentic AI 能在 Edge 與 Cloud 間協作（Fig. 9）。  
AI Accelerator 採用 Digital Compute-in-Memory (DCIM) 架構，在多任務下達成能效最佳化 (Hsieh & al., 2025)。

進一步的 Physical AI 將採用 cognitive architecture（System 1/2 dual cognition），以 Working-Memory 為基礎之 Conscious Planning (Yuan & al., 2025)。Neuromorphic Computing 透過 Spiking Neural Network (SNN) 提供 brain-like 能效 (Kuhdjura & al., 2025)。

裝置可於 cellular、WiFi、NTN 間無縫切換。MediaTek 的 Intelligent Device Edge Cloud 結合 data-plane 與 control-plane，實現跨裝置安全協作。

MediaTek 亦主導 5G NTN/6G NTN 標準化 (Fu I.-K., 2025)，結合衛星與地面網路，實現全球普及連線並助力聯合國永續發展目標。

---

### **4.0 生態系統與夥伴關係**

面對龐大技術挑戰，開放式生態系統與策略夥伴合作至關重要。正如 Linux、Android、Tesla、Wikipedia 所示，開放是創新的催化劑。  
AI 的快速崛起同樣源於開放模式：TensorFlow、PyTorch、LLaMA、Mistral 等皆是典範。未來，高效能系統的建構需仰賴開放標準與異質整合架構。

MediaTek 構建從 Edge 到 Cloud 的完整 AI Computing Foundation（Fig. 11），從 on-device AI (MDLA) 到 data center ASIC，搭配彈性商業模式與強大供應鏈，成為推動 AI 實現的關鍵夥伴。

---

### **5.0 結論**

半導體產業正從「性能導向的縮放時代」邁入「系統導向的能效時代」。AI 帶來的挑戰包括訓練與推論的能耗、資料中心規模化，以及個人化的 Edge AI。

未來運算與通訊將是異質且分層的，結合 Data Center 與 Edge Device 形成開放生態。AI-Driven Design、New Materials、Multi-Physics Co-Optimization 將釋放新一代強大且高能效的硬體。

要實現這一願景，產業思維必須從「晶片設計」轉向「系統設計」，從「封閉開發」轉向「開放協作」。最終目標是構建一個普及、可信且能提升人類生產力與安全的 AI 世界，成功跨越這一技術分水嶺。

---
