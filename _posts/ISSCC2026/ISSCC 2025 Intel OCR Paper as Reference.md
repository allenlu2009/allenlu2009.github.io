Here’s the **complete, clean, and searchable Markdown version** of the OCR’d text from all your uploaded pages — covering **Sections 1.0 through 9.0** of _“AI Era Innovation Matrix”_ by Navid Shahriaf (Intel Chandler AZ, ISSCC 2025):

---

# AI Era Innovation Matrix

**Navid Shahriaf**  
_Senior Vice President, Foundry Technology Development, Intel, Chandler, AZ_

AI holds transformative potential for humanity, enhancing our ability to solve complex problems with speed and accuracy, and unlocking new realms of innovation and advancement. With the highest integration of AI, unprecedented in history, necessitating the co-optimization from the system level, from software and system architecture to cloud-based infrastructure in the communication network front, in silicon, packaging, and manufacturing, AI is reshaping the innovation frontier in all technologies that empower the industry today. This remarkable progress at every level, from chips to systems, marks the beginning of a new AI era.

---
## 1.0 Introduction

The rapid expansion of Artificial Intelligence (AI) is pushing traditional compute technology and infrastructure to fundamental and energy-efficiency limits for exponential scaling of computing systems. The compute industry must therefore transform across every level of the system hierarchy, including hardware, connectivity, high-performance infrastructure, and data centers. This paper emphasizes the system approach to co-optimization across all sectors. In 1.1, from software and system architecture to silicon and advanced packaging, AI innovation must be enabled by continued technology advancement, new materials, power, and cost. Strong ecosystem and faster time to market, setting the foundation for AI’s transformative potential.

---

## 2.0 Silicon

Silicon scaling has been a fundamental driver of progress in the semiconductor industry and a key enabler of modern technology innovation. The silicon roadmap is enabled by non-incremental transistor and interconnect architectural advances, along with High NA EUV lithography and incremental transistor and interconnect technical advances. The feature scaling and improvements for process nodes are measured using Design Technology Co-Optimization (DTCO) methodologies and generate various physical design goals for logic, memory, and analog/mixed-signal. Process-architecture area (PPA), and scaling benefits. Continued innovation in materials and process technology is essential to achieve continued silicon scaling benefits.

---

### 2.1 RibbonFET

RibbonFET, a gate-all-around transistor, advances beyond FinFET architecture, offers performance scaling and workload efficiency with the same technology base. Its diverse performance benefits and increased efficiency extends Moore’s Law.

---

### 2.2 PowerVia

PowerVia (Power Via) is a high-yielding backside power delivery technology, integrates power delivery to the transistor, reducing IR drop by 30% and providing improved power efficiency and routing. It meets all JEDEC combination standards and test requirements with leading-zero failures and shows over 5% frequency benefit in silicon. Intel 18A, Intel’s leading-process node, will offer a first-time combination of RibbonFET and PowerVia technologies.

---

### 2.3 The High NA EUV Advantage

High NA EUV enables flexible design rules, reducing parasitic capacitance and enhancing performance. It simplifies aspects of Electronic Design Automation (EDA) by reducing the dependency and the need for multi-patterning. Intel 14A front-end interconnects are optimized for High NA single-exposure patterning, improving yield and reliability.

---

### 2.4 Empowering AI with High NA EUV for Full-Field Large Die AI Applications

High NA EUV tools have a smaller imaging field size, but Intel has developed solutions for stitching and scaling across boundaries. The EDA ecosystem is creating tools to support this, and the mask ecosystem is working toward full field size capability without yield drifting. Figure 1.1.3, increasing foundry by 23–50%.

---

### 2.5 Enhancing the High NA EUV Advantage with AI and Curvilinear Mask Solutions

High NA EUV lithography requires advanced design and manufacturing solutions. Intel uses AI and Machine Learning to achieve accurately while managing computational costs. Curvilinear masks improve pattern space utilization, process window, and significantly reduce variability.

---

## 3.0 3D Integrated Circuits (3DIC), Packaging, and Assembly

As data processing demand grows, achieving more computing power in a smaller area with less power consumption via vertical stacking becomes essential. The base die on an advanced node is critical for enabling Through Silicon Vias (TSV) and advanced interfaces, integrating 3D elements seamlessly.

On-package vertical and lateral interconnects must continue to scale, providing increased interconnect density or bandwidth growth and improving power efficiency. (Figure 1.1.6). UCie (Universal Chiplet Interconnect Express) enables a common interface with die-to-die and die-to-package communication standards, combining the use of standardized-based protocols and custom extensions. Sizing, scaling, and complex system-level integration diversity and customization, matching use of glass to scale package substrate and redistribution layers for fine-pitch routing (Figure 1.1.5), is an important technology vector.

The increasing power demanded by AI applications must be addressed by improving system power delivery performance (described later) and expanding the thermal envelope through component and system-level innovation (Figure 1.1.6).

Advanced packaging technologies are evolving in a manner where the boundary between silicon and silicon backend interconnections is increasingly blurred as fine-feature interposers approach manufacturing process accuracy. Additionally, the package becomes a mechanical stress and structural process vector. Manufacturing and test processes must evolve to ensure that yield stays high (Figure 1.1.7).

A modular design environment that allows for straightforward assembly of multi-Si, co-optimized system optimization, cost performance, and bandwidth is critical. Comprehensive EDA tool co-design and analysis are needed for design partitioning across dies, enabling system-level and mechanical stress modeling, leading to potential failures and redesign efforts that impact time to market. 3DIC Design Technology Co-Optimization (DTCO) requires integration, extraction, reliability, and verification to ensure seamless integration (Figure 1.1.8).

---

## 4.0 Interconnect

The exponential scaling of parallel AI workloads is putting pressure on interconnect bandwidth density, latency, and power. All three of these metrics are improved by better integration of components within dense 2.5D and 3D assembly technologies, as described in Section 3. New packaging techniques provide better total cost of ownership (TCO) by narrowing the distance between components (like GPUs, NPUs, and power interposers). The growing demand in each bit of data scales as a function of the channel loss. This tradeoff has driven the definition of industry specifications like UCie for low-power, high-density in-package communication. UCie enables up to 1.35 Tb/s per millimeter of die perimeter at <1 pJ/bit.

Longer interconnects within the board and rack, which constitute the long-haul portion of domain in a scale-up network topology, require increasing data serialization to account for the dramatic data rates already scaling by a factor of 2× every 3–4 years, including industry standards (Ethernet, PCIe, and OIF-CEI). The latest production wireline SerDes has achieved 212 Gb/s PAM4 to support within-rack or off-rack interconnect communication at 4 pJ/bit. The energy-per-bit for training circuits and digital equalization both continue to constrain process technology scaling. Figure 1.1.9 shows measured TX and RX eye diagrams for a 212 Gb/s SerDes operating on Intel 18A over a 40 dB channel.

As wireline interconnect data rates continue to scale up, the distance that can be bridged between SerDes elements reduces because of higher channel loss at higher symbol rates. Adding more retimers reduces loss, but adds power, latency, and cost. This empirical tradeoff has led to the adoption of optical interconnects across a range of applications, from data centers to high-performance networks. In addition, extending the reach of high-bandwidth domain beyond the rack with optics aligns with the scale-up network model for AI. Therefore, optical interconnects need to move into the rack to scale bandwidth, area, and thermal at an acceptable power envelope. Technologies like packaged optical interconnects (POCI) are being developed to make this transition. Intel recently demonstrated a 4 Tb/s (8 fibers × 8 wavelengths × 32 Gb/s/wavelength per direction) bidirectional fully integrated Optical Compute Interconnect (OCI) chiplet based on Intel’s silicon photonics technology (Figure 1.1.10) and 224 Gb/s PAM4 over 23 km fiber with direct drive linear optics (Figure 1.1.11). An industry-wide effort to co-authorize in-rack optical interconnect ecosystem is underway, developing system manufacturing processes, materials, and equipment while improving bandwidth density, power, reliability, and testability.

---

## 5.0 Power Delivery

Per-package power for parallel workloads like AI is scaling up rapidly (Figure 1.1.12). A common approach to powering power to the package is a motherboard voltage regulator (MBVR) (Figure 1.1.14). The MBVR regulates the board-level power supply (e.g., 12 V) down to the voltage used by the die package (V out). Whether positioned next to the package (lateral MBVR) or under the package (vertical MBVR), the current density provided by MBVRs will not keep pace with huge high-performance chips. Furthermore, regulator  Efficiency degrades with higher power and current (IR loss), costing system performance (Figure 1.1.13). Solutions are needed that bring the voltage-conversion closer to the die with high-current density and fine-grained regulation innovation.

One solution uses the fully integrated voltage regulators (FIVRs) that bring the regulator within the package to reduce loss (Figure 1.1.14). Having a miniaturized high-frequency VR within the package minimizes IR drop and improves transient response for AI computing cores. This architecture has been proven in Intel’s previous client and server products and continues to evolve for AI and high-performance domains. This architecture further reduces board-level power delivery and simplifies design for dense multi-chip assemblies. An example implementation includes a CMOS-based stand-alone 2.4 W chiplet integrated with a high-density advanced high-definition multiphase buck voltage regulator (SGV) with a continuously scalable voltage conversion ratio [12].

Further evolutionary scaling of on-package power capacity beyond 1 kW will suffer from an unacceptable IR loss with traditional board-level MBVR architecture as illustrated in Figure 1.1.12. This problem can be mitigated by integrating high-voltage (12 V) to low-voltage (1 V) conversion directly in the package. By reducing the current delivered into the package, thereby reducing the power loss (heat), this approach supports a pair of high-voltage (IVR) switched-capacitor regulator (SCVR) on-package architectures (6 to 12 V→1.8–0.2 V) (1 VR) for two-step conversion (Figure 1.1.14). The key benefits and efficiency of this two-stage approach relies on dense on-package capacitor integration and novel power devices such as gallium-nitride (GaN). GaN can enable high-voltage converters with higher efficiency and density than traditional silicon power devices at comparable switching frequencies. However, it requires a higher switching frequency and can operate at higher voltages without breakdown or damaging devices. Fabricating GaN devices with silicon CMOS can open more options for on-package regulators and power delivery for AI systems, helping extend the design of CMOS energy and power FET on the same chip. To this end, Intel demonstrated a technology that combines GaN-on-Silicon technology highlighted on the same 300 mm wafer [13][14]. This technology can support high-voltage IVR options with an input voltage of up to 12 V to enable power scaling beyond 1–2 kW.

---

## 6.0 Architecture and Software

Next-generation compute architectures must drive exponential improvements in system performance metrics like performance-per-Watt-s mm³ while addressing thermal and power integrity challenges. Innovations should enable cohesive systems by stacking and co-optimizing 3D architecture and advanced packaging alongside silicon processes. Additionally, they must support the seamless integration of custom accelerators for various workloads [15].

Software, a crucial part of the innovation matrix, must advance through collaboration, standardization, and interoperability in open-source ecosystems. Automation should enhance security and streamline processes, while highly optimized software is essential for efficient silicon resource use. Distributing software across thousands of GPUs presents significant bandwidth and latency challenges, like high-performance computing. AI training will be driven by fine-tuning system environments, ensuring seamless integration and delivering remarkable advancements.

---

## 7.0 Looking Beyond Classical Computing

Technologies such as neuromorphic and quantum computing are critical to breakthroughs in scaling and speed needed to scale. Since 2018, Intel’s Loihi research chips, used by over 250 labs globally, have shown that neuromorphic chips manufactured with CMOS process technology can deliver orders-of-magnitude gains to a broad range of neural algorithms and applications [16]. While many of these examples relate to now brain-inspired algorithms that are currently not compatible with today’s software and hardware, an emerging class of techniques shows that 1000× gain will be achieved by emulating how the brain learns and adapts based on spike-based learning [17][18]. These neuromorphic innovations may be essential for intelligent systems with “always-on” capabilities in low-power, latency-, and data-constrained intelligent devices operating in real-time settings.

Quantum computing represents a new paradigm that harnesses the power of quantum physics to solve complex problems exponentially faster than conventional compute. It will transform precision industries and solve critical problems including digital security; chemical engineering; drug design and discovery; finance; and aerospace design. Making these benefits a reality requires transformative technology from the lab to fab and domain of engineering to deliver solutions for useful, real-term, error-tolerant systems. A quantum approach to semiconductors such as full-stack integration is critical. Intel is uniquely poised to advance quantum research paths from qubit to system, including qubit manufacturing [19], cryogenic-CMOS technologies for qubit control [20], software, compilers, algorithms, and applications.  With more than fifty years of experience in transistor manufacturing at scale, Intel is utilizing its proven technology to develop silicon spin qubits as the optimal path forward for quantum computing scalability [21].  Intel is also investing in capabilities like custom-designed cryoprobers that dramatically speed up Intel quantum testing and validation workflows [19].

The current state of quantum computing hardware does not yet have the robustness and scale to have direct impact on AI today. Another challenge for AI with quantum computing is the sheer large amounts of data in these complex machines. However, there are clear benefits once we have a scalable fault-tolerant quantum computer. Quantum computers can perform complex computations faster than classical computers and thus could enable faster training and analysis of AI models. Two of the key principles of quantum computing are superposition and entanglement, which enable exploration of multiple states at the same time, and this could directly benefit training and optimization of AI models. The possibility of analyzing large amounts of data in parallel can accelerate the ability for AI to recognize larger and more complex images or speech. Instead of using classical AI algorithms, new AI algorithms will be natively optimized to leverage quantum properties for classical development.

Finally, quantum computers should not be seen as a replacement for classical systems, but rather as compute accelerators for special applications. Therefore, the synergistic solution for AI in the future will likely leverage a hybrid implementation of classical and quantum computing.

---

## 8.0 Ecosystem Collaboration

Rapidly developing the next advanced compute systems will require collaboration on this innovation matrix across the ecosystem. Engaging with end users and partners across the technology stack—from manufacturing to design tools and IP to system design and software—is essential. The improvement process is aligned with market needs and timelines, environmentally sustainable, and leveraging key learnings and experiences from across the ecosystem. System-level co-optimization requires close collaboration to achieve rapid progress. Interdisciplinary expertise and knowledge sharing across strategic partners ensures the most efficient problem-solving and accelerated development cycles. Leveraging cross-industry standards and avoiding duplication of efforts will enable teams to work more effectively.

---

## 9.0 Industry Challenges & Opportunities

Nearly twenty years ago, CPU clock-frequency scaling faced a dilemma—the continued pursuit of exponential performance improvements hit a wall in terms of power density. The outcome of that era was a new set of parallel processor architectures along with an array of supporting technologies for silicon, packaging and thermals, interconnect, power delivery, and core architectures. Today, we are in a similar situation where exponential AI performance growth, this time in support of AI, is running into fundamental challenges for power, connectivity, and cost. Once again, incremental scaling of our systems will not be enough, and we will require new approaches to solve this problem—an innovation matrix. There is no shortage of engineering challenges to take on, from process technology scaling to 3DIC system design to power delivery, interconnect, and core architecture. We will need the combined benefits of innovation across these areas to meet the industry demand for compute power in a manufacturable, sustainable, and cost-effective way.

