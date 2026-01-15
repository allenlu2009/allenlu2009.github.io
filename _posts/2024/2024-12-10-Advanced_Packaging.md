---
title: Advanced Packaging
date: 2024-12-10 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-10-10-Attention_Math]]"
categories:
  - Science
---


## Introduction

Moore's law faces challenging in process.  Advanced packaging can stack chip/chiplet in both vertical direction and horizontal plan become the savior.

## Technologies and Benefits

Advanced packaging help to fulfill Moore's law with higher density by stacking or side-by-side more dies.   

The side benefit is: 
- smaller, lighter for mobile and wearable from system perspective
- higher throughputs (and potentially cost) by wafer and panel
- high speed because of closer distance.

The challenging are:
- alignment precision and TSV: 1 generation lags SoC
- heat 

- Denser IO pitch and higher IO pin counts:  
	- Wafer-Level Chip Scale Packaging (WLCSP or WLP) - Fan-In (FI-WLP)
		- Benefit: small but limited IOs, cheap (direct cut on wafer), high production throughput
		- Applications: RF, IoT, flash controller
	- Wafer-Level Packaging - Fan-Out (FO-WLP)
		- Benefit: small but with more IOs, high speed multiple-chips (on substrate - OS, or local Si interconnect - LSI).  就是 TSMC: InFO-SO and InFO-LSI.  
		- Applications: smartphone, wearable
	- Panel-Level Packaging - Fan-Out Only (PLP)
		- Benefit: 理論上和 FO-WLP 一樣，但是技術上還無法達到。High production throughput, lower cost
		- Applications: smartphone, wearable
	
- Vertical stacking to increase transistor density / smaller form factor and high speed because of shorter inter-die distance
	- POP (package-on-package, **no TSV on die or interposer**): mainly for DRAM, Flash
		- Benefit: simple stacking, often combine with WLP, e.g. InFO-POP
		- Applications: smartphone, wearable
	- 2.5D (with interposer and **TSV only on interposer**)
		- Benefit: side-by-side using interposer.  Interposer has 3D massive interconnect, can combine with WLP, e.g. TSMC CoWoS (Chip-on-Wafer on Substrate), Intel EMIB (CoGoS: Chip-on-Glass on Substrate), Samsung I-Cube
		- Applications: HPC, AI
	- 3D (**TSV on dies**, no interposer)
		- Benefit: smaller because stacking, high speed
		- Applications: HPC, AI, e.g. TSMC: SOIC,  Intel: Foveros, Samsung: X-Cube


Note:  TSV and silicon interposer are fabricated in semiconductor foundry.   POP, WLP, PLP can be performed in semiconductor or packing houses.

### **Why PLP is Not Classified as 2.5D or 3D** 兩者的目的不同

| **Feature**                | **PLP**                                                                | **2.5D Packaging**                                            | **3D Packaging**                             |
| -------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| **Interposer**             | None. Redistribution layers (RDLs) directly fan out I/O from the die.  | Uses a silicon, glass, or organic interposer to connect dies. | Stacks dies vertically using TSVs.           |
| **Die Placement**          | Single die or side-by-side placement on a reconstituted mold compound. | Side-by-side dies on an interposer.                           | Stacked dies in vertical layers.             |
| **Vertical Interconnects** | No TSVs or vertical interconnects.                                     | TSVs used within the interposer.                              | TSVs connect vertically stacked dies.        |
| **Application Focus**      | Cost-efficient packaging for consumer electronics and IoT.             | High-performance, high-bandwidth computing.                   | High-performance, memory-dense applications. |
| **Thermal Dissipation**    | Good, due to planar design.                                            | Better than 3D due to side-by-side die placement.             | Challenging due to stacked dies.             |
### TSMC WLP

![[Pasted image 20241210120956.png]]


![[Pasted image 20241210120727.png]]


### **Comparison of FO-WLP and FO-PLP**

| **Material**        | **FOWLP**                           | **FOPLP**                                 |
| ------------------- | ----------------------------------- | ----------------------------------------- |
| **Substrate Base**  | Mold compound in circular format    | Mold compound in rectangular format       |
| **Carrier**         | Silicon wafer (temporary)           | Glass or organic panel (temporary)        |
| **Mold Compound**   | Epoxy-based                         | Epoxy-based (higher stability for panels) |
| **RDL Conductors**  | Copper                              | Copper                                    |
| **RDL Dielectrics** | Polyimide, epoxy                    | Polyimide, epoxy                          |
| **Size**            | Limited to wafer sizes (200–300 mm) | Larger panel sizes (e.g., 600-700 mm)     |



![[Pasted image 20241210110307.png]]

### **How PoP Differs from Other Packaging Techniques**

| **Feature**             | **PoP**                                        | **SiP (System-in-Package on substrate)** | **2.5D/3D IC**                              |
| ----------------------- | ---------------------------------------------- | ---------------------------------------- | ------------------------------------------- |
| **Integration**         | Stacks separate logic and memory packages.     | Integrates multiple dies in one package. | Uses interposers or TSVs for chip stacking. |
| **Thermal Dissipation** | Moderate. Heat dissipation may be challenging. | Better with advanced designs.            | Excellent with advanced materials.          |
| **Cost**                | Lower than 2.5D/3D IC but higher than SiP.     | Moderate cost.                           | High cost.                                  |
| **Applications**        | Mobile devices, IoT.                           | IoT, automotive, compact devices.        | HPC, AI, data centers.                      |

### **TSMC InFO vs. Other FO-WLP**

| **Parameter**              | **InFO**                                       | **FOWLP**                             |
| -------------------------- | ---------------------------------------------- | ------------------------------------- |
| **Electrical Performance** | Lower parasitics, better signal integrity.     | Adequate for mid-performance devices. |
| **Thermal Dissipation**    | Better due to thinner mold and optimized RDLs. | Lower thermal efficiency than InFO.   |
| **Package Thickness**      | Ultra-thin (suitable for compact designs).     | Thicker compared to InFO.             |
| **High-Density I/O**       | Higher I/O density with advanced RDLs.         | Slightly lower I/O density.           |

#### Why TSMC acquires Innolux?

CoWoS:  (Multi-) Chip-on-Wafer on Substrate:  和 PLP 無關，但是可以 offload InFO capacity of smartphone, wearable to PLP.  空出 WSL for HPC and AI. 

### **CoWoS Capacity Constraints**

- CoWoS relies on **wafer-based processing**, which is inherently limited by the size of silicon wafers (300 mm).
- TSMC has reported **capacity constraints** for CoWoS, especially with the rising demand for AI GPUs and accelerators (e.g., NVIDIA's GPUs).
- By expanding **PLP capacity**, TSMC can:
    1. **Relieve pressure** on wafer-based packaging (WLP) facilities by offloading cost-sensitive applications to PLP.
    2. **Focus wafer-level capacity on high-margin CoWoS products** for HPC and AI.

Wafer -> Silicon Interposer -> glass interposer

2.5D:
![[Pasted image 20241210110856.png]]

AI chip:  GPU + HBMs

![[Pasted image 20241210111041.png]]

3D:

![[Pasted image 20241210110919.png]]


Dynamics:

![[Pasted image 20241210111245.png]]


Taiwan:  TSMC, ASE

Korea: Samsung, Amkor

USA: Intel

China: JCET: 長電

The recent dynamics, competition, and updates in **Panel Level Packaging (PLP)** are as follows:

---

### **1. Market Dynamics**

- **Growing Demand**:
    
    - The PLP market is expanding rapidly due to increasing demand for advanced packaging in high-performance applications like 5G, AI, IoT, and automotive electronics.
    - Devices requiring higher integration and smaller form factors are driving the adoption of PLP.
- **Shift Toward Miniaturization**:
    
    - As traditional **Wafer-Level Packaging (WLP)** reaches its limits, PLP is being embraced for its ability to handle more compact and complex designs with high I/O density.
- **Cost Pressures**:
    
    - Despite its advantages, PLP faces cost challenges due to high initial capital expenditures and technical barriers, such as yield improvements for large-panel processing.

---

### **2. Competitive Landscape**

- **TSMC**:
    
    - TSMC has been investing heavily in PLP, leveraging its panel display fab acquisitions to expand capacity and improve yield. It is considered a leader in advanced packaging technologies.
    - The company is focusing on scaling **Fan-Out Panel-Level Packaging (FOPLP)** to meet the demands of AI accelerators and high-end mobile processors.
- **Samsung**:
    
    - Samsung is actively developing **Fan-Out Panel Level Packaging (FOPLP)** for applications in mobile, memory, and automotive. They aim to compete with TSMC in cost and scalability.
- **ASE Technology**:
    
    - ASE is focusing on **multi-die and heterogeneous integration** using PLP. Their goal is to provide cost-effective packaging solutions for mid-range applications.
- **Amkor Technology**:
    
    - Amkor is developing **large-panel solutions** for high-volume manufacturing, targeting IoT and consumer electronics.
- **China's Market**:
    
    - Chinese companies are rapidly entering the PLP market, supported by government initiatives to localize semiconductor manufacturing. However, they face challenges in competing with established players in yield and scalability.

---

### **3. Technology Updates**

- **Fan-Out Panel Level Packaging (FOPLP)**:
    
    - An evolution of **Fan-Out Wafer Level Packaging (FOWLP)**, where the use of larger rectangular panels reduces material waste and increases efficiency.
- **Challenges in Scaling**:
    
    - As panel sizes increase, issues like warping, handling, and precision alignment become significant. Companies are working on improving **substrate materials** and **equipment technologies** to address these issues.
- **Integration with Advanced Nodes**:
    
    - PLP is being integrated with **chiplets** and **heterogeneous dies**, allowing high-performance processors to achieve greater functionality and lower latency.
- **Yield Improvements**:
    
    - Improving the **yield rate** for large panels is a critical focus, as defects in large substrates can cause significant losses.

---

### **4. Outlook**

- **Opportunities**:
    
    - The PLP market is expected to grow significantly, driven by applications in smartphones, automotive radar, AI accelerators, and IoT devices.
- **Threats**:
    
    - High costs and technical challenges, especially for scaling and yield optimization, remain obstacles to broader adoption.
- **Focus Areas**:
    
    - Companies are focusing on **cost reduction**, **process scalability**, and **multi-die integration** to stay competitive in the evolving PLP landscape.

In summary, **Panel Level Packaging** represents the future of semiconductor packaging, but its adoption hinges on overcoming yield and cost challenges while meeting the demand for advanced functionality in compact devices.
