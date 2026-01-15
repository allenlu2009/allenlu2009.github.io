---
title: PN Junction vs. Neuron Membrane
date: 2024-10-13 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2024-10-10-Attention_Math]]"
categories:
  - Science
tags:
  - Neuron
---

[[2023-03-26-Transformer_LLM]] , [[2023-02-18-Attn_All_U_Need_Visual]]


## Reference

[Animal Electricity | The Neuron Doctrine (youtube.com)](https://www.youtube.com/watch?v=HK8tea029Ps)
[Why do neurons have an electric charge? | Nernst Potential | Resting Potential of Neurons (youtube.com)](https://www.youtube.com/watch?v=Yd6olw-DzPE)


## Takeaways
### Summary of Key Differences:

| Aspect                     | PN Junction                                | Neuron Membrane                                  |
| -------------------------- | ------------------------------------------ | ------------------------------------------------ |
| **Type**                   | Solid-state (semiconductor)                | Liquid/biological (lipid bilayer)                |
| **Charge Carriers**        | Electrons and holes                        | Positive and negative ions (Na⁺, K⁺)             |
| **Intrinsic Potential**    | Fermi potential                            | Nernst potential                                 |
| **Resistance/Conductance** | Forward: resistance; Backward: capacitance | Conductance through ion channels, bi-directional |
| **Capacitance**            | Junction capacitance                       | Membrane capacitance                             |
| **Turn-on Threshold**      | ~0.7V for silicon                          | ~-55mV action potential threshold                |
| **Current Mechanism**      | Diffusion, drift                           | Diffusion, **active transport**                  |

**非常重要： Neuron Membrane 需要 active transport (pump),  即使什麽都不做，也會消耗能量 ATP -> ADP。** 
爲什麽 PN Junction 不動作不需要 active pump?  因爲在固體中離子是被固定在晶格上，只有在 junction 附件的電子才有機會 diffuse。Diffusion current 和電壓差產生的 Drift current 自然平衡。但是在液體中，離子的 Diffusion 如果沒有 active transport 會讓 membrane 電壓 = 0,  内外的濃度差為 0. 

在神經元膜中，主動運輸（如鈉鉀泵）起著至關重要的作用。這些泵通過將鈉離子(Na⁺)從細胞內部排出，同時將鉀離子(K⁺)輸送進入細胞，來維持細胞的電位差和內外環境的穩定。由於這一過程需要不斷消耗ATP，因此神經元在活躍狀態時會消耗大量能量。

除了主動運輸外，神經元膜上的其他通道和受體也對神經信號的傳遞至關重要。例如，在動作電位生成過程中，當某個刺激使得膜電位達到臨界值時，鈉通道會迅速打開，導致鈉離子大量流入細胞，使膜電位迅速上升；隨後鉀通道打開，使鉀離子流出細胞，恢復膜電位。因此，正確的離子濃度和膜的功能性對於神經系統的正常運作是必不可少的。

總之，Neuron Membrane 的主動運輸機制不僅是維持靜息電位的重要手段，也是神經信號傳遞和反應過程中不可或缺的一部分。. 

## PN Junction Vs. Neuron Membrane Comparison

The comparison is primarily based on different electrical properties and mechanisms:

### Type: Solid State vs. Liquid

- **PN Junction**: A PN junction is a solid-state semiconductor device where two types of semiconductor materials, P-type and N-type, are brought together. It is composed of solid silicon or other semiconductor materials.
    
- **Neuron Membrane**: The neuron membrane is a biological structure made of lipid bilayers and proteins, surrounding the cell in a liquid environment. It’s more dynamic and fluid compared to the solid-state nature of a PN junction.
    

### Charge Carriers: Negative Electrons vs. Positive/Negative Ions

- **PN Junction**:
    - **N-type**: Contains excess negative electrons as charge carriers (majority carriers).
    - **P-type**: Contains positive "holes" (absence of electrons) as charge carriers (majority carriers).
- **Neuron Membrane**:
    - Charge carriers are **positive and negative ions**, primarily sodium (Na⁺), potassium (K⁺), calcium (Ca²⁺), and chloride (Cl⁻). These ions move across the membrane through ion channels to propagate electrical signals.

### Intrinsic Potential: Fermi Potential vs. Nernst Potential

- **PN Junction**:
    - The **Fermi potential** (or built-in potential) is the potential difference that exists across the junction in equilibrium, where the Fermi levels of P-type and N-type semiconductors align. It results from the diffusion of electrons and holes.
- **Neuron Membrane**:
    - The **Nernst potential** (or equilibrium potential) describes the voltage across the membrane that balances the concentration gradient of a particular ion, like Na⁺ or K⁺. It is calculated using the Nernst equation and depends on the relative concentrations of ions inside and outside the cell.

### Resistance/Conductance: Resistance vs. Membrane Channels

- **PN Junction**:
    
    - The **resistance** of a PN junction varies based on the applied voltage and biasing (forward or reverse). In forward bias, the resistance decreases as current flows, while in reverse bias, it acts as an insulator with high resistance until breakdown occurs.
- **Neuron Membrane**:
    
    - The membrane acts as a dynamic conductor, regulated by **membrane channels** (voltage-gated and ligand-gated ion channels). These channels control the flow of ions across the membrane, influencing conductance. When channels open, the membrane becomes more conductive (depolarization), while closed channels increase resistance (hyperpolarization).

### Capacitance

- **PN Junction**:
    - A PN junction exhibits **junction capacitance**, particularly under reverse bias. This capacitance changes with voltage and is linked to the depletion region, where charge carriers are absent.
- **Neuron Membrane**:
    - The neuron membrane acts as a **capacitor**, storing charge across the membrane. The lipid bilayer acts as the insulator, and ions on either side of the membrane form the charge.

### Threshold for Action: Turn-on Voltage vs. Action Potential Threshold

- **PN Junction**:
    
    - The **turn-on voltage** in a PN junction (in forward bias) is the minimum voltage required to allow significant current to flow through. For silicon, this is typically around **0.7V**.
- **Neuron Membrane**:
    
    - Neurons have an **action potential threshold**, usually around **-55mV**. Once this threshold is exceeded, voltage-gated ion channels open, leading to a rapid depolarization (the action potential).

### Current Flow: Directionality

- **PN Junction**:
    
    - In forward bias, current flows easily from the P-type (positive) to N-type (negative) side. In reverse bias, current flow is minimal or blocked until breakdown occurs.
- **Neuron Membrane**:
    
    - Current in neurons is carried by ions and is typically unidirectional along the axon during an action potential. This is because of the sequential opening and closing of ion channels and the refractory period after firing.

### Rest State

- **PN Junction**:
    
    - In equilibrium (no external bias), there is a depletion region at the junction where no free charge carriers are present. A small potential difference (the built-in voltage) exists, but no net current flows.
- **Neuron Membrane**:
    
    - In the resting state, the neuron has a **resting membrane potential**, usually around **-70mV**, maintained by the sodium-potassium pump and leak channels, creating a negative charge inside relative to the outside.

### Current Mechanism: Diffusion vs. Active Transport

- **PN Junction**:
    
    - Current results primarily from **diffusion** of electrons and holes across the junction. In reverse bias, current is mainly due to minority carrier drift.
- **Neuron Membrane**:
    
    - Ion flow across the membrane is controlled by both **diffusion** (via passive ion channels) and **active transport** (via the sodium-potassium pump), which uses ATP to maintain ion gradients.