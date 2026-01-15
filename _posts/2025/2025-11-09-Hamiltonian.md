---
title: Lagrangian and Hamiltonian Mechanics
date: 2025-11-09 09:28:08
description: revision control
typora-root-url: ../../allenlu2009.github.io
categories:
  - Science
tags:
  - Geometry
  - Math
  - Mechanics
  - Physics
---




Good youtube video: https://www.youtube.com/watch?v=Ohrl3S2wcBU&list=PLDcSwjT2BF_WVum1CMO9hMSIa7TyrGjsK&ab_channel=Mathemaniac

Mathemaniac 的 youtube video 是從 Michigan University 的 Gabriele Carcassi 的 paper 和 video 得出的！ 他另外有 12 不同的 Hamiltonian 的表示法。

Dr. Jorge Diaz: Virtual work to Lagrangian: https://www.youtube.com/watch?v=QbnkIdw0HJQ


Braintruffle, very interesting video to explain from geometry viewpoint (Symplectic geometry) that Hamiltonian is robust over long time evolution!   https://www.youtube.com/watch?v=nCg3aXn5F3M

# 深度解讀 Newtonian、Lagrangian、Hamiltonian 力學的三種觀點

以下三點補充說明，可以更清楚地看出這三種力學形式之間的深層關係，以及為何在不同情境下會選擇不同的框架。

---

## 1. **抽象層次的演化：從 Vectors → Scalars → Geometry**

每一種力學形式代表物理描述方式的不同抽象層次。

### **Newtonian Mechanics — 向量 (Vectorial)**

* 建立在 **force**、**acceleration** 與 **vector equations** 上。
* 需要選定座標系，並將每個向量分解成 (x, y, z) 分量。
* 對複雜系統或曲面問題會變得繁瑣。

### **Lagrangian Mechanics — 純量 (Scalar)**

* 以一個 **單一純量函數** 描述整個系統：
  **L = T − V**（動能減位能）。
* **Scalar** 與座標系無關，因此能自然處理座標轉換。
* 系統動力學來自對 **Action (作用量)** 的最小化，提供比平衡力更統一與優雅的描述方式。

### **Hamiltonian Mechanics — 幾何 (Geometric)**

* 觀點移動到抽象的 **phase space (相空間)**，其座標為 ((q, p))。
* 系統演化成為由 Hamiltonian 產生的 **flow (流)**。
* 揭露更深層的結構性，包括：

  * symmetries（對稱性）
  * conserved quantities（守恆量）
  * canonical transformations（正則變換）
  * Liouville’s theorem（相空間體積守恆）

---

## 2. **對 Constraints 的詮釋：從 Forces → Coordinates**

這個角度強調各種力學如何「處理約束」。

### **Newtonian：需要 Constraint Forces**

* 必須引入額外的 **forces of constraint**（例如支持力、張力）讓物體保持在受限空間中。
* 這些力可能不做功，但在 Newtonian 框架中不可省略。

### **Lagrangian：使用 Generalized Coordinates**

* 選擇座標時即可將 **constraints 直接寫進座標系**。
* 例如：描述球面運動 → 使用 ((\theta, \phi)) 而不是 ((x, y, z))。
* Constraint forces 在方程式中完全消失，大幅簡化問題。
* 在以下領域極具優勢：

  * robotics（機器人學）
  * 多連桿機構
  * 分子動力學
  * 任意含 holonomic constraints 的系統

---

## 3. **作為通往現代物理的關鍵橋樑**

從 Lagrangian 過渡到 Hamiltonian，不只是數學技巧，而是 20 世紀物理學的核心入口。

### **Quantum Mechanics**

* Hamiltonian (H) 成為決定時間演化的 **operator**（經由 Schrödinger equation）。
* (q) 與 (p) 變成 **non-commuting operators**，導致：

  * uncertainty principle（不確定性原理）
  * quantum observables（量子可觀測量）
  * canonical quantization（正則量子化）

### **Statistical Mechanics**

* 系統的微觀狀態對應到高維 **phase space** 中的一個點。
* Hamiltonian 決定：

  * Boltzmann factor 的能量分佈
  * energy surfaces（能量曲面）
  * 相空間中的統計流動
* 宏觀熱力學行為源自 phase-space trajectories 的統計分布。



### 📘 Lagrangian 與 Hamiltonian 力學之比較

| 項目 | **Lagrangian 力學** | **Hamiltonian 力學** |
|---|---|---|
| **基本變數** | $(q_i, \dot{q}_i)$：位置與速度 | $(q_i, p_i)$：位置與動量 |
| **定義函數** | $L(q, \dot{q}, t)$：動能 − 位能 | $H(p, q, t)$：總能量（動能 + 位能） |
| **關聯式** | $p_i = \frac{\partial L}{\partial \dot{q}_i}$ | $H = p_i \dot{q}_i - L$（Legendre 變換） |
| **運動方程** | Euler–Lagrange 方程：<br>$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0$ | Hamilton 方程組：<br>$\dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}$ |
| **方程階數** | $N$ 個二階微分方程 | $2N$ 個一階微分方程 |
| **幾何空間** | 構形空間 (Configuration space) | 相空間 (Phase space) |
| **幾何結構** | 路徑積分、變分原理 | 辛幾何 (Symplectic geometry) |
| **直觀物理意義** | 動作量 (Action) 最小原理 | 能量守恆與流形結構 |
| **優勢應用** | 對稱性分析、Noether 定理、場論 | 量子化、能量分析、數值模擬、統計力學 |
| **方程求解特性** | 適合從物理過程導出運動方程 | 適合解析守恆律與相軌跡演化 |
| **在量子化中的角色** | 路徑積分形式 (Feynman) | 能量算符 (Schrödinger) |
| **數值分析特性** | 容易引入能量漂移 | 可保辛結構（symplectic integrator） |
| **代表性思維方式** | 「最小作用量 → 運動」 | 「能量流 → 狀態演化」 |

---

### 📘 補充說明

#### **基本概念差異**
- **Lagrangian 力學**：以廣義坐標 $(q_i, \dot{q}_i)$ 為基本變數，通過 Lagrangian 函數 $L = T - V$ 描述系統。
- **Hamiltonian 力學**：以正則坐標 $(q_i, p_i)$ 為基本變數，通過 Hamiltonian 函數 $H = T + V$ 描述系統。

#### **變數的本質差異**
- **Hamiltonian 力學** 明確承認動量 $p$ 與位置 $q$ 是兩個不同的物理量，在相對論或量子力學中，$p$ 不再僅僅是 $\dot{q}$ 的線性函數，因此 Hamiltonian 形式更一般化。
- **Lagrangian 力學** 只使用 $(q, \dot{q})$，其中 $p = \frac{\partial L}{\partial \dot{q}}$。

#### **形式的等價與轉換關係**
兩種表述在經典力學中等價，通過 **Legendre 變換** 相互對應：
$$
H(p, q) = p\dot{q} - L(q, \dot{q}), \quad \text{其中 } p = \frac{\partial L}{\partial \dot{q}}
$$

#### **幾何與物理意涵**
- **Hamiltonian 力學** 具有明確的**相空間**幾何結構，能用辛幾何描述。
- **Lagrangian 力學** 對應於構形空間上的路徑最小作用量原理。
- 在物理直觀上，Hamiltonian 通常對應「總能量」，因此在量子化與統計物理中更自然。

#### **微分方程形式與數值分析**
- **Hamiltonian 力學** 導出 $2N$ 個一階方程：
  $$
  \dot{q}_i = \frac{\partial H}{\partial p_i}, \quad \dot{p}_i = -\frac{\partial H}{\partial q_i}
  $$
- **Lagrangian 力學** 導出 $N$ 個二階方程：
  $$
  \frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0
  $$
- 從數值分析角度，Hamiltonian 形式較易保持能量守恆與辛結構，在長時間模擬上更穩定。

#### **延伸應用**
- **量子力學**：Hamiltonian 直接對應於系統的能量算符，為薛丁格方程的核心。
- **場論**：Lagrangian 形式常用於推導對稱性與守恆律（Noether 定理）；Hamiltonian 形式則用於量子化與正則對易關係。
- **總結**：兩者在不同層面上各有優勢——Lagrangian 側重「變分與對稱性」，Hamiltonian 側重「結構與演化」。



## Lagrangian 和 Least Action Principle

### 1. Virtual work 解釋 : from Newtonian to Least Action

![[Pasted image 20251109221823.png]]

假設所有的 forces 都是 conservative force.
![[Pasted image 20251109222159.png]]
![[Pasted image 20251109222013.png]]
![[Pasted image 20251109222459.png]]



### 2. Extended phasor plan 解釋 : 包含 Hamiltonian 和 least action.  經過比較複雜的 flow concept 得到 vector potential, 最後可以得到 least action.  太複雜！ 不是從 Newtonian 開始，而是 divergence free extended phase space.


Trick 1: 從 phase space (p, q) 延申到 extended phase space (p, q, t)
Trick 2: make $\nabla \cdot u = 0$  by changing the coordinate system
Trick 3: 最重要的部分:  $u = -\nabla \times \theta$
![[Pasted image 20251110203055.png]]
![[Pasted image 20251110203120.png]]

$L= p \dot{q} - H(p,q)$
結論是 vector potential (無法觀察物理量) 的綫積分就是 action.  vector potential 在 extended phase space 切綫的内積就是 Lagrangian.  

![[Pasted image 20251110203154.png]]



![[Pasted image 20251109221212.png]]

## Hamiltonian 的 12 種解釋：沒有 least action principle!!

Hamiltonian 側重「結構與演化」。沒有 least action principle 可以理解，因爲沒有 second order differential equation!

![[Pasted image 20251109220343.png]]


## Relation (and physical interpretation) between Lagrangian, Hamiltonian, and Legendre Transformation

**Trajectory:  phasor plane**

**Motion:  Extended Phasor plane , 加上時間 axis.**

**Time history: 對應 p(t), q(t)** 

Motion 投影在 (p, q) plane 就是 trajectory.
Motion 投影在 (p, t) 或是 (q, t) 就是 time history. 


1, Lagrangian = T - V  (KineTic energy - Potential energy)

2. generalized Lagrangian

   Hamiltonian: why it equal to least action?  but action is the integration of Lagrangian, not Hamiltonian?

3. Why the Lagrangian and Hamiltonian is linked by Legendre transformation?



![[Pasted image 20250516000511.png]]

![[Pasted image 20250516000539.png]]



