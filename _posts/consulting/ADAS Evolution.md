
Prompt: deep research of adas algorithm evolution from earlier a few cameras, to BEV, occupancy, end-to-end, and VLM/VLA

Awesome topic. Here’s a compact, opinionated “deep research” map of how ADAS/AV perception–planning stacks evolved: from classic camera-centric ADAS to BEV, occupancy, end-to-end (E2E), and finally VLM/VLA.

# 1) Classic ADAS (≈2000–2016): few cameras, modular stack

**What it looked like.** Mono/stereo front camera + radar; handcrafted CV (edges, Hough lanes), Kalman/particle filters, rule-based planning, safety policies like RSS. Typical features: AEB, ACC, LKA.
**Why it dominated.** Compute was scarce; datasets small; explainability and standards mattered.
**Representative references.** NVIDIA’s PilotNet showed early E2E potential from a single cam (2016), but industry largely stuck with modular pipelines; in parallel, Mobileye pushed formal safety (RSS). ([arXiv][1], [MDPI][2], [SafeAI][3])

# 2) Surround view & multi-camera fusion (≈2016–2020)

**What changed.** 4–8 cameras became common; surround-view stitched panoramas; early learned fusion with classic 3D grids.
**Limitations.** Perspective warping, depth ambiguity, and occlusions made planning brittle, motivating unified 3D representations.

# 3) BEV as the lingua franca (≈2020–)

**Key idea.** Lift image features to a unified **bird’s-eye-view** where detection, mapping, and planning can share geometry.

* **Lift-Splat-Shoot (LSS, 2020):** implicit unprojection to BEV; even showed BEV cost-maps for planning. ([arXiv][4])
* **BEVFormer (2022):** spatiotemporal Transformers learn a unified BEV across time; strong camera-only 3D detection & map segmentation; widely adopted/baselined (official code, challenge reports). ([arXiv][5], [img.shlab.org.cn][6], [GitHub][7], [Google Cloud Storage][8])
* **HD map from cameras:** HDMapNet (2021) popularized online map learning from multi-view images/LiDAR; spawned vectorized/streaming map lines. ([arXiv][9], [tsinghua-mars-lab.github.io][10], [CVF Open Access][11])
  **Why BEV won.** Viewpoint-invariant geometry, clean fusion with LiDAR/radar, task sharing (det, lanes, freespace) on a common grid.

# 4) Occupancy-centric perception & forecasting (≈2022–)

**Key idea.** Predict **which space is taken** (and how it will move), not just boxes/lanes. This supports long-horizon planning, occlusions, and “speculative” agents.

* **Waymo Occupancy Flow (2022):** spatiotemporal grids with per-cell occupancy + flow; launched a benchmark/challenge. ([arXiv][12], [Waymo][13])
* **Tesla Occupancy Network (2022, AI Day):** camera-only occupancy/freespace as a planning substrate (industry briefings & technical recaps). ([Think Autonomous][14], [TESLARATI][15], [jakepoz.com][16])
* **Surveys & follow-ups:** 2024 survey of 3D occupancy for AV; 2024–2025 methods scale vision-only occupancy and unify 2D/3D benchmarks. ([arXiv][17])
  **Why it matters.** Occupancy unifies static/dynamic elements, handles occlusion, and plugs directly into risk/cost maps for planners.

# 5) End-to-End driving (sensors → actions)

**Two waves.**

* **Wave 1 (2016–2021):** proofs of concept (PilotNet; ChauffeurNet-style IL/RL; TransFuser-type sensor fusion) struggled with data scale, long-tail, and evaluation. ([arXiv][1])
* **Wave 2 (2023–2025):** **foundation-model** flavored E2E with bigger data, closed-loop eval, and intermediate supervision (waypoints, costmaps, occupancy). Solid surveys emerged; OEMs and startups report rapid gains. ([arXiv][18])
  **Case examples.**
* **Wayve (AV2.0):** vision-only E2E that generalizes without HD maps; public comms on scaling, closed-loop, and US expansion; recent press demos in London. ([wayve.ai][19], [WIRED][20], [Business Insider][21])
* **Tesla:** migrated perception to occupancy/lanes with large-scale auto-labeling; ongoing E2E planning discussions in AI Day materials/recaps. ([Think Autonomous][14], [Kevin Chen][22])
  **Why the resurgence.** Better data engines, synthetic/world-model augmentation, improved intermediate targets, and stronger simulators/leaderboards.

# 6) Vision-Language(-Action): transparency, reasoning, and instruction following (≈2023–)

**What’s new.** VLMs explain *why* and can be aligned to *do*. For driving, they’re used to probe understanding, justify actions, and increasingly to condition/control policies.

* **Wayve LINGO-2 (2024):** links language with closed-loop driving to explain and condition behavior; LingoQA benchmark measures truthful explanations. ([wayve.ai][23], [arXiv][24])
* **DriveVLM (2024):** integrates VLM reasoning with hierarchical planning; explores hybridization with classic stacks and real-car deployment. ([arXiv][25])
* **Toward VLA:** 2025 work (e.g., SimLingo) frames **vision-language-action** for closed-loop driving with explicit language–action alignment. ([arXiv][26])
  **Caveat.** VLMs still trail humans on nuanced video Q\&A and spatial grounding, but are improving quickly. ([arXiv][24])

---

## Trade-offs by paradigm

* **Modular ADAS:** highly interpretable; safety cases (RSS) tractable; but brittle handoffs and covariate shift. ([MDPI][2])
* **BEV:** common metric space, robust multi-tasking, sensor fusion; still needs accurate lifting (depth/ego-motion) and strong temporal modeling. ([arXiv][5])
* **Occupancy:** handles occlusion/unknowns and plugs into planners; costs more compute/labels (though vision-only scaling is emerging). ([arXiv][12])
* **E2E:** optimizes for driving objective and scales well with data; must solve data coverage, safety guarantees, and closed-loop evaluation. Surveys now standardize eval (bench2drive/CARLA, nuPlan, etc.). ([arXiv][18])
* **VLM/VLA:** adds language supervision, introspection, and instruction following; challenges: latency, grounding, and aligning talk with action. ([arXiv][24])

---

## What actually enabled each shift?

* **Sensors & sync:** from single cam+radar → multi-cam rigs with tight calibration (pre-req for BEV).
* **Representation learning:** LSS/BEVFormer cracked camera-only 3D via lift + temporal transformers. ([arXiv][4])
* **Supervision at the right level:** occupancy/costmaps/waypoints as mid-level targets made closed-loop learning stable. ([arXiv][12])
* **Data engines:** auto-labeling, active mining, long-tail upweighting; Wayve/Tesla publicize this heavily. ([WIRED][20], [TESLARATI][15])
* **Evaluation:** from open-loop metrics → closed-loop sims/benchmarks and new occupancy-flow leaderboards. ([Waymo][13])

---

## Where things are heading (2025→)

1. **Occupancy-first camera stacks** (sometimes with lightweight radar) feeding **neural planners** (cost-map or waypoint-conditioned). ([arXiv][12])
2. **E2E with interpretable heads** (language rationales, scene graphs, counterfactuals) to satisfy safety/UX. ([wayve.ai][23], [arXiv][24])
3. **VLA driving models** that can be *instructed* (e.g., “prefer defensive gaps,” “pull over at next legal spot”) with policy alignment to language. ([arXiv][26])
4. **Benchmarks** shifting from mAP to **risk-aware closed-loop** and **occupancy-flow** scores; emerging unified occupancy benchmarks simplify cross-dataset comparisons. ([arXiv][27])

---

## Starter reading pack (chronological breadcrumbs)

* 2016 – **PilotNet** (first widely cited camera→steering E2E). ([arXiv][1])
* 2017 – **RSS** (formal safety rules). ([SafeAI][3])
* 2020 – **LSS** (camera→BEV lifting). ([arXiv][4])
* 2021 – **HDMapNet** (online HD map learning). ([arXiv][9])
* 2022 – **BEVFormer** (spatiotemporal BEV with Transformers). ([arXiv][5])
* 2022 – **Occupancy Flow** (joint occupancy + motion). ([arXiv][12])
* 2023 – **E2E surveys** consolidate methods/eval. ([arXiv][18])
* 2024 – **LingoQA/LINGO-2** (driving VLMs; explanations and closed-loop language-linked control). ([arXiv][24], [wayve.ai][23])
* 2024–2025 – **Occupancy surveys/benchmarks & VLA prototypes** (SimLingo; unified occupancy). ([arXiv][17])

---

## Practical guidance (what to build, today)

* **Shipping ADAS**: camera-first **BEV + occupancy freespace**, lightweight radar fusion for adverse weather; planner trained on risk/cost grids with *limited* E2E fine-tuning on waypoints. ([arXiv][5])
* **Research**: couple an occupancy-flow head with a language **rationale** head; align rationales with actions via closed-loop losses (VLA-style). Track both **safety** (RSS-inspired constraints) and **explanatory truthfulness** (LingoQA-like). ([SafeAI][3], [arXiv][24])

If you want, I can turn this into a 1–2 page slide summary with diagrams (pipeline transitions + pros/cons matrix) or add a reading curriculum tailored to your NPU formats focus (e.g., where occupancy/BEV/E2E/VLMs stress memory/compute).

[1]: https://arxiv.org/abs/1604.07316?utm_source=chatgpt.com "End to End Learning for Self-Driving Cars"
[2]: https://www.mdpi.com/1424-8220/21/20/6733?utm_source=chatgpt.com "On the Development of Autonomous Vehicle Safety ..."
[3]: https://safeai.webs.upv.es/wp-content/uploads/2020/02/Rick-Salay-safeAI-presentation.pdf?utm_source=chatgpt.com "Model-Based Engineering of Automotive Systems and ..."
[4]: https://arxiv.org/abs/2008.05711?utm_source=chatgpt.com "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera ..."
[5]: https://arxiv.org/abs/2203.17270?utm_source=chatgpt.com "BEVFormer: Learning Bird's-Eye-View Representation from ..."
[6]: https://img.shlab.org.cn/pjlab/files/2022/06/637910706299480000.pdf?utm_source=chatgpt.com "arXiv:2203.17270v1 [cs.CV] 31 Mar 2022"
[7]: https://github.com/fundamentalvision/BEVFormer?utm_source=chatgpt.com "fundamentalvision/BEVFormer: [ECCV 2022] This is the ..."
[8]: https://storage.googleapis.com/waymo-uploads/files/research/3DCam/3DCam_BEVFormer.pdf?utm_source=chatgpt.com "Improving BEVFormer for 3D Camera-only Object Detection"
[9]: https://arxiv.org/abs/2107.06307?utm_source=chatgpt.com "HDMapNet: An Online HD Map Construction and Evaluation Framework"
[10]: https://tsinghua-mars-lab.github.io/HDMapNet/?utm_source=chatgpt.com "HDMapNet"
[11]: https://openaccess.thecvf.com/content/WACV2024/papers/Yuan_StreamMapNet_Streaming_Mapping_Network_for_Vectorized_Online_HD_Map_Construction_WACV_2024_paper.pdf?utm_source=chatgpt.com "StreamMapNet: Streaming Mapping Network for Vectorized ..."
[12]: https://arxiv.org/abs/2203.03875?utm_source=chatgpt.com "Occupancy Flow Fields for Motion Forecasting in Autonomous Driving"
[13]: https://waymo.com/intl/fil/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/?utm_source=chatgpt.com "Occupancy Flow Fields for Motion Forecasting in ..."
[14]: https://www.thinkautonomous.ai/blog/occupancy-networks/?utm_source=chatgpt.com "A Look at Tesla's Occupancy Networks"
[15]: https://www.teslarati.com/tesla-ai-day-2022-new-roundup-updated/?utm_source=chatgpt.com "Tesla's AI Day in 2022 - Optimus, FSD Beta & Dojo updates"
[16]: https://www.jakepoz.com/thoughts/tesla-ai-day-2022.html?utm_source=chatgpt.com "Tesla AI Day 2022 Review | Jake Poznanski"
[17]: https://arxiv.org/pdf/2405.05173?utm_source=chatgpt.com "A Survey on Occupancy Perception for Autonomous Driving"
[18]: https://arxiv.org/pdf/2306.16927?utm_source=chatgpt.com "End-to-end Autonomous Driving: Challenges and Frontiers"
[19]: https://wayve.ai/science/?utm_source=chatgpt.com "Wayve Science: Innovating Embodied AI Research for Self- ..."
[20]: https://www.wired.com/story/wayves-ai-self-driving-system-is-here-to-drive-like-a-human-and-take-on-waymo-and-tesla?utm_source=chatgpt.com "Wayve's AI Self-Driving System Is Here to Drive Like a Human and Take On Waymo and Tesla"
[21]: https://www.businessinsider.com/i-took-chaotic-robotaxi-ride-through-london-impressive-one-question-2025-6?utm_source=chatgpt.com "'I've never seen that before': My chaotic robotaxi ride through London with Wayve's CEO"
[22]: https://kevinchen.co/blog/tesla-ai-day-2022/?utm_source=chatgpt.com "Analyzing Tesla AI Day 2022 - Kevin Chen"
[23]: https://wayve.ai/thinking/lingo-2-driving-with-language/?utm_source=chatgpt.com "LINGO-2: Driving with Natural Language"
[24]: https://arxiv.org/html/2312.14115v4?utm_source=chatgpt.com "LingoQA: Visual Question Answering for Autonomous Driving"
[25]: https://arxiv.org/abs/2402.12289?utm_source=chatgpt.com "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models"
[26]: https://arxiv.org/html/2503.09594v1?utm_source=chatgpt.com "SimLingo: Vision-Only Closed-Loop Autonomous Driving ..."
[27]: https://arxiv.org/html/2503.24381v2?utm_source=chatgpt.com "UniOcc: A Unified Benchmark for Occupancy Forecasting ..."


# ADAS/AV Algorithm Evolution — Slide Summary

---

## 1. Evolution Timeline (pipeline view)

**Early ADAS (2000–2016)**

* Few cameras + radar
* Hand-crafted CV, Kalman filters
* Tasks: AEB, ACC, LKA
* Pros: simple, explainable
* Cons: brittle, limited perception

**Multi-Cam Fusion (2016–2020)**

* 4–8 cams, stitched surround-view
* Learned fusion starts
* Pros: better coverage
* Cons: depth/occlusion issues

**BEV (2020–)**

* Lift-Splat-Shoot, BEVFormer
* 2D BEV grid with implicit depth ("2.5D")
* Pros: unified ground-plane space, multi-task
* Cons: z compressed, needs good depth cues

**Occupancy Networks/Flow (2022–)**

* Tesla: Occupancy Net (perception in 3D)
* Waymo: Occupancy Flow (perception + prediction in 3D+time)
* Pros: handles occlusion, dynamic motion
* Cons: heavy compute/memory

**End-to-End Driving (2023–)**

* Sensors → waypoints/controls directly
* E.g., Wayve, Tesla E2E stacks
* Pros: optimizes for driving objective, scales with data
* Cons: safety/interpretability still hard

**Vision-Language(-Action) (2024–)**

* LINGO-2, DriveVLM
* Add reasoning, explanation, instruction following
* Pros: transparency, alignment with humans
* Cons: grounding/latency challenges

---

## 2. Representations: Dimensionality & Task Scope

| Representation         | Space                     | Perception                | Prediction              |
| ---------------------- | ------------------------- | ------------------------- | ----------------------- |
| Classic ADAS (cameras) | 2D image                  | Yes                       | No                      |
| BEV (LSS, BEVFormer)   | 2D grid (x–y, implicit z) | Yes                       | Optional (extra head)   |
| BEV-3D / Depth-aug BEV | 2.5D/3D                   | Yes                       | Sometimes               |
| Occupancy Network      | 3D voxels                 | Yes                       | Separate                |
| Occupancy Flow         | 3D voxels + time          | Yes                       | Yes                     |
| End-to-End             | Latent → action           | Joint perception–planning | Joint                   |
| VLM/VLA                | Multimodal tokens         | Yes                       | Yes + explain/condition |

---

## 3. Pros/Cons Matrix (today’s trade-offs)

**BEV (2.5D)**

* ✅ Efficient
* ✅ Unified detection/segmentation
* ❌ Loses fine vertical structure
* ❌ Needs strong lifting module

**Occupancy (3D)**

* ✅ Richer representation, occlusion-aware
* ✅ Natural bridge to planning
* ❌ Heavy compute/memory
* ❌ Harder labeling at scale

**End-to-End**

* ✅ Direct optimization, scales with data
* ✅ Closed-loop friendly
* ❌ Hard safety guarantees
* ❌ Debug/interpretation challenges

**VLM/VLA**

* ✅ Explainability, human-AI interaction
* ✅ Instruction-following
* ❌ Grounding gaps
* ❌ Latency & deployment hurdles

---

## 4. NPU / Compute Considerations

* **BEV (2D feature maps):** moderate memory, bandwidth-bound; efficient for embedded/edge NPUs.
* **Occupancy (3D voxels):** high memory & compute; requires aggressive quantization (e.g., int4/3-bit) to fit edge NPUs.
* **End-to-End (transformer-heavy):** sequence length dominates; low-precision formats (fp4, int3) critical.
* **VLM/VLA:** largest memory footprint (vision tokens + language tokens + temporal context). Hybrid precision (int4 activations, int8 weights) essential.

---

## 5. Key Messages

1. **BEV is 2.5D** → efficient middle ground; still dominant in industry.
2. **Occupancy flow unifies perception + prediction** → but compute-hungry.
3. **End-to-End resurgence** due to foundation-scale data + closed-loop eval.
4. **VLM/VLA adds reasoning & explainability** → likely to be required for safety and UX.
5. **For NPUs**: memory formats (int4/3-bit) will be crucial to make occupancy/E2E/VLM stacks run on edge.


**Two questions: 1. is BEV 2D or 3D? 2. does occupancy flow or network including both perception and prediction? or perception only? How about BEV?**

### 1. **Is BEV 2D or 3D?**

* **BEV (Bird’s-Eye View)** is **typically 2D**, i.e. a top-down projection of the world onto the ground plane (x–y).

  * Each grid cell carries features like class labels (road, vehicle), vectors (lanes, boundaries), or aggregated embeddings.
  * Height (z) is often *collapsed* (via max-pooling or learned lifting), which works well because driving happens on the road plane.

* **Extensions to 3D BEV exist**, sometimes called **BEV-3D** or **4D occupancy (3D + time)**. These explicitly keep height layers (x–y–z). Examples:

  * BEVDepth, BEVDet4D — discretize depth bins per pixel before splatting into BEV.
  * Occupancy Networks (Tesla, Waymo) can be seen as a **volumetric BEV** (2.5D/3D grid).

👉 **Rule of thumb:** “BEV” in papers usually means **2D top-down grid**, unless explicitly noted as “3D occupancy BEV.”

---

### 2. **Does Occupancy Flow / Occupancy Network include both perception and prediction, or perception only?**

* **Occupancy Flow (Waymo, 2022):**

  * Predicts **which voxels (3D cells) are occupied now AND how occupancy will evolve over time** (flow field).
  * So it’s **both perception (static occupancy)** *and* **prediction (dynamic motion/flow)** in one representation.

* **Tesla’s Occupancy Network (2022):**

  * Primarily **perception**: infers static 3D occupancy (free space, drivable space, obstacles) from cameras.
  * They later layer **motion forecasting** on top, but the *occupancy network itself* is more perception-oriented.

👉 **Occupancy flow = perception + prediction.**
👉 **Occupancy network (Tesla’s original) = perception first, prediction added separately.**

---

### 3. **How about BEV?**

* **Classic BEV methods (LSS, BEVFormer, HDMapNet):**

  * Mainly **perception**: detect objects, segment lanes, free space, or HD map elements in BEV.
  * They don’t inherently predict future trajectories.

* **Extended BEV models:**

  * Some integrate **temporal prediction** (e.g., BEVFormer uses multi-frame temporal attention, but still outputs present BEV).
  * Prediction is usually added as a **separate head** on top of BEV features (e.g., forecasting future agent positions on BEV grid).

👉 **BEV by itself = perception; BEV + forecasting head = perception + prediction.**
👉 **Occupancy flow natively fuses the two.**

---

✅ **Summary table**

| Representation            | Space               | Perception | Prediction                   |
| ------------------------- | ------------------- | ---------- | ---------------------------- |
| BEV (standard)            | 2D (x–y) top-down   | Yes        | No (unless added separately) |
| BEV-3D / volumetric       | 3D grid             | Yes        | Sometimes                    |
| Occupancy Network (Tesla) | 3D occupancy        | Yes        | Motion added on top          |
| Occupancy Flow (Waymo)    | 3D occupancy + time | Yes        | Yes (motion flow)            |

---



**Is BEVformer 2D, 2.5D, or 3D?**
Great follow-up — **BEVFormer** sits in between 2D and 3D, so people sometimes call it “2.5D.” Let’s be precise:

---

### **BEVFormer (2022)**

* **Core representation:**

  * BEVFormer generates **2D BEV feature maps** (x–y grid) using multi-camera views + temporal attention.
  * Each BEV query corresponds to a cell on the ground plane (like a pixel in top-down view).
  * **Height (z) is not explicitly modeled** as multiple layers/voxels — it’s implicitly encoded in the features when lifting from camera rays.

* **Tasks supported:**

  * 3D object detection (cars, pedestrians, etc.)
  * Map segmentation (lanes, drivable areas, crosswalks)
  * Ego-motion–aware temporal fusion

* **Why not 3D?**

  * True 3D occupancy grids are cubic in memory/compute (x–y–z). BEVFormer avoids this by compressing along z and working in 2D BEV space.
  * Instead of volumetric prediction, BEVFormer outputs **3D boxes** and map elements on top of the **2D BEV feature plane.**

---

✅ **Classification:**

* **Not 3D** (no explicit voxels in z).
* **More than 2D image** (ground-plane geometry-aware, fused across cameras & time).
* So best described as **2.5D**: a **2D BEV feature plane enriched with implicit 3D info** (depth + temporal fusion).

---

👉 Short answer: **BEVFormer is 2.5D.**
It projects 3D cues into a 2D BEV grid, which is why it’s efficient but still supports 3D tasks.

---

Do you want me to also contrast it with **BEVDepth** and **occupancy networks**? Those sit more toward 2.5D→3D, and the comparison helps show the progression.




**if BEV is 2D feature maps, how to do 3D object detection for car and pedastrian?**

Excellent — this is the subtle but important part of why BEV works so well in AV perception.

---

### **How BEV → 3D object detection works**

Even though BEV is a **2D top-down grid (x–y)**, you can still detect **3D objects** (x–y–z, size, yaw) because:

1. **Lifted features already carry depth cues.**

   * Before projecting into BEV, the model lifts image features along camera rays (with learned depth distributions or geometric priors).
   * So each BEV cell has information not just about “what’s here in x–y,” but also about **where in z the features came from.**

2. **Detection head predicts 3D boxes, not just 2D points.**

   * On top of BEV features, you put a detection head (similar to anchor-based or anchor-free 3D detection).
   * For each object, the network regresses:

     * **(x, y, z)** → 3D position in ego/world frame
     * **(w, l, h)** → object size (width, length, height)
     * **yaw/rotation** → orientation
     * **class/confidence**

   Example: In BEVFormer, queries attend to BEV cells, then predict 3D bounding boxes directly.

3. **Bird’s-eye geometry simplifies the problem.**

   * Since BEV is aligned to the ground plane, object **footprints** are nicely represented (cars become rectangles in BEV).
   * The **z and height** are learned as offsets/regressions, rather than needing a full 3D voxel grid.

---

### ✅ Analogy

Think of it like this:

* BEV gives you a **map of where things are on the ground plane.**
* From that, the detector learns to **“pop up”** a 3D box (height + z-offset) for each footprint.

---

### **Why this works well for driving**

* Most relevant actors (cars, pedestrians, cyclists) are grounded (touch the road surface).
* Variation in z is limited compared to x–y (few cars float in the air).
* That makes BEV (2D + learned z regression) efficient and effective for 3D detection.

---

👉 **Summary:**

* BEV itself = **2D feature grid (x–y).**
* The **3D part** comes from regression heads that use BEV features to predict **z + height + orientation**, thanks to depth-aware lifting during feature construction.

---

Would you like me to also show how **this differs from occupancy networks**, where z is explicit in a 3D voxel grid rather than regressed? That contrast makes the trade-offs clearer.


# ADAS/AV Algorithm Evolution – BEV, Occupancy, End-to-End, VLM/VLA

## 1. Evolution Pipeline

* **Early ADAS (2000s–2015)**: Rule-based + few front cameras → lane-keeping, FCW.
* **Multi-Camera Fusion (2015–2020)**: CNNs fuse multiple RGB inputs → perception-first stacks.
* **BEV (2019–2022)**: Lift-Splat / BEVFormer → 2.5D feature map in ground plane.
* **BEVDepth (2022)**: Adds depth estimation per pixel → stronger 3D grounding than plain BEV.
* **Occupancy Networks / Occupancy Flow (2022–2023)**: Full 3D voxel grid with free-space + dynamics → perception+prediction unified.
* **End-to-End Driving (2023–2024)**: Vectorized planning directly from features → joint perception-prediction-planning.
* **VLM/VLA (2024– )**: Multimodal reasoning (vision+language+action) → explainability, instruction following.

---

## 2. Representation Dimensionality

| Method                    | Dimensionality                                  | Scope                                  | Notes                                            |
| ------------------------- | ----------------------------------------------- | -------------------------------------- | ------------------------------------------------ |
| **BEV (e.g., BEVFormer)** | 2.5D (top-down XY + semantic/depth channels)    | Perception → Detection/Tracking        | Efficient, loses vertical detail                 |
| **BEVDepth**              | 2.5D + depth per ray                            | Perception → 3D detection              | Better height awareness, more compute            |
| **Occupancy Networks**    | 3D voxel grid (XYZ occupancy + semantics)       | Perception + Prediction                | Heavy memory/compute; unifies freespace + motion |
| **End-to-End**            | Latent space (no explicit geometry)             | Perception + Prediction + Planning     | Removes modularity, harder to debug              |
| **VLM/VLA**               | Token space (vision + language + action tokens) | Reasoning, Explainability, Instruction | Not yet real-time, high compute cost             |

---

## 3. Pros/Cons Matrix

| Method                 | Pros                                         | Cons                                      |
| ---------------------- | -------------------------------------------- | ----------------------------------------- |
| **BEV**                | Efficient 2D convs, robust spatial layout    | Limited height info                       |
| **BEVDepth**           | Stronger 3D localization                     | Depth adds compute, noisy depth hurts     |
| **Occupancy Networks** | Full 3D scene, dynamic motion                | Huge memory, latency heavy                |
| **End-to-End**         | Simple pipeline, fewer hand-designed modules | Opaque, requires massive data             |
| **VLM/VLA**            | Multimodal, natural instruction following    | Far from real-time, requires large models |

---

## 4. NPU/Hardware Relevance

* **BEV/BEVDepth**: 2D conv-friendly, moderate memory, works well with int4/int8 quantization.
* **Occupancy**: Heavy voxel grids; bandwidth & SRAM bottleneck, needs sparsity compression.
* **End-to-End**: Large transformer-like blocks; token efficiency critical.
* **VLM/VLA**: Massive multimodal LLMs; edge deployment requires MoE + aggressive quantization.
