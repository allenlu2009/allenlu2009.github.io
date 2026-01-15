
### Slide 1 – Evolution: From Images → Image Editing → Video (Today’s Focus)

- **Phase 1: Image Generation (2022–2023)**
    
    - Diffusion models (Stable Diffusion, DALL·E, Imagen, Midjourney) made high-fidelity _still_ images cheap and ubiquitous.[Wikipedia](https://en.wikipedia.org/wiki/Text-to-video_model?utm_source=chatgpt.com)
        
    - Use cases: ad creatives, thumbnails, concept art, mood boards, low-cost A/B testing.
        
- **Phase 2: Image Editing & Compositional Control (2023–2024)**
    
    - Inpainting/outpainting, style transfer, background replacement, “generative fill” for photos.
        
    - Integration into mainstream tools (Adobe Firefly, Canva, Figma), plus more controllable pipelines (masks, reference images, LoRAs).
        
    - This phase is where **Apple** is most relevant: FastVLM and Apple Intelligence focus on _understanding_ and lightly transforming visual content on-device, not on full movie-length generation.[Apple Machine Learning Research+1](https://machinelearning.apple.com/research/fast-vision-language-models?utm_source=chatgpt.com)
        
- **Phase 3: Video Generation (2024–2025, your main topic)**
    
    - Text→short-video and image→video become usable at “marketing-grade” quality.
        
    - Western flagships: **OpenAI Sora 2**, **Google Veo 3.1**, **Runway Gen-4**.[OpenAI+2Google DeepMind+2](https://openai.com/index/sora-2/?utm_source=chatgpt.com)
        
    - China flagships: **Kuaishou Kling 2.0**, **Alibaba Wan 2.2**, **ByteDance Seedance / Seedream ecosystem**.[https://picma.magictiger.ai+2Alibaba Cloud+2](https://picma.magictiger.ai/blog/kling_2.0_launched_one_sentence_video_revolution.html?utm_source=chatgpt.com)
        
- **Emerging Phase 4: Video Editing & World-Model Style Simulation (2025→)**
    
    - Unified _creation + editing_: first/last frame control, multi-image reference, multi-shot storyboards, temporal extensions.
        
    - Research shifts toward **world models / spatial intelligence** (Meta V-JEPA 2, LongCat-Video, SANA-Video, “general world models” from Runway) as a bridge from “pretty clips” → “physical/simulation-grade” video.[Runway+4arXiv+4AI Meta+4](https://arxiv.org/abs/2506.09985?utm_source=chatgpt.com)
        

---

### Slide 2 – Landscape: Key Players & Relative Importance (2025)

**West – Creative Video**

- **Tier 1 (flagship, general-purpose)**
    
    - **OpenAI – Sora 2**: multi-scene, physically more accurate video + synchronized audio; now 15s for all users, 25s for Pro.[Bylo AI+3OpenAI+3IntuitionLabs+3](https://openai.com/index/sora-2/?utm_source=chatgpt.com)
        
    - **Google DeepMind – Veo 3 / 3.1**: 8s 720p/1080p clips with native audio, strong physics & prompt adherence, vertical support (9:16), integrated with Gemini & Flow and YouTube Shorts.[Lifewire+4Gemini+4Google DeepMind+4](https://gemini.google/overview/video-generation/?utm_source=chatgpt.com)
        
- **Tier 1.5 (pro-creator platforms)**
    
    - **Runway – Gen-3 Alpha / Gen-4**: widely used by independent creators and studios; 10s text/image→video, strong tools for compositing, keyframing, 4K upscaling; positions itself explicitly as a _“general world model”_ company.[Runway+3Runway+3DataCamp+3](https://runwayml.com/research/introducing-gen-3-alpha?utm_source=chatgpt.com)
        
- **Tier 2 (niche / vertical)**
    
    - **Pika, Luma, Synthesia, PixVerse, etc.** – more specialized in marketing, avatars, or social clips; important in usage but below Sora/Veo/Runway in frontier capability.[Synthesia+1](https://www.synthesia.io/post/best-ai-video-generators?utm_source=chatgpt.com)
        
- **Figure AI?**
    
    - **Figure AI is _not_ a creative video generator.** It’s a _humanoid-robot company_ using multimodal models and video data to train robots (Helix, etc.), with world-model flavor, but its “product” is physical robots, not text-to-video tools.[The Robot Report+1](https://www.therobotreport.com/figure-ai-raises-1b-in-series-c-funding-toward-humanoid-robot-development/?utm_source=chatgpt.com)
        

**China – Creative Video**

- **Tier 1 (consumer-scale flagships)**
    
    - **Kuaishou – Kling 2.0**: perhaps the most aggressive _duration + realism_ play; up to 2-minute 1080p@30fps clips with good physics and temporal consistency; deeply integrated into Kuaishou’s short-video ecosystem, 10M+ videos generated and >20M users reported in 2025 reviews.[Google Play+4https://picma.magictiger.ai+4Brandeploy+4](https://picma.magictiger.ai/blog/kling_2.0_launched_one_sentence_video_revolution.html?utm_source=chatgpt.com)
        
    - **Alibaba – Wan 2.2**: first large open-source MoE video model; strong in cinematic quality and controllability, widely embedded across third-party tools and clouds.[Alibaba Cloud+2Hugging Face+2](https://www.alibabacloud.com/en/press-room/alibaba-releases-wan2-2-to-uplift-cinematic?_p_lc=1&utm_source=chatgpt.com)
        
- **Tier 1.5 (ecosystem-driven)**
    
    - **ByteDance – Seedream 4.0 (image) + Seedance 1.0 (video)**:
        
        - Seedream 4.0 dominates high-quality / low-cost image gen + editing.
            
        - Seedance 1.0 is integrated into Envato’s generator alongside Kling and Veo, showing growing global footprint in video.[TechRadar](https://www.techradar.com/ai-platforms-assistants/want-to-try-ai-video-creation-envatos-removing-usage-limits-on-veo-3-and-kling-this-september?utm_source=chatgpt.com)
            
- **Other CN video players** (PixVerse, Hailuo, MiniMax Hailuo-02, LongCat-Video etc.)
    
    - Often lead on _price_ and _runtime efficiency_ (small DiT / linear attention, NFVP4), pushing down global unit economics for western providers.[arXiv+2arXiv+2](https://arxiv.org/abs/2509.24695?utm_source=chatgpt.com)
        

**Apple**

- **Apple is _not_ a frontline player in text-to-video generation as of late-2025.**
    
    - Public work is on **vision-language models (FastVLM)** and **on-device Apple Intelligence** focusing on _image_ understanding, UI-level “smart editing”, and text/image generation, not 1080p cinematic video synthesis.[Apple Machine Learning Research+2Apple+2](https://machinelearning.apple.com/research/fast-vision-language-models?utm_source=chatgpt.com)
        
    - CVPR 2025 mentions “video diffusion” in research, but no public Sora/Veo-style product.
        

---

### Slide 3 – Deep Dive: Kling 2.0 (Kuaishou)

- **Positioning**
    
    - Marketed explicitly as a Sora/Veo competitor; “Kling 2.0” launched April 2025 with strong PR around _physical realism and long duration_.[https://picma.magictiger.ai+1](https://picma.magictiger.ai/blog/kling_2.0_launched_one_sentence_video_revolution.html?utm_source=chatgpt.com)
        
    - Tight integration with **Kuaishou’s short-video app**, and also available via a standalone Kling app + third-party platforms (e.g., Envato).[Google Play+1](https://play.google.com/store/apps/details?hl=en&id=kling.ai.video.chat&utm_source=chatgpt.com)
        
- **Capabilities**
    
    - **Duration & resolution:** up to **2-minute** 1080p 30fps; some frontends (Kling mobile app) advertise up to 3-minute “video extension” flows.[https://picma.magictiger.ai+2Brandeploy+2](https://picma.magictiger.ai/blog/kling_2.0_launched_one_sentence_video_revolution.html?utm_source=chatgpt.com)
        
    - **Input modes:** text→video, image→video, and video extension/continuation; multi-image reference for consistent characters and scenes.[ir.kuaishou.com+2Yahoo Finance+2](https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-kling-ai-unveils-multi-image-reference-feature-further?utm_source=chatgpt.com)
        
    - **Quality metrics (from third-party reviews / demos):**
        
        - Strong physical plausibility (gravity, lighting, reflections) and temporal coherence (identity consistency) for everyday scenes; competitive with Veo 3 and first-gen Sora, sometimes surpassing them in duration.[https://picma.magictiger.ai+2Brandeploy+2](https://picma.magictiger.ai/blog/kling_2.0_launched_one_sentence_video_revolution.html?utm_source=chatgpt.com)
            
        - Weaknesses similar to peers: occasional artifacting on hands, text, very complex multi-object interactions.
            
- **Probable architecture**
    
    - Kuaishou has not fully disclosed details. External analysis and benchmarks strongly suggest a **diffusion-transformer (DiT) style video diffusion model** with:
        
        - Spatio-temporal latent tokens,
            
        - Block-sparse attention or windowed attention for long sequences,
            
        - RLHF-style fine-tuning for aesthetics and instruction following.
            
    - Functionally similar class to Wan 2.2, SANA-Video, LongCat-Video.[Hugging Face+3arXiv+3arXiv+3](https://arxiv.org/abs/2509.24695?utm_source=chatgpt.com)
        
- **Relative importance**
    
    - In China: **top-3** along with Wan & ByteDance’s stack.
        
    - Globally: one of the **most aggressive on duration and price**, and widely rebundled via aggregators (Envato, etc.), so you should treat Kling 2.x as _strategic_, not peripheral.[TechRadar+1](https://www.techradar.com/ai-platforms-assistants/want-to-try-ai-video-creation-envatos-removing-usage-limits-on-veo-3-and-kling-this-september?utm_source=chatgpt.com)
        

---

### Slide 4 – Deep Dive: Sora 2, Veo 3.1, Wan 2.2, Runway Gen-4

#### OpenAI – Sora 2

- **Model class:** closed-source, but continuity from Sora 1 suggests a **diffusion transformer defined over 3D space-time (world-model-like)** rather than simple frame-wise generation.[Reuters+1](https://www.reuters.com/technology/artificial-intelligence/openai-releases-text-to-video-model-sora-chatgpt-plus-pro-users-2024-12-09/?utm_source=chatgpt.com)
    
- **Key metrics**
    
    - Duration: 15s (standard), **25s for Pro**.[The Economic Times](https://m.economictimes.com/tech/artificial-intelligence/openai-allows-15-second-videos-for-all-sora-2-users-25-seconds-for-pro-users/articleshow/124594772.cms?utm_source=chatgpt.com)
        
    - Resolution: up to 1080p; multi-aspect ratios.
        
    - Inputs: text, image-to-video, video editing, “cameos” with reference people (now under stricter opt-in rules due to deepfake issues).[Superprompt+3OpenAI+3IntuitionLabs+3](https://openai.com/index/sora-2/?utm_source=chatgpt.com)
        
    - Strengths: best-in-class **multi-shot narrative consistency**, improved physical realism, lip-synced speech and environment audio, strong instruction following for complex prompts.[Superprompt+3OpenAI+3Medium+3](https://openai.com/index/sora-2/?utm_source=chatgpt.com)
        

#### Google DeepMind – Veo 3.1

- **Model class:** video diffusion with strong emphasis on cinematic “lens language” and safety (SynthID watermark).[The Verge+1](https://www.theverge.com/2024/12/4/24312938/google-veo-generative-ai-video-model-available-preview?utm_source=chatgpt.com)
    
- **Key metrics**
    
    - Duration: 8s clips (current consumer offering via Gemini).[Gemini+2GenApe 生成猿+2](https://gemini.google/overview/video-generation/?utm_source=chatgpt.com)
        
    - Resolution: 720p / 1080p; supports vertical 9:16.[The Verge](https://www.theverge.com/news/774352/google-veo-3-ai-vertical-video-1080p-support?utm_source=chatgpt.com)
        
    - Inputs: text, image→video, first/last frame control, “ingredients-to-video” (multi-image references), all with **native audio generation**.[Google Cloud+2Google DeepMind+2](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1?utm_source=chatgpt.com)
        
    - Strengths: prompt adherence, scene physics, and mobile/social integration (Gemini app, Flow, YouTube Shorts).
        

#### Alibaba – Wan 2.2

- **Model class:** **MoE video diffusion**; two-expert denoising (high-noise vs low-noise experts) to reduce compute while improving detail and motion.[Alibaba Cloud+2DeepLearning.ai+2](https://www.alibabacloud.com/en/press-room/alibaba-releases-wan2-2-to-uplift-cinematic?_p_lc=1&utm_source=chatgpt.com)
    
- **Key metrics**
    
    - Open-source: T2V-A14B / I2V-A14B (27B params total, ~14B active), TI2V-5B (5B) for efficient 720p@24fps on 4090-class GPUs.[Alibaba Cloud+1](https://www.alibabacloud.com/en/press-room/alibaba-releases-wan2-2-to-uplift-cinematic?_p_lc=1&utm_source=chatgpt.com)
        
    - Duration: typical open-source pipelines generate **~5s 480p–720p**; production SaaS frontends stretch longer via stitching/extension.
        
    - Strengths: cinematic aesthetics, controllable lighting/camera, improved physical law adherence vs Wan 2.1, and _Apache-style_ licensing that makes Wan a default choice for cost-sensitive workloads.[Media.io+3Alibaba Cloud+3Reuters+3](https://www.alibabacloud.com/en/press-room/alibaba-releases-wan2-2-to-uplift-cinematic?_p_lc=1&utm_source=chatgpt.com)
        

#### Runway – Gen-4 & A2D VLM

- **Model class:** transformer-based video diffusion; plus **Autoregressive-to-Diffusion (A2D)** research that turns autoregressive VLMs into parallel diffusion decoders.[Wikipedia+1](https://en.wikipedia.org/wiki/Gen-4_%28AI_image_and_video_model%29?utm_source=chatgpt.com)
    
- **Key metrics**
    
    - Duration: ~10s 1080p clips from text/image; strong I2V and editing; 4K upscale.[Wikipedia+2DataCamp+2](https://en.wikipedia.org/wiki/Gen-4_%28AI_image_and_video_model%29?utm_source=chatgpt.com)
        
    - Differentiator: heavy investment in **tools** (green-screen, motion brush, rotoscoping) + production pipeline (festival partnerships, film workflows).
        

---

### Slide 5 – Technology Comparison: Diffusion vs Autoregressive vs Hybrid

#### 1. Core generation paradigm

- **Video diffusion (dominant for creative video)**
    
    - Sora 2, Veo 3.1, Kling 2.0, Wan 2.2, Runway Gen-4, ByteDance Seedance, most CN models: all effectively **video diffusion transformers** with variations in:
        
        - Latent compression (3D VAE),
            
        - Attention structure (global vs block-sparse vs linear),
            
        - Mixture-of-Experts (Wan, some CN small models).[Reuters+4Alibaba Cloud+4arXiv+4](https://www.alibabacloud.com/en/press-room/alibaba-releases-wan2-2-to-uplift-cinematic?_p_lc=1&utm_source=chatgpt.com)
            
- **Autoregressive video models**
    
    - Pure token-by-token autoregressive video (e.g., early VLM-based decoders) is largely _too slow_ for high-res video.
        
    - AR is still key for **text & audio generation** (scripts, subtitles, dialogue) and some hybrid pipelines.
        
- **Hybrid AR→Diffusion (Runway A2D, diffusion language models)**
    
    - Runway’s **A2D-VLM** converts autoregressive VLMs into diffusion decoders with block-diffusion, enabling parallel token generation and KV caching.[Runway](https://runwayml.com/research/autoregressive-to-diffusion-vlms)
        
    - Research like SANA-Video and LongCat-Video combine **block-wise autoregressive temporal progression + diffusion denoising** to get minute-long videos at reasonable compute.[arXiv+1](https://arxiv.org/abs/2509.24695?utm_source=chatgpt.com)
        
    - This direction is a template for **next-gen Sora/Veo/Kling/Wan** when they push to >1–5 minute, 720p/1080p clips at scale.
        

#### 2. How they compare on key metrics (high-level, based on public reports/demos)

- **Video quality (resolution, fidelity)**
    
    - **Sora 2 / Veo 3.1 / Kling 2.0 / Wan 2.2** all hit **1080p**; Wan is especially important for open-source 720p/1080p.[Hugging Face+4Google DeepMind+4OpenAI+4](https://deepmind.google/models/veo/?utm_source=chatgpt.com)
        
    - Runway Gen-4 also produces high-fidelity 1080p with 4K upscale.[Wikipedia+1](https://en.wikipedia.org/wiki/Gen-4_%28AI_image_and_video_model%29?utm_source=chatgpt.com)
        
- **Instruction following (prompt adherence & control)**
    
    - **Best-in-class**: Veo 3.1 (ingredients, first/last frame), Sora 2 (storyboards, multi-shot prompts), Wan 2.2 (aesthetic prompt system), Runway (UI-driven control).[Wikipedia+5Google Cloud+5Google DeepMind+5](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1?utm_source=chatgpt.com)
        
    - Kling is improving, especially with multi-image references and video editing / replacement features, but external reviews still note edge cases in complex compositing.[Brandeploy+3ir.kuaishou.com+3Yahoo Finance+3](https://ir.kuaishou.com/news-releases/news-release-details/kuaishou-kling-ai-unveils-multi-image-reference-feature-further?utm_source=chatgpt.com)
        
- **Max duration**
    
    - **Kling 2.0**: up to **2 minutes 1080p@30fps** (longest among mainstream consumer-facing generators).[https://picma.magictiger.ai+2Brandeploy+2](https://picma.magictiger.ai/blog/kling_2.0_launched_one_sentence_video_revolution.html?utm_source=chatgpt.com)
        
    - **Sora 2**: 15–25s at high quality with multi-shot story structure.[The Economic Times+1](https://m.economictimes.com/tech/artificial-intelligence/openai-allows-15-second-videos-for-all-sora-2-users-25-seconds-for-pro-users/articleshow/124594772.cms?utm_source=chatgpt.com)
        
    - **Veo 3.1**: ~8s in consumer Gemini offering, though earlier Veo versions demonstrated >1-minute in research demos.[Gemini+2GenApe 生成猿+2](https://gemini.google/overview/video-generation/?utm_source=chatgpt.com)
        
    - **Wan 2.2**: open-source T2V models target 5s 480p–720p; longer sequences can be built via continuation.[Hugging Face+1](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B?utm_source=chatgpt.com)
        
- **Physical / spatial compliance**
    
    - **Sora 2** specifically markets improved physics and object permanence vs Sora 1 (e.g., fewer melting limbs, better 3D motion).[OpenAI+2Medium+2](https://openai.com/index/sora-2/?utm_source=chatgpt.com)
        
    - **Veo 3.1**: strong on cinematic camera, realistic acting, lighting; widely seen as a physics benchmark for short clips.[Google DeepMind+2Skywork+2](https://deepmind.google/models/veo/?utm_source=chatgpt.com)
        
    - **Kling 2.0 / Wan 2.2**: 3rd-party tests show competitive physical realism, especially for everyday motions and driving/camera shots; Wan claims top VBench-style metrics in multi-object interaction.[Brandeploy+2Reuters+2](https://www.brandeploy.io/en-kling-ai-2-0/?utm_source=chatgpt.com)
        

---

### Slide 6 – World Models, Spatial Intelligence, JEPA: Relevance to Video Gen

- **V-JEPA 2 (Meta)**
    
    - A **joint-embedding predictive architecture** trained on >1M hours of video + 1M images; predicts future visual tokens rather than generating pixels from scratch.[arXiv+1](https://arxiv.org/abs/2506.09985?utm_source=chatgpt.com)
        
    - Strength: _understanding_ and anticipating motion for action recognition, anticipation, and control.
        
    - Relevance to creative video:
        
        - Direct: not a production T2V model, but its learned dynamics could be used as a **critic / reward model** to improve physical plausibility of diffusion video.
            
        - Indirect: foundation for robotics, AR/VR, and “world simulators” (think: training Figure-style robots, not making TikTok ads).
            
- **LongCat-Video & SANA-Video (Meituan / others)**
    
    - Explicitly frame **long-video generation as a step toward world models**, using diffusion transformers with linear or block-sparse attention, constant-memory KV caches, and coarse-to-fine generation.[arXiv+2arXiv+2](https://arxiv.org/abs/2509.24695?utm_source=chatgpt.com)
        
    - These architectures are highly relevant to any Sora/Veo/Kling successor that wants “minutes-long, high-res video at manageable cost”.
        
- **Runway’s “General World Models”**
    
    - Runway’s research agenda explicitly talks about building **general world models** across text, images, video, 3D, and audio, and their A2D work is about closing the speed/quality gap.[Runway](https://runwayml.com/research/autoregressive-to-diffusion-vlms)
        
    - Today this manifests mainly as better controllability and consistency in creative video; tomorrow, it’s the bridge into simulation, games, and robotics.
        
- **Bottom line on world-model / JEPA tech for your talk**
    
    - For **2025 video products**, JEPA-style world models are **supporting tech**, not the main generators.
        
    - Over 2–3 years, expect:
        
        - World-model-style pretraining → better physics and long-horizon motion.
            
        - Video generators doubling as **simulators for robotics and agents** (NVIDIA Cosmos, Gemini Robotics, Helix are early signals in robotics).[WIRED+2The Verge+2](https://www.wired.com/story/nvidia-cosmos-ai-helps-robots-self-driving-cars?utm_source=chatgpt.com)
            

---

### Slide 7 – Capability Matrix: Image Gen, Image Editing, Video Gen, Video Editing

(✅ = strong product; ⚪ = present but secondary; R = mainly research / internal)

|Company|Image Gen|Image Editing / Inpainting|Video Generation|Video Editing / Control|Notes|
|---|---|---|---|---|---|
|**OpenAI**|✅ (DALL·E)|✅ (ChatGPT image editing)|✅ **Sora 2** T2V/I2V 1080p 15–25s|✅ Storyboard, cameos, re-rendering, audio|Flagship closed model; strong narrative + physics.[OpenAI+2Superprompt+2](https://openai.com/index/sora-2/?utm_source=chatgpt.com)|
|**Google DeepMind**|✅ (Imagen 3)|✅ (Gemini tools, Photoshop-style integrations)|✅ **Veo 3.1** 8s 720/1080p + audio|✅ First/last frame, ingredients-to-video, Flow, YouTube Shorts|Leading short-form video for social / mobile.[The Verge+3Google DeepMind+3Google Cloud+3](https://deepmind.google/models/veo/?utm_source=chatgpt.com)|
|**Kuaishou (Kling 2.0)**|⚪ (image tools)|⚪ (basic, app-level)|✅ **Kling 2.0** T2V/I2V, up to 2min 1080p@30fps|✅ Video extension, add/remove/replace elements, multi-image refs|China’s most aggressive long-duration consumer model.[Google Play+3https://picma.magictiger.ai+3Brandeploy+3](https://picma.magictiger.ai/blog/kling_2.0_launched_one_sentence_video_revolution.html?utm_source=chatgpt.com)|
|**Alibaba (Wan 2.2)**|✅ (Wan for images)|✅ (VACE editing pipeline)|✅ Open-source MoE Wan 2.2, 480p–720p 5s|✅ Unified create+edit via Wan-VACE and TI2V|Defines the open-source / low-cost benchmark.[Alibaba Cloud+2Hugging Face+2](https://www.alibabacloud.com/en/press-room/alibaba-releases-wan2-2-to-uplift-cinematic?_p_lc=1&utm_source=chatgpt.com)|
|**ByteDance (Seedream / Seedance)**|✅ **Seedream 4.0**|✅ (strong image editing, character-consistent gen)|⚪ Seedance 1.0 used via partners|⚪ Early compositional controls in BytePlus tools|Very strong in image; video still behind Kling/Wan but catching up.[TechRadar](https://www.techradar.com/ai-platforms-assistants/want-to-try-ai-video-creation-envatos-removing-usage-limits-on-veo-3-and-kling-this-september?utm_source=chatgpt.com)|
|**Runway (Gen-4)**|✅ (image tools)|✅ Pro-grade video editing & compositing|✅ Gen-3/Gen-4 10s 1080p + 4K upscale|✅ Timeline editing, motion brushes, keyframes, masking|“Pro-creator” hub; feature-complete editor + generator.[Runway+2DataCamp+2](https://runwayml.com/research/introducing-gen-3-alpha?utm_source=chatgpt.com)|
|**Apple**|✅ (Apple Intelligence image tools, FastVLM)|✅ On-device smart edits, generative photo tweaks|**R only** (research-level video diffusion; no public T2V)|⚪ Basic “memories”/montage style, not Sora-class|Apple is _not_ yet a Sora/Veo/Kling peer; focus is on on-device vision and privacy.[Apple Machine Learning Research+3Apple Machine Learning Research+3Apple+3](https://machinelearning.apple.com/research/fast-vision-language-models?utm_source=chatgpt.com)|
|**Figure AI**|–|–|– (uses video for training, not for content gen)|–|Robotics company; relevant for _physical AI_ and world-model usage, not creative video tools.[The Robot Report+1](https://www.therobotreport.com/figure-ai-raises-1b-in-series-c-funding-toward-humanoid-robot-development/?utm_source=chatgpt.com)|

---

### Slide 8 – Is Video _Editing_ the “Next Phase” After Video Generation?

Short answer: **yes, for creative workflows – but it splits into two tracks.**

1. **Track A – “Editing-First” Creative Tools**
    
    - After image gen → image editing → short video gen, the bottleneck is no longer _can I make a clip?_ but _can I precisely control & revise it?_
        
    - Evidence:
        
        - **Kling 2.0** new features: add/remove/replace objects in generated clips.[ir.kuaishou.com+1](https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-advances-20-era-empowering-everyone-tell-great-stories?utm_source=chatgpt.com)
            
        - **Wan 2.1 VACE & Wan 2.2**: “unified video creation and editing” as a core selling point.[Alibaba Cloud+1](https://www.alibabacloud.com/en/press-room/alibaba-releases-wan2-2-to-uplift-cinematic?_p_lc=1&utm_source=chatgpt.com)
            
        - **Veo 3.1**: first/last frame, ingredients-to-video (multi-shot consistency).[Google Cloud+1](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1?utm_source=chatgpt.com)
            
        - **Runway Gen-4**: entire brand is “video editor with AI”, not “raw model API”.
            
    - For business users, the _next wave of value_ is:
        
        - AI assisting in **editing existing footage** (ad variations, localization, removing/adding characters, language dubbing).
            
        - Style, continuity, and brand control rather than raw generation alone.
            
2. **Track B – “Simulation / World-Model” Video**
    
    - In parallel, JEPA, V-JEPA 2, LongCat-Video, SANA-Video, and robotics VLAs (GR00T, Gemini Robotics, Helix) push video toward **“physics engine + policy”** for robots, games, and virtual worlds.[The Verge+5arXiv+5AI Meta+5](https://arxiv.org/abs/2506.09985?utm_source=chatgpt.com)
        
    - This is not “editing” in the creative sense, but **controlling future frames** under constraints – crucial for _Physical AI_ (which you care a lot about).
        

So for your presentation, I’d frame the “next phase” as:

> **From “generate clips” → “edit & control clips” in the creative market, and from “render frames” → “simulate worlds” in the robotics/agent market.**

---

### Slide 9 – How to Position Kling vs Sora vs Veo vs Wan (Executive Takeaway)

- **Sora 2** – frontier on **multi-shot narrative + audio + physics**; short-form but high-quality, premium positioning.
    
- **Veo 3.1** – frontier on **short social clips** with strong prompt control and deep integration into Google’s consumer ecosystem.
    
- **Kling 2.0** – frontier on **duration and consumer scale**, especially in China; aggressive in pricing and deeply embedded in a huge UGC platform.
    
- **Wan 2.2** – frontier on **open-source and cost-efficient cinematic video**, likely to dominate:
    
    - open-source R&D,
        
    - budget-sensitive SaaS,
        
    - and as a _building block_ for others’ platforms.
        
- **Runway Gen-4** – frontier on **end-to-end creator workflow**, not just model quality; most important independent western studio+tooling player.
    
- **ByteDance** – huge latent threat: very strong in image (Seedream) and low-cost scale; video (Seedance) is already being plugged into western SaaS like Envato.