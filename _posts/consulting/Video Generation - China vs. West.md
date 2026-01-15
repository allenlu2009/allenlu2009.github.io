

**Introduction:**

Assistant General Manager of the Computing and Artificial Intelligence Technology Group at MediaTek. Since 2020, he has led the algorithm and software teams for edge AI development and deployment in phone, camera, tablet, and TV products. His focus spans AI applications such as image and video object detection, quality enhancement, understanding, and large language models for text generation, summarization, and agent-based applications. Dr. Lu also oversees the AI technology roadmap planning at MediaTek.

Prior to joining MediaTek, Dr. Lu served as the General Manager of the Video IoT (iVoT) Business Unit at Novatek, where he was responsible for business and technology planning and execution. Under his leadership, iVoT became a market leader in the surveillance and dash cam industry. In 2017, the unit developed the first SoC integrating an AI accelerator with a 4K-resolution ISP (image signal processing) and video codec for surveillance cameras. This innovation replaced the previous discrete solution, achieving commercial success.

Before Novatek, Dr. Lu founded Afatek, a fabless semiconductor company that developed digital TV receiver technology. Afatek introduced the first silicon integrating RF front-end with a demodulator for digital TV in 2006, followed by the first silicon integrating RF front-end with a multi-standard modulator for surveillance applications. Afatek was acquired by iTE Tech in 2008.

Dr. Lu began his career in Silicon Valley, working at Excess Bandwidth Corp., a startup funded by Stanford professors specializing in signal processing. Excess Bandwidth Corp. was acquired by Conexant in 2000. Prior to this, Dr. Lu was a member of the technical staff at Hewlett Packard.

**Relationship with video generation**
1. use generative AI as key selling features for smartphone and AI PC.  recently, I am looking into on-premise server.   LLM, image, and video generation are selling features.


**China vs. West**
China leading company: Kling from Kuaishou, Hailuo, Seeddance

Quality:  mainly 1080p 30fps up to 2 min, good enough quality for short social video, youtuber, vlog context providers., and fast iteration Kling2.5 "china speed".

Veo3: 4K 30fps for 10 seconds high quality and HD for professional use, cinama-grade ads, studio
Fast/Turbo with lower quality but 80% cost reduction

Sora2: 1080p up to 30fps with 30 seconds but with good quality. realistic short clips.

**Monetization model**
China: freemium (first month free and daily free credits) and aggressive pricing.
plus $10 monthly subscription fees with generous usage.

Veo: enterprise SaaS and pay-per-use case, charged by second.

Sora: bundle with $20 or $200 ChatGPT subscription or API charge with usage limits

**Expect a hybrid model globally.**
a low monthly subscription with sufficient credits, plus pay-as-you-go for heavy use.
**Chinese model intensified the price world**. West will sacrifice margins and shift to volume-based economics.


**Real-Time Generation**

On the horizon but there are still tech challenges:
1. stable quality - checked
2. instruction following - improving
3. longer duration - still in progress
4. (optional) compliant with physical law for physical world simulation

Two tech branches:  AR transformer model for video generation (good candidate) and streaming diffusion (good quality but need more progress)

There are new technologies under development like spatial intlligence by Fei-Fei Li, or world model by a couple of startups


**Ecosystem Moat**

Social Network effect (China): Kuaishou and ByteDance
Data flywheel
Workflow integration (OpenAI, Google)

Not winner take all
Regional preferrance
APAC: creator-led and consumer - centric

North America: enterprise and professional

Europe: not clear, regulation compliance



🎨 5. Alibaba's Open Animation Models

  Wan Series - Comprehensive Video Generation Platform

  🚀 Latest Releases:

  Wan2.2-S2V (Speech-to-Video) - August 2025

  - Breakthrough Innovation: Converts portrait photos into film-quality avatars
  - Capabilities: Speaking, singing, and performing from single image + audio clip
  - Quality: Professional film-quality output
  - Accessibility: Open source via Hugging Face, GitHub, and ModelScope
  - License: Apache 2.0 for commercial use

  Wan2.2 - July 2025

  - Resolution: Native 1080p HD output without upscaling
  - Architecture: Mixture of Experts (MoE) with high/low-noise expert models
  - Multimodal Support:
    - Text-to-Video (T2V-A14B)
    - Image-to-Video (I2V-A14B)
    - Text+Image-to-Video (TI2V-5B)
  - Current Output: Up to 720p, with 1080p planned

  Wan2.1 Series - February 2025

  - Wan2.1-FLF2V-14B: Start/end frame input for controlled video creation
  - Wan2.1-VACE: First open-source unified video creation and editing solution

  📊 Impact: Over 30 million downloads across open-source platforms, making it one of the most successful
  open-source video generation tools globally.

  ---
  🖼️ 6. ByteDance's New Image Models

  Seedream 4.0 - September 2025

  🏆 Market Position: Claimed to surpass Google DeepMind's "Nano Banana" across key indicators

  🔧 Technical Capabilities:
  - Unified Architecture: Combines generation + editing in single model
  - Resolution: 4K high-fidelity output
  - Speed: Significantly faster inference than competitors
  - Advanced Features: Knowledge-based generation, complex reasoning, reference consistency

  💰 Competitive Pricing: $30 per 1,000 generations (same as previous models despite enhanced capabilities)
  - $0.03 per image via BytePlus
  - 200-image free trial for new users

  Additional 2025 Models:

  Seed3D 1.0 (October 2025):
  - Innovation: Single image to simulation-grade 3D assets
  - Architecture: Diffusion transformer-based
  - Performance: Outperforms both open and closed-source alternatives

  InfiniteYou:
  - Specialty: Character-consistent image generation
  - Feature: Preserves facial characteristics across different scenarios

  🌍 Global Impact: ByteDance models are gaining international adoption due to high quality + competitive pricing,
  challenging US AI companies despite chip export controls.

  ---
  🍎 7. Apple's Open Vision Model

  FastVLM - CVPR 2025

  🎯 Core Innovation: Efficient Vision-Language Models for on-device processing

  🏗️ Technical Architecture:
  - FastViTHD: Novel hybrid vision encoder
  - Token Efficiency: Outputs fewer tokens, reduces encoding time
  - High-Resolution Support: Up to 1152×1152 pixels

  ⚡ Performance Breakthroughs:
  - 85x faster Time-to-First-Token vs LLaVA-OneVision-0.5B
  - 3.4x smaller vision encoder
  - 7.9x faster TTFT for larger variants

  📱 Device Integration:
  - Apple Silicon Optimization: Three models (0.5B, 1.5B, 7B variants)
  - Cross-Platform: iPhone, iPad, Mac compatibility
  - Real-Time Applications: Camera description, receipt scanning
  - Privacy-First: On-device processing aligns with Apple's privacy philosophy

  🔓 Open Source Commitment: Full GitHub release with inference code, model checkpoints, and iOS/macOS demo app
  using MLX framework

  ---
  🏆 Strategic Significance:

  1. Alibaba: Democratizing professional video creation with 30M+ downloads
  2. ByteDance: Challenging Western AI dominance with cost-effective, high-quality solutions
  3. Apple: Advancing on-device AI while maintaining privacy standards


