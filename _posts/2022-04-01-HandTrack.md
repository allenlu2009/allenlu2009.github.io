---
title: AI Hand Pose and Tracking
date: 2022-04-01 09:28:08
categories: 
- AI
tags: [Hand Tracking]
description: Combine CNN with Transformer
typora-root-url: ../../allenlu2009.github.io

---

Reference

[@chatzisComprehensiveStudy2020]

[@chenAwesomeHand2022]



手是我們的主要操作工具，它們在空間中的位置、方向和關節對於許多人機界面至關重要。 Hand pose estimation 對於 VR/XR、手語識別、手勢識別和機器人人機界面等各種應用非常有用。

Pose estimation 包含 hand pose 和 body pose estimation 目的在精確測量人體運動，有機會成爲新一代的自動化工具。雖然 hand pose estimation 和 body pose estimation 這兩個領域在目標和難度上有很多相似之處，但 hand pose estimation 有特別的問題需要解決，例如缺乏 characteristic local features, pose ambiguity, and self-occlusion.



## Hand Pose Estimation

Hand pose estimation 和其他的 computer vision task 例如 (object or people) detection, image segmentation 有一個基本的不同。**就是 hand pose estimation 或 hand pose reconstruction 都是 output 3D joint points.**  

比較 face 算法像是美顔、表情、識別，一般只要 output 2D feature points.  

像是 face recognition, body pose estimation.  不過 face recognition output feature points 一般只要 2D 就可以 (3D 只有在 structure light input 或是更安全的 anti-spoofing face recognition).  body pose estimation 也類似。如果是正面的 body pose, 2D 基本有一定應用。但是行走或是更複雜的 dancing, 則需要 3D.

hand pose 基本是 3D task：input 可以是 3D (Depth or RGB+D) 或是 2D (RGB, ill-posed), **但是 output 不論是 keypoints 或是 3D reconstruction 都是 3D output!**     2D hand pose output 基本沒有太大的用途 for hand sign, or user interface.





<img src="/media/image-20220403092456373.png" alt="image-20220403092456373" style="zoom: 50%;" />

傳統的 vision-based **3D hand pose estimation** 主要靠 depth information 判斷，需要 stereoscope 或是 depth camera (e.g. ToF, structure light)。不過這些硬體都有 cost 以及 overhead (e.g. calibration).  Depth camera 一般在戶外都有一些限制。

因此另一個方式是使用 monocular RGB camera 估計 hand joint locations for both hands.  **2D or 3D?**

2D 比較簡單，可以用在 social media only.

**但如果要精確的 UI, 應該還是要 3D hand pose estimation!**  這是一個 ill-posed problem!!! 從 2D 的 video 得到 3D hand pose video!!!

**hand detection --> crop hand bounding box --> use time sequence to (1) predict the future position; (2) reconstruct the 3D hand position!** 



The difficult problem!

Input:  RGB (no D) video 

Output:  21/24/26 keypoints sequence in 3D space！ 



The more difficult problem!

Input:  mono video 

Output:  21/24/26 keypoints sequence in 3D space！ 



Technique:

Time sequence --> RNN + CNN?

SSL --> for hand



接下來討論如何進行

1.  Develop a hand post estimation pipeline: hand detection, hand landmark and prediction (tracking), estimation model.
2.  Data-augmentation and learning methodology for model training



**Hand detection and Track**

Use a simple and efficient CNN architecture modified by YOLOv4, simultaneously localize and classifies.  這和 surveillance camera 用 YOLOv4 做 people detection 類似。



Track and prediction!!



**Hand landmark and keypoint (關鍵點) estimation**

在 hand detection 之後做 crop bounding box.  再來用 cropped image 預測 key-point.  這裏可以用 OpenXR 定義的 26 key-points.   OpenXR 規範中定義的 26 個關鍵點 (4 for thumb finger, 5 for the other four fingers, 1 for wrist, 1 for palm)，如下圖。

我們的 keypoint estimation network 使用上述 hand detection 步驟中的 crop image 的 **predicted bounding box**。 這與以前的做法不同，以前的做法關鍵點只基於每個圖像而非 predicted bounding box。 如果手被物體遮擋或在框架中僅部分可見，這是有問題的。 為了克服這個問題，我們的模型考慮了從先前幀中提取的信息。 因此，我們的模型明確地將外差關鍵點作為額外的網絡輸入。 



<img src="/media/image-20220402203003556.png" alt="image-20220402203003556" style="zoom: 50%;" />



以下是 Unity 的 hand pose estimation, 一共有 26 關鍵點。

<img src="/media/image-20220402204718342.png" alt="image-20220402204718342" style="zoom:50%;" />

以下是 Ultraleap 的 hand pose estimation, 一共有 24 關鍵點 (5x4+4)。

<img src="/media/image-20220402202141960.png" alt="image-20220402202141960" style="zoom:67%;" />



<img src="/media/image-20220402205010166.png" alt="image-20220402205010166" style="zoom:33%;" />

Collabora 是 21 關鍵點 (5x4 + 1) + prediction。網絡產生 **2D heatmap** for each of the 21 **predicted keypoints**.  也就是不是 100% 的當幀的結果，而是 predicted 下一幀的機率 (use 2D heatmap)。 Constructed by evaluting a Gaussian following, which is processed by a simple regression model to calculate the final landmarks.





### 2D Hand Pose and 3D Hand Pose

如果是有 depth 的 image 





Collabora 對在 XR 中使用手部姿勢估計特別感興趣，因為該應用程序與我們在 Monado（世界上第一個開源 OpenXR 運行時）上的工作非常吻合。 

