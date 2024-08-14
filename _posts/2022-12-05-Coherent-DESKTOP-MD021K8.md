---
title: Coherent Optical Communication
date: 2022-11-19 09:28:08
categories: 
- Language_Tool
tags: [Coherent]
description: revision control
typora-root-url: ../../allenlu2009.github.io
---



## Citation

[@liRecentAdvances2009] : 很好的 coherent optical communication review paper.

[@kikuchiFundamentalsCoherent2016] : 比較老派的 coherent optical communication review paper.

[@liSiliconMicroring2016] : 新的 micro-ring tunable laser for coherent optical communication.

[@parkCoherentKnocking2022] : coherent is knocking on the data center door



## Introduction

30 年前做的 coherent optical communication 前幾年又火紅起來。最近看了幾篇 review papers, 感覺還蠻能 catch up.

大致整理一下 30 年前的技術和這幾年的不同。

簡單說 30 年前 coherent optical communication 主要賣點是 sensitivity at receiver.   All optical Er-doped fiber amplifer 可以 distributedly amplify optical signal, 可以讓簡單的 intensity modulation + direct detection optical communcation system kill coherent communication.  

因此 IM-DD transmitter/receiver + all optical Er-doped fiber amplifer is the king!  做掉 coherent optical communication.



之後由於 WDM 和 DWDM 興起。Channel capacity 變成 key issue.  第一步是 narrow channel spacing, 第二步是增加 spectral efficiency (modulation efficiency).  IM-DD + optical filter 系統只能做到 WDM (>100GHz);  coherent optical communcation 因為可以結合 electrical filter (50GHz or smaller) 以及 better modulation efficiency (8PSK, 16QAM) 又復活！ 



不過重點是高速 DSP 能夠減輕之前 coherent optical communcation 的系統複雜度。像是 polarization, frequency drift, modulation/demodulation, FEC encode/decode.  

| Issues                          | 1992 Coherent Comm.                                          | 2022 Coherent Comm.                                          | Comment                    |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- |
| Sensitivity                     | (Electrical) LO gain at Rx only<br>Lose to (distributed) Er-doped fiber amplifier | (Optical) Er-doped fiber amplifier + LO gain at Rx           |                            |
| Laser phase noise               | (Electrical) Phase diversity reciver<br>Lose to IM-DD (Intensity modulation-direct detection) | (Optical) Narrow linewidth laser (for phase modulation) of 100's kHz FWHB | Micro-ring laser           |
| WDM wavelength accuracy         | Bulky tunable wavelength laser + auto-frequency control loop (AFC) | Silicon-integrated tunable wavelength laser + auto-frequency control loop | Micro-ring laser           |
| (D)WDM close wavelength spacing | (Optical) tunable wavelength laser + optical filter<br>Only WDM, not DWDM | (Electrical) coherent + RF filter<br>ITU G694.1defines 100GHz (0.8nm) for 40 channels, or 50GHz (04nm) for 80 channels |                            |
| SE (Spectral Efficiency)        | (Optical) IM-DD of low SE                                    | (Optical) PSK, QAM due to better laser + (electrical) DSP for phase lock |                            |
| Polarization                    | (Optical) Polarization control                               | (Electrical) Polarization diversity receiver using DSP       | Polarization shift keying? |
| Dispersion                      | (Optical) Dispersion compensation fiber                      | (Elecrical) Equalization using DSP                           |                            |
| Fiber nonlinearity              | (Optical) lower power, soliton, wavelength spacing           | (Electrical) nonlinearity mitigation using DSP               |                            |





## Why IM-DD to Coherent?

我們先看目前 IM-DD 的 status 以及問題。

首先 data rate 和距離如下圖：

* Data rate: 100G(bps) / 400G / 1.6T
* 距離:  Long haul (>100km) / Metro (300km) / Edge (80km) / Campus (10km) / Intra data center (0.5-2km)
* 直觀說 data rate 和距離成反比



<img src="/media/image-20221209222748077.png" alt="image-20221209222748077" style="zoom:50%;" />



### Coherent for Long Haul and Metro

目前 CMOS electrical serdes link 是 25Gbps using NRZ (Non-Return to Zero).  因此 100Gbps 需要 4 wavelength WDM.  如果使用 1.3um wavelength, fiber loss 是 0.5dB/km, 只能到達 20km 就需要 optical or electronic repeater.  如果使用 1.5um wavelegnth, fiber loss 是 0.2dB/km, 距離可以加倍。但是 fiber dispersion 又會限制距離。

因此在 Long haul 和 Metro 已經開始使用 coherent communication.

另外 coherent optical communication 可以使用 QPSK 以及 dual polarization quadruple (x4) per wavelength. 因此原來 100Gbps 的系統可以直接 upgrade 到 400Gbps per wavelength using the same fiber.

Coherent 的兩個好處:  

* 可以用 1.5um wavelength 加上 electronic dispersion compensation 加長 repeater link (40km/repeater).
* 可以 quadruple link capacity per wavelength

### Coherent for Data Center (2022)

目前 (2022) 最熱門的是 1.6T for data center.  目標是 2-10km.   傳統的 IM-DD 系統需要非常多的 wavelegnth.

因爲 IM-DD link 沒有 phase information, 只能在 amplitude 上做文章。因此推出 PAM4 系統增加 spectra efficiency 加倍 data rate per wavelength.  不過會讓距離更短，因爲更容易受到 impairment 的影響。

QSFP-DD (Double Density, PAM4) 的特性：

* Electrical:  8 links of 25Gbps (NRZ) for 200Gbps;  或是 8 links of 50Gbps (PAM4) for 400Gbps
* Optical:  8 wavelength?

IEEE 和 Optical Internetworking Forum (OIF) 已經 propose 800Gbps coherent link for campus network (2-10km), 不過 industry 思考 1.6Tbps coherent link.

**但是 coherent 的成本相對比較高。因此有所謂的 coherent lite communication 目標是 data center (2-10km) for 1.6Tbps.**

幾個節省成本的重點

* Fixed wavelength 而非 tunable wavelength
* Low power for lower overall system cost (density, cooling)





* 106/112 Gbps, 56Gbaud, PAM4

* retimer, POR switch

* multiple wavelength lasers: chromatic dispersion laser



TOR (top array switch):

* chip-to-chip
  * copackage optics: fiber in package 
  * inpackage optics:  customer (amazon, ...)
  * QSFP: opto
* 2m cable to fiber, TOR switch can be eliminated, no TOR !! eliminated 1-layer.   Power and thermal limit
  * switch is lower due to process advantage
  * DSP serdes (Lawrence Loh)
  * Infenaria is too expensse $20,000 module.  Infi acadia \$2000, integeated

* chip to module

224G

106/112 Gbps, 56Gbaud, PAM4

retimer, POR switch

## Reference

