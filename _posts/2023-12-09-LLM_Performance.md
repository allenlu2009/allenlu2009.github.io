---
title: LLM 性能分析
date: 2023-12-09 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

* Optimizing LLM Inference - A Performance Engineering Approach  https://www.linkedin.com/pulse/optimizing-large-language-model-inference-performance-engineering-ggzmf/

  

## Takeaway

* 





## LLM Service Important Metrics (指標)



#### 第一個 token 的時間 (TTFT: Time To First Token)

用戶输入 prompt 后,多快开始看到模型的输出? 实时互动中,低等待时间对于获取响应至关重要,但在离线工作中则不那么重要。这个指标由处理 prompt 和生成第一个输出标记所需的时间决定。

可以推廣到 Time to First N Token:  TTFNT



#### **每個輸出 token 的時間 (TPOT: Time Per Output Token)**

系统中的_每个_用户生成一个输出标记的时间。这个指标对应了每个用户对模型"速度"的感受。例如,100ms/token 的TPOT将是每个用户 10 token/sec, 即每分钟约450个字 (10 x 60 x 0.75 = 450) ,这比典型的阅读速度都快。



#### **延遲 (Latency)**

模型为用户生成完整回复的总时间。可以用前两个指标计算总体响应延迟:

Latency = _(TTFT)_ + _(TPOT)_ *(生成的 token number)。



#### **吞吐量 (Throughput)**:

推理服务器每秒可以为所有用户和请求生成的输出标记数 output tokens per second。



### 模型带宽利用率(MBU)

#### LLM推理服务器的优化程度如何?

如前面简要解释的那样,在较小的batch size下进行LLM推理(尤其是在解码阶段)的瓶颈在于我们从设备内存向计算单元加载模型参数的速度。内存带宽决定了数据移动的速度。我们引入了一个新的指标,称为模型带宽利用率(MBU),以测量底层硬件的利用率。



MBU定义 =  (achieved memory BW)/(peak memory BW)

Achieved memory BW =  (模型参数总大小, static + KV缓存大小, dynamic) / TPOT



#### **吞吐量**

我们可以通过 batch prompt trade-off 吞吐量和每个 token 的时间。在GPU评估期间对查询进行分组可以增加与顺序处理查询相比的吞吐量,但每个查询完成要花更长时间(忽略排队效应)。

有几种常见的批处理推理请求的技术:



#### **静态批处理**

客户端将多个提示打包成请求,并在批处理中的所有序列完成后返回响应。我们的推理服务器支持这一点,但不要求如此。

#### **动态批处理**

提示会在服务器内部动态批处理在一起。通常,这种方法的性能比静态批处理差,但如果响应很短或长度统一,则可以接近最佳。当请求具有不同的参数时,此方法效果不佳。



#### **持续批处理**

这篇优秀的论文提出了一种想法,即根据需要批处理到达的请求,这目前是最先进的方法。它不是等待批处理中的所有序列完成,而是在迭代级别上将序列分组。与动态批处理相比,它可以实现10倍到20倍的吞吐量改进。
