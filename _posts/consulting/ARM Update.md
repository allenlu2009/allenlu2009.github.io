**1.V8->V9 提价: ARM Royalty Rate 从2-3%提升到5%的原因, 有哪些新增产品, 从客户维度怎么考虑使用V9的ROI/ Payback**

We estimate royalty rates for v9 are 4%-5%, compared with the 1.7% blended rate the firm reported in its IPO filing for 2022.

V9 架构支持更高效的计算，适用于高性能智能手机、**数据中心处理器和 AI 应用 and auto** 等领域。



**2.ARM CSS (Compute Subsystem): CSS和V9相比有哪些新增的产品, ARM能收取10% Royalty的原因, 从客户维度怎么考虑使用CSS的ROI/ Payback?** 

- **新增产品**：CSS 是基于 V9 架构的完整计算解决方案，集成了 CPU、GPU 和其他组件，简化了芯片设计流程。
    [Arm Newsroom](https://newsroom.arm.com/blog/arm-css-for-client-platform?utm_source=chatgpt.com)
- **收取 10% 授权费的原因**：由于 CSS 提供了完整的计算平台，显著降低了客户的开发成本和时间，ARM 因此收取更高的授权费。
  
- **客户 ROI 和回收期**：客户需评估采用 CSS 所节省的开发成本和缩短的上市时间，衡量这些优势是否能抵消更高的授权费用。

1. AI Smartphone /AI PC - 由于ARM在AI设备的功耗优势, 可以收取更高的royalty吗, royalty rate的天花是多少 - 如果热/form factor不是瓶颈, AIPC &手机芯片价格/核数的天花板是多少? 

**3 如果手机/PC客户每年性能提升20-30%, 多少来自先进制成 vs 芯片厂自身工艺 vs ARM IP适配?**

**性能提升的来源**：每年 20-30% 的性能提升可能来自以下因素：

- **先进制程**：约 40% 的提升。
- **芯片厂自身方法**：约 30% 的提升。
- **IP **：约 30% 的提升。

这些比例可能因具体情况而异。





The **ARM v8** and **ARM v9** architectures represent significant advancements in ARM's CPU design, each introducing various features and improvements tailored for modern computing needs. Here’s a detailed comparison of the two architectures:

## **Key Differences Between ARM v8 and ARM v9**

## **1. Performance Enhancements**

- CPU core performance: frequency and IPC (Instructions per cycle)/micro-architecture
- AI: SVE2 (scalable vector) and CME (logic)/SME (matrix)
- Cache coherence / coherent fabric for multi-cores (CSS) for multi-cores
- 

![[Pasted image 20241118154547.png]]


![[Pasted image 20241118154606.png]]

The performance differences in CPU cores between ARMv8 and ARMv9 architectures are significant, particularly in terms of efficiency, speed, and capabilities tailored for modern workloads such as AI and machine learning. Here are the key aspects of these differences:


## **3. Instruction Set Extensions**

- **ARM v9** supports new instruction set extensions that are not available in ARM v8:
  
    - **Scalable Vector Extension 2 (SVE2)**: An evolution of the original SVE introduced in ARM v8, SVE2 allows for more efficient vector processing, particularly beneficial for applications requiring high-performance computing and machine learning tasks.
    - **Scalable Matrix Extension (SME)**: This new extension is designed to optimize matrix operations, which are crucial for AI and machine learning workloads [

## **2. Security Features**

- **ARM v9** incorporates advanced security features, such as:
  
    - **Confidential Compute Architecture (CCA)**: This provides hardware-based isolation for sensitive data processing, enhancing security against various attack vectors.
    - **Memory Tagging Extension (MTE)**: This feature helps to improve memory safety by tagging memory allocations, which aids in detecting memory corruption issues [
        


## **4. Virtualization Improvements**

- The virtualization capabilities have been enhanced in ARM v9, including better support for nested virtualization and improved performance for virtual machines [
  
    2
    
    ](https://www.arm.com/zh-TW/architecture/cpu/a-profile). This makes it more suitable for cloud computing environments where virtualization is essential.

## **5. Compatibility**

- ARM v9 is backward compatible with ARM v8, meaning that software developed for ARM v8 can run on ARM v9 processors without modification. This ensures a smooth transition for developers moving to the newer architecture while still leveraging existing applications.

## **Conclusion**

In summary, while both ARM v8 and ARM v9 architectures are designed to provide efficient performance and low power consumption, ARM v9 introduces significant enhancements in security, performance capabilities, instruction set extensions, and virtualization support. These advancements make ARM v9 particularly well-suited for modern applications like cloud computing, AI, and machine learning, reflecting the evolving demands of technology today.

The **ARMv8** and **ARMv9** architectures introduce several key differences in their instruction sets, reflecting advancements in performance, security, and support for modern workloads. Here are the primary distinctions:

### **1. Scalable Vector Extensions (SVE) and SVE2**
- **ARMv8** introduced the **NEON** instruction set for SIMD (Single Instruction Multiple Data) operations, which supports fixed-length vectors.
- **ARMv9** builds upon this with **SVE2**, which allows for variable-length vectors, enhancing flexibility in processing large data sets. SVE2 is designed to improve performance in applications such as machine learning and digital signal processing (DSP) by enabling more efficient vector operations and broader data handling capabilities [1][4].

### **2. Scalable Matrix Extension (SME)**
- **ARMv9** introduces the **Scalable Matrix Extension (SME)**, which provides new instructions specifically optimized for matrix operations. This is crucial for accelerating AI and machine learning workloads, allowing developers to perform matrix multiplications and other operations more efficiently than with the previous instruction sets [3][4].

### **3. Security Features**
- **ARMv9** enhances security through several new features:
  - **Pointer Authentication Codes (PAC)**: This feature helps protect against attacks that manipulate pointers in memory, making it harder for malicious code to exploit vulnerabilities.
  - **Branch Target Identification (BTI)**: This provides control flow integrity by ensuring that indirect branches only target valid locations, helping to prevent certain types of attacks like Return-Oriented Programming (ROP) [3] [4].
  - The introduction of the **Confidential Compute Architecture (CCA)** allows for secure execution of sensitive code even from authorized software, enhancing overall system security [3].

### **4. Virtualization Enhancements**
- Both architectures support virtualization, but **ARMv9** includes improved features such as enhanced nested virtualization capabilities and better support for secure hypervisors. This makes it more suitable for cloud computing environments where virtualization is critical [1][2].

### **5. Instruction Set Compatibility**
- ARMv9 maintains backward compatibility with ARMv8, meaning that existing software developed for ARMv8 can run on ARMv9 processors without modification. However, ARMv9 introduces new instructions that are not present in ARMv8, allowing developers to take advantage of the latest enhancements while ensuring legacy support [1][4].

### **Conclusion**
In summary, ARMv9 significantly advances the instruction set compared to ARMv8 by introducing SVE2 and SME for enhanced vector and matrix processing capabilities, improved security features such as PAC and BTI, and better virtualization support. These changes reflect ARM's focus on meeting the demands of modern computing environments, particularly in areas like AI and machine learning.

Citations:
[1] https://www.anandtech.com/show/16584/arm-announces-armv9-architecture
[2] https://www.arm.com/architecture/learn-the-architecture/a-profile
[3] https://www.trustonic.com/technical-articles/an-introduction-to-armv9-and-its-key-features/
[4] https://www.arm.com/zh-TW/architecture/cpu/a-profile
[5] https://www2.lauterbach.com/pdf/debugger_armv8v9.pdf
[6] https://en.wikipedia.org/wiki/ARM_architecture_family
[7] https://jsdevjournal.com/the-super-simple-guide-to-understanding-arm-processors/
[8] https://zh.wikipedia.org/zh-tw/ARM%E6%9E%B6%E6%A7%8B

Yes, ARMv9 enhances coherent bus performance compared to ARMv8. Here are the key improvements and features related to coherent bus performance in ARMv9:

### **1. Improved Coherent Memory Access**
- ARMv9 introduces enhancements that optimize how processors access shared memory, which is crucial for multi-core systems. This improvement leads to reduced latency and increased throughput for memory operations, making the coherent bus more efficient in handling data between cores.

### **2. DynamIQ Shared Unit (DSU) Enhancements**
- The **DynamIQ Shared Unit (DSU)** has been updated in ARMv9, specifically with the DSU-120 configuration. This update allows for better power management and area optimization while maintaining high performance. The enhancements in DSU contribute to improved coherence protocols, enabling faster communication between CPU cores and reducing the overhead associated with maintaining cache coherence.

### **3. Enhanced Cache Coherence Protocols**
- ARMv9 incorporates advanced cache coherence protocols that improve the efficiency of data sharing across multiple cores. These protocols ensure that all CPU cores have a consistent view of memory, which is essential for performance in multi-threaded applications.

### **4. Support for More Cores**
- ARMv9 architecture supports configurations that can scale up to 14 cores in a cluster while maintaining efficient coherence management. This scalability allows for higher performance in applications that benefit from parallel processing.

### **5. Low Power Modes**
- The new architecture includes low power modes that help reduce energy consumption during idle times without sacrificing performance when active, which is particularly beneficial for mobile and embedded devices.

### **Conclusion**
Overall, the enhancements in coherent bus performance in ARMv9 over ARMv8 make it better suited for modern applications requiring high efficiency and performance in multi-core environments, especially those involving complex workloads like AI and machine learning. These improvements contribute to a more responsive and efficient computing experience across various devices.

Citations:
[1] https://www.arm.com/zh-TW/resources/blueprint/armv9-cpus-consumer-devices
[2] https://www.anandtech.com/show/16584/arm-announces-armv9-architecture
[3] https://www.arm.com/zh-TW/architecture/cpu/a-profile
[4] https://www.trustonic.com/technical-articles/an-introduction-to-armv9-and-its-key-features/
[5] https://www.informationsecurity.com.tw/article/article_detail.aspx?aid=9143
[6] https://www.bnext.com.tw/article/63060/armv9-cpu-gpu
[7] https://jsdevjournal.com/the-super-simple-guide-to-understanding-arm-processors/
[8] https://www.arm.com/zh-TW/products/silicon-ip-cpu/neoverse/neoverse-v2