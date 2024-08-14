---
title: Pesudo Code for GPT-4o
date: 2024-05-25 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---



## Source

[GPT-4o四步併成一步走_效能不打折 (youtube.com)](https://www.youtube.com/watch?v=YxOJF0of-lU)



## 前言



## 示例

Step 1:  upload e3sconf_icobar23_02152.pdf

Step 2:  提供以下的 prompt.   這個 prompt 非常像 python code.  效果不錯。

prompt:

請先閱讀提供給你的pdf之後 

```python
entities = 列舉所有的實體清單(應該包含企業名稱、人名、產品名以及技術名，排除本文或參考文獻作者，請確認不要遺漏) print(entities) 

# 開始整理實體細節
for e in entities:     
	將該實體 e 以 markdown 二級標題輸出     
	從提供的 pdf 文本中提煉出與該實體相關之內容，以無序列表的形式列出
	
# 開始整理實體關係
for e in entities:    
	用[{e}]->[關係]->[實體B]表示所有可以抽取出來的實體關係，    
	以[{e}]->(屬性)->[屬性值]來表示實體屬性

```

  #  

僅根據前面已經整理出來**全部的**實體關係

以及實體屬性內容參考下方MERMAID  ER DIAGRAM的範例

來改以這種方式呈現 (所有中文內容請改為英文) 

```
"""
erDiagram
    CAR ||--o{ NAMED-DRIVER : allows
    CAR {
        string registrationNumber PK
        string make
        string model
        string[] parts
    }
    PERSON ||--o{ NAMED-DRIVER : is
    PERSON {
        string driversLicense PK "The license #"
        string(99) firstName "Only 99 characters are allowed"
        string lastName
        string phone UK
        int age
    }
    NAMED-DRIVER {
        string carRegistrationNumber PK, FK
        string driverLicence PK, FK
    }
    MANUFACTURER only one to zero or more CAR : makes
""" 
```

使用 Mermaid Live editor  表示以上流程圖： [Online FlowChart & Diagrams Editor - Mermaid Live Editor](https://mermaid.live/edit#pako:eNqFU12P2jAQ_Csrv_Qlh1SpakXeUq6HKiFAPar2IRLy2UtiNdlN_VGKOP577QYExx3gF8frsXdmMt4KxRpFLtDeG1lZ2ZYEcRTz-eQLPD_f3fEWHmY_R7PpFHLopPWEFvVybXx9Ct32izSct4YqINkilKLougbhK6lBKY4YLT3CigNp1BH0fvjp4-n2_ooapf4dYk-0LqJGoUPrDXEGo-IA3_XTgeNFHg_8VzERLFDVxA1XGxhbDt11Vh9us1oEVSNVGUxxDQtpOjRZmteSzjlOiteWGvqDzpsqtnZH2GUZ0liYyCe2UDjHykhvmK6KGA5vi_ghXR3LnimD-8FokMH3x3OLx_Mr9o4bfpINzC3roBKlaIdfs_114T_tfUh39rFa8uqN6N3aT17lgOSCRbdU3HaNkaTwJJ9nHSfF59m3dCZCebO3vC--FmdNVfvkT1KC9p3bV94wNIZLmyT8AH9ZvByFA6HOskfl3dHsM4Ah1QSdUiIy0aJtpdHx5f5nXQpfY4ulyOOnxpUMjU8tdxEqg-fHDSmRexswEzH1VS3ylWxcXIUuBWb_-HvI7h_5FkWf)

<img src="/media/image-20240526201548699.png" alt="image-20240526201548699" style="zoom:50%;" />

## GPT-4o 回復

### Entities

The PDF includes the following entities:

- Apple
- Foxconn
- Fair Labor Association (FLA)
- Global Production Network (GPN)

### Entity Details

#### Apple

- **Apple** is the largest electronics company in the world, collaborating with Foxconn, a Taiwan-owned electronics manufacturing service provider.
- Apple's alliance with Foxconn began with the manufacture of the iPod in 2006.
- The partnership has significantly increased Apple's competitiveness in the global market.

#### Foxconn

- **Foxconn** is the largest electronics company aimed at designing and assembling multinational brand electronics globally.
- It plays a crucial role in increasing Apple’s production and competitiveness.
- The company has faced criticism regarding heavy manual labor productivity pressures without adequate guarantees of workers' rights.

#### Fair Labor Association (FLA)

- **Fair Labor Association (FLA)** is involved in protecting workers' rights worldwide.
- It has conducted independent investigations into Apple’s suppliers, including Foxconn, to ensure the implementation of a Code of Work Ethics.

#### Global Production Network (GPN)

- **Global Production Network (GPN)** theory is used to analyze production networks across economic boundaries, including the roles of companies like Apple and Foxconn.
- GPN involves various actors, including the workforce, which is significant in both the company’s and the state’s structures and global capital.

### Entity Relationships

#### Apple

- **Apple** -> [partnered with] -> **Foxconn**
- **Apple** -> [ensures] -> [competitiveness]
- **Apple** -> [alliance] -> [Foxconn] (started with iPod production in 2006)
- **Apple** -> (largest electronics company) -> [global market]

#### Foxconn

- **Foxconn** -> [increases] -> **Apple** [production]
- **Foxconn** -> [faces] -> [criticism] (regarding labor practices)
- **Foxconn** -> [plays role in] -> **GPN**
- **Foxconn** -> [investigated by] -> **FLA**

#### Fair Labor Association (FLA)

- **FLA** -> [protects] -> [workers' rights]
- **FLA** -> [investigates] -> **Foxconn**
- **FLA** -> [implements] -> [Code of Work Ethics]

#### Global Production Network (GPN)

- **GPN** -> [analyzes] -> [Apple and Foxconn production network]
- **GPN** -> [includes] -> [workforce] (significant in company and state structures)

### Mermaid ER Diagram

```
erDiagram
    APPLE ||--o{ FOXCONN : partnered_with
    APPLE {
        string name "Apple Inc."
        date founded "1976"
        string headquarters "Cupertino, CA"
    }
    FOXCONN {
        string name "Foxconn Technology Group"
        date founded "1974"
        string headquarters "Tucheng, New Taipei, Taiwan"
    }
    FLA ||--o{ FOXCONN : investigates
    FLA {
        string name "Fair Labor Association"
        date founded "1999"
        string headquarters "Washington, D.C., USA"
    }
    GPN {
        string name "Global Production Network"
    }
    FOXCONN ||--o{ GPN : part_of
    APPLE ||--o{ GPN : part_of
    APPLE ||--o{ FLA : ensures_compliance_with
    FOXCONN ||--o{ LABOR : employs
    LABOR {
        string rights "worker's rights"
        string conditions "working conditions"
    }
    FLA ||--o{ LABOR : protects
    GPN ||--o{ LABOR : includes

```



使用 Mermaid Live editor  [Online FlowChart & Diagrams Editor - Mermaid Live Editor](https://mermaid.live/edit#pako:eNqFU12P2jAQ_Csrv_Qlh1SpakXeUq6HKiFAPar2IRLy2UtiNdlN_VGKOP577QYExx3gF8frsXdmMt4KxRpFLtDeG1lZ2ZYEcRTz-eQLPD_f3fEWHmY_R7PpFHLopPWEFvVybXx9Ct32izSct4YqINkilKLougbhK6lBKY4YLT3CigNp1BH0fvjp4-n2_ooapf4dYk-0LqJGoUPrDXEGo-IA3_XTgeNFHg_8VzERLFDVxA1XGxhbDt11Vh9us1oEVSNVGUxxDQtpOjRZmteSzjlOiteWGvqDzpsqtnZH2GUZ0liYyCe2UDjHykhvmK6KGA5vi_ghXR3LnimD-8FokMH3x3OLx_Mr9o4bfpINzC3roBKlaIdfs_114T_tfUh39rFa8uqN6N3aT17lgOSCRbdU3HaNkaTwJJ9nHSfF59m3dCZCebO3vC--FmdNVfvkT1KC9p3bV94wNIZLmyT8AH9ZvByFA6HOskfl3dHsM4Ah1QSdUiIy0aJtpdHx5f5nXQpfY4ulyOOnxpUMjU8tdxEqg-fHDSmRexswEzH1VS3ylWxcXIUuBWb_-HvI7h_5FkWf)



```mermaid
erDiagram
    APPLE ||--o{ FOXCONN : partnered_with
    APPLE {
        string name "Apple Inc."
        date founded "1976"
        string headquarters "Cupertino, CA"
    }
    FOXCONN {
        string name "Foxconn Technology Group"
        date founded "1974"
        string headquarters "Tucheng, New Taipei, Taiwan"
    }
    FLA ||--o{ FOXCONN : investigates
    FLA {
        string name "Fair Labor Association"
        date founded "1999"
        string headquarters "Washington, D.C., USA"
    }
    GPN {
        string name "Global Production Network"
    }
    FOXCONN ||--o{ GPN : part_of
    APPLE ||--o{ GPN : part_of
    APPLE ||--o{ FLA : ensures_compliance_with
    FOXCONN ||--o{ LABOR : employs
    LABOR {
        string rights "worker's rights"
        string conditions "working conditions"
    }
    FLA ||--o{ LABOR : protects
    GPN ||--o{ LABOR : includes
```



以下是 image capture 圖

<img src="/media/image-20240526153814315.png" alt="image-20240526153814315" style="zoom:80%;" />



