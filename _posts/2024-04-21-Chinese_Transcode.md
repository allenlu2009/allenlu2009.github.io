---
title: 中文編碼，亂碼，轉碼
date: 2024-04-21 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
typora-root-url: ../../allenlu2009.github.io


---





## Source

* [彻底搞明白 GB2312、GBK 和 GB18030 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/453675608)
* [線上文字亂碼還原工具 - ToolsKK 線上工具網](https://www.toolskk.com/garbled-text-recover)
* [寫代碼總是莫名其妙的亂碼？這一篇教你從原理出發徹底幹掉它 - 每日頭條 (kknews.cc)](https://kknews.cc/zh-tw/code/3qp3goa.html)
* [三類亂碼原因分析 - 每日頭條 (kknews.cc)](https://kknews.cc/tech/zpkx5ql.html)
* [有亂碼？口算修復下 - 每日頭條 (kknews.cc)](https://kknews.cc/tech/boen356.html)







## 字符集 & 字符編碼

**字符集**：一個系統支持的所有抽象字符的集合。

**字符編碼**：在符號集合與數字系統之間建立的映射關係，是一種對應法則。

常見的字符集有以下幾種：ASCII字符集、GB2312字符集、BIG5字符集、GB18030字符集、Unicode字符集等。通常來說，一種字符集對應一種字符編碼，但也有例外，比如Unicode字符集，對應的字符編碼就是UTF-32,UTF-16和大名鼎鼎的UTF-8。

## 編碼&解碼

**編碼**：將源對象內容（文字、符號等）按照一種標準轉換為一種標準格式（數字編碼）內容。

**解碼**：編碼的逆過程，它使用和編碼相同的標準將編碼內容還原為最初的對象內容（文字符號）。

這麼說有些抽象，舉個例子：

在一個GBK編碼的文件中，有個「碼」字，那麼在存儲時，就會將其用GBK編碼，保存為「C2EB」。當你打開文件時，又用GBK對其解碼，就展示為之前的「碼」字了。亂碼就是用其他解碼方式。

## ASCII 碼

**ASCII字符集：**主要包括控制字符（回車鍵、退格、換行鍵等）；可顯示字符（英文大小寫字符、阿拉伯數字和西文符號）。

**ASCII編碼：**將ASCII字符集轉換為計算機可以接受的數字系統的數的規則。使用7位（bits）表示一個字符，共 128 字符。

ASCII碼是正式投入計算機使用的第一種編碼，也是後起之輩GBK，UTF-8等等編碼的鼻祖，因此在後續發展的所有編碼中，對於ASCII碼中出現的字符都保持了一致。

## 中文編碼：Unicode, BIG5, GB

UTF 是後來全球統一的編碼。

### BIG5

適用於台灣、香港地區的一個繁體字編碼方案。使用了雙八碼存儲方法，以兩個字節來安放一個字，第一個字節稱為"高位字節"，第二個字節稱為"低位字節“。一級 （常用）漢字 5401 個，二級（不常用）漢字 7652 個，一共 13060 個繁體漢字。



### GB (國標)

GB 是一個系列，源自於大陸的簡體中文編碼，擴展至繁體字以及日韓的漢字。

#### GB2312 （1980，英文 1-byte, 簡體中文 2-byte）

1980 年，中国发布了第一个汉字编码标准，也即 GB2312 ，全称 《信息交换用汉字编码字符集·基本集》，通常简称 GB （“国标”汉语拼音首字母）是對 ASCII 的中文擴展。**兩個大於127的字符連在一起時，就表示一個漢字**。高字節從0xA1用到 0xF7，低字節從0xA1到0xFE， 共收录了 **6763** 个常用的汉字和字符，此标准于次年5月实施，它满足了日常 99% 汉字的使用需求。

GB2312 是 two-byte encoding.   有效的编码范围如下图所示。红色栏 (0-127) 表示 ASICII 的编码范围，绿色栏表示 GB2312 编码范围。

<img src="/media/image-20240421224258828.png" alt="image-20240421224258828" style="zoom:50%;" />



#### GBK（1995，英文 1-byte, 繁簡中文 2-byte)

由于有些汉字是在 GB2312 标准发布之后才简化的，还有一些人名、繁体字、日语和朝鲜语中的汉字也没有包括在内，所以，在 GB2312 的基础上添加了这部分字符，就形成了 GBK ，全称 《汉字内码扩展规范》，共收录了 **20000** 多个汉字和字符，它完全兼容 GB2312

<img src="/media/image-20240421224614577.png" alt="image-20240421224614577" style="zoom: 50%;" />



#### GB18030 (英文 1-byte, 繁簡中文 2/4-byte)

GB18030 全称《信息技术 中文编码字符集》 ，共收录 **70244** 汉字和字符， 它在 GBK 的基础上增加了中日韩语中的汉字 和 少数名族的文字及字符，完全兼容 GB2312，基本兼容 GBK

GB18030 发布过两个版本，第一版于 2000 年发布，称为 GB18030-2000，第二版于 2005 年发布，称为 GB18030-2005

与 GBK 不同的是，GB18030 是变长多字节字符集，每个字或字符可以由一个，两个或四个字节组成，所以它的编码空间是很大的，最多可以容纳 16M 个字符

由于需要兼容 GBK，四个字节的前两个字节和 GBK 编码保持一致，GB18030 具体的编码范围如下

<img src="/media/image-20240421231925312.png" alt="image-20240421231925312" style="zoom: 50%;" />



## Unicode (碼點) 

Unicode是一种字符标准，我們稱爲碼點 (code point)，旨在统一表示世界上所有语言的字符集。它使用数字来表示文本中的每个字符，包括字母、数字、标点符号、符号和特殊字符。Unicode定义了一个庞大的字符集，目前包括超过143K个字符，覆盖了几乎所有的语言和符号系统。

 **Unicode** 码点，最多可表示 $2^{31}$（4-byte, 大约 2B）个字符。

CJK（中日韩）统一表意文字（Common Ideographs）**在 unicode 编码中的表示范围是从U+4E00到U+9FFF**

，即常用的中文汉字和一些特殊用途的汉字。

Unicode的主要特点包括：

1. **统一性**：Unicode致力于统一不同语言和符号系统的字符表示，使得任何一种字符在不同计算机系统和应用程序中都能正确地显示和处理。
2. **多字节编码 (UTF-8/16/32)**：为了容纳如此庞大的字符集，Unicode使用了多字节编码方案，允许每个字符的表示采用1到4个字节不等的编码长度。
3. **扩展性**：Unicode不断扩展字符集，以适应新的语言、符号和特殊字符的需求，确保未来的字符能够得到正确的表示和处理。
4. **字符标准化**：Unicode还定义了字符的标准化形式，包括组合字符、规范等价性和兼容性等，以确保在不同环境中字符的显示和处理一致性。

### UTF8/16/32 (編解碼)

UTF是一系列的多字节编码方案，可以表示 **Unicode** 码点。 UTF 包括UTF-8、UTF-16和UTF-32等。

#### UTF8 (中文 variable 3-byte or 4-byte)

其中，UTF-8是最常用的一种多字节编码方案，因为它具有良好的兼容性、节省空间和易于传输的特点。UTF-8 使用1到4个字节来表示前2^21（大约 2M）个码点。大部分常用字符采用1个字节表示，而较少使用的字符采用2至4个字节表示，实现了字符编码的高效使用和存储。

任何码点低于127的字符，即7位安全ASCII字符，都由与大多数其他单字节编码相同的1字节序列表示。任何码点高于127的字符都由两个或更多字节的序列表示。

**中文字符 UTF8 通常会占用3个字节**。这个范围涵盖了常用的中文字符、汉字和部分符号，包括了基本的中文字符集、部首和常用的汉字。

注意，这个范围并不包含所有的中文字符和汉字，只包括了常用的部分。如果需要表示更多的中文字符，可能需要考虑使用Unicode的其他范围或者使用Unicode的扩展字符集。

UTF-8的编码规则很简单，只有二条：

1）对于单字节的符号，字节的第一位设为0，后面7位为这个符号的unicode码。因此对于英语字母，UTF-8编码和ASCII码是相同的。

2）对于n字节的符号（n>1），第一个字节的前n位都设为1，第n+1位设为0，后面字节的前两位一律设为10。剩下的没有提及的二进制位，全部为这个符号的unicode码。

下表总结了编码规则，字母x表示可用编码的位。

**Unicode符号范围** | UTF-8编码方式 
(十六进制) | （二进制） 
————-----------——–+——————————————— 
0000 0000-0000 007F | 0xxxxxxx 
0000 0080-0000 07FF | 110xxxxx 10xxxxxx 
0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx 
0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

跟据上表，解读UTF-8编码非常简单。如果一个字节的第一位是0，则这个字节单独就是一个字符；如果第一位是1，则连续有多少个1，就表示当前字符占用多少个字节。

下面，还是以汉字”严”为例，演示如何实现UTF-8编码。

已知”严”的 unicode 是 **4E25**（100,1110,0010,0101），根据上表，可以发现4E25处在第三行的范围内（0000 0800-0000 FFFF），因此”严”的UTF-8编码需要三个字节，即格式是”1110xxxx 10xxxxxx 10xxxxxx”。然后，从”严”的最后一个二进制位开始，依次从后向前填入格式中的x，多出的位补0。这样就得到了，”严”的UTF-8编码是”1110**0100**,10**111000**,10**100101**”，转换成十六进制就是 **E4B8A5**, or b'\xe4\xb8\xa5 in python.


#### UTF16 (中文 variable 2-byte or 4-byte)

中文在UTF-16编码中的范围通常从U+4E00到U+9FFF，这是指汉字的基本区块。在UTF-16编码中，每个字符通常使用2个字节来表示，但对于一些罕见的字符，可能需要4个字节。因此，中文字符在UTF-16编码下通常占用2个字节。

#### UTF16 (中文 fixed 4-byte)

中文在UTF-32编码中的范围是从U+4E00到U+9FFF，与UTF-16相同。UTF-32编码是一种固定长度编码，每个字符使用4个字节来表示，不像UTF-8或UTF-16那样具有可变长度。因此，中文字符在UTF-32编码下也占用4个字节。

- UTF-8: Variable-width encoding, backwards compatible with ASCII. ASCII characters (U+0000 to U+007F) take 1 byte, code points U+0080 to U+07FF take 2 bytes, code points U+0800 to U+FFFF take 3 bytes, code points U+10000 to U+10FFFF take 4 bytes. **Good for English text, not so good for Asian text.**
- UTF-16: Variable-width encoding. Code points U+0000 to U+FFFF take 2 bytes, code points U+10000 to U+10FFFF take 4 bytes. **Bad for English text, good for Asian text.**
- UTF-32: Fixed-width encoding. All code points take four bytes. An enormous memory hog, but fast to operate on. **Rarely used.**



### GB18030 与 Unicode

GB18030 和 Unicode 相当于两套单独的编码体系，它们都对世界上大部分字符进行编码，赋予每个字符一个唯一的编号，只不过对于同一个字符，GB18030 和 Unicode 对应的编号是不一样的， 比如：汉字 "中" 字的 GB18030 编码是 0xD6D0, 对应的 Unicode 码元是 0x4E2D， 从这一点上可以认为 GB18030 是一种 Unicode 的转换格式

注意：要表达 Unicode 的编码格式才真正算得上 Unicode 转换格式，所以严格意义上说 GB18030 并不是真正的 Unicode 转换格式

**GB18030 既是字符集又是编码格式**，也即字符在字符集中的编号以及存储是进行编码用的编号是完全相同的，**而 Unicode 仅仅是字符集，它只规定了字符的唯一编号，它的存储是用其他的编码格式的，比如 UTF8、UTF16 等等**

既然 GB18030 和 Unicode 都能表示世界上大部分字符，为什么要弄两套字符集呢，一套的话不更有利于信息的传播吗？

1、在 Unicode 出现之前，没有统一的字符编码，每个操作系统上都有自己的一套编码标准，像早期的 window 上需要安装字符集，才能支持中文，这里的字符集就是微软自定的标准，换个其他系统就会失效

2、对于大部分中文字符来说，采用 GB18030 编码的话，只需两个字节，如果采用 UTF8 编码，就需要三个字节， 所以用 GB18030 存储和传输更节省空间

<img src="/media/image-20240428175437501.png" alt="image-20240428175437501" style="zoom:80%;" />



## ISO10646 和 Unicode

ISO/IEC 10646和Unicode是涉及数字系统中字符编码和表示的相关标准，但它们并不相同。以下是ISO/IEC 10646和Unicode之间的主要区别：

1. **ISO/IEC 10646**：
   - ISO/IEC 10646是由国际标准化组织（ISO）和国际电工委员会（IEC）制定的国际标准。
   - 它的目标是定义一个通用字符集，涵盖全球使用的所有主要文字、现代和历史文字、符号和特殊字符。
   - ISO/IEC 10646定义了通用编码字符集（UCS），其中包括来自各种语言和文字的字符的代码点。
   - 该标准使用固定宽度的编码，每个字符被分配一个唯一的代码点，通常以十六进制格式表示（例如，拉丁大写字母'A'的代码点为U+0041）。
   - ISO/IEC 10646旨在成为一个全面的标准，涵盖全球通信和数据交换所需的所有字符。
2. **Unicode**：
   - Unicode是计算机行业的标准，也定义了通用字符编码方案。
   - Unicode标准由Unicode联盟（Unicode Consortium）维护，该联盟是一个非营利组织，包括主要技术公司、学者和语言学专家。
   - Unicode最初设计为与ISO/IEC 10646兼容。事实上，多年来Unicode和ISO/IEC 10646一直保持同步，确保字符集和编码方法一致。
   - 虽然ISO/IEC 10646主要关注定义字符集和代码点，但Unicode还包括附加规范，如编码形式（UTF-8、UTF-16、UTF-32）、规范化规则、双向文本处理等。
   - Unicode还提供了广泛的文档、字符属性和文本处理以及不同语言和文字脚本的渲染算法。

总之，**ISO/IEC 10646定义了通用字符编码系统的字符集和代码点，而Unicode则涵盖了与字符编码**、文本处理和国际化相关的更广泛的规范和指南。然而，在实际应用中，Unicode和ISO/IEC 10646经常可以互换使用，因为它们高度兼容并且同步。

在1993年的版本中，BMP由A、I、O、R四大部份所組成。但經過增補修訂，已改為A、I、O、S、R五區，S區主要是用作UTF-16使用：

|      | 內　容                   | 碼　位       | 可編碼數 |
| ---- | ------------------------ | ------------ | -------- |
| Ａ區 | 拼音字母、符號和其它符號 | 0000 ～ 4DFF | 19,903   |
| Ｉ區 | 中日韓漢字區             | 4E00 ～ 9FFF | 20,992   |
| Ｏ區 | 保留未來使用             | A000 ～ D7FF | 14,336   |
| Ｓ區 | UTF-16使用區             | D800 ～ DFFF | 2,048    |
| Ｒ區 | 專用區（用戶╱業者）      | E000 ～ FFFD | 8,192    |



<img src="/media/image-20240428175219265.png" alt="image-20240428175219265" style="zoom:80%;" />

雖然有如上的分區原則，但因編碼時的碼位及區塊的整體考慮，使得BMP現況　與上述分區有所差異。典型的例子如1998年剛擴編完成的「中日韓認同表意文字擴充A」字集6,582字，因為I區僅餘零星位置，因此就編入A區3400~4DFF的位置，而該區原為韓文拼音(Hangul)符號，但因後來韓文拼音移置O區，空出的碼位就給了「中日韓認同表意文字擴充A」，造成實際編碼與理論分區有別的現象。 在I區的中日韓漢字部份，最當初進行編碼時，因各國漢字型體不盡相同，必須先進行認同(unify)整理工作，SC2/WG2因此邀集有關各國指派專家組成CJK/JRG(中日韓聯合研究工作組，即IRG前身)，進行字集的總整理。CJK/JRG歷經五次會議完成此項艱鉅工作，所整理的「中日韓認同表意文字」(CJK Unified Ideographs)參考了我國75年版CNS 11643之第1、2、14字面(T欄)，大陸的GB 2312、GB 12345、GB 7589、GB 17590、GB 8565(G欄)，日本的JIS X 0208、JIS X 0212(J欄)及南韓的KS C 5601、KSC 5667(K欄)等標準字符集，可說已包含這四地所常用的字。其字序主要是參考康熙字典、大漢和詞典、漢語大詞典及大字源字典，以先部首後筆劃的順序排列。CJK/JRG將此結果送交SC2/WG2編碼，完成了ISO 10646:1993 之BMP中I區的表意文字編碼標準，總計含20,902個漢字。其中包含了CNS 11643用字共有17,011個字。 



## ISO, Window CP, Shift-JIS

附帶是拉丁字母的編碼。如果編碼是用中文，但是解碼誤用或是裝置只支持拉丁字母解碼，也常會造成亂碼。所以我們也看拉丁字母編碼。

#### ISO-8859 (拉丁, fixed 1-byte)

ISO-8859是一系列用于表示范围在127到255之间的字母表的单字节编码方案。这些不同的字母表以ISO-8859-n的格式定义为“部分”，其中最熟悉的可能是ISO-8859-1，也称为'Latin-1'。与UTF-8一样，7位安全ASCII字符不受所使用的编码族的影响。

这种编码方案的缺点是其无法容纳由128个以上符号组成的语言，也无法同时安全显示多个符号族。此外，**随着UTF的崛起，ISO-8859编码已经不受欢迎。负责ISO-8859的ISO“工作组”于2004年解散**。

#### Windows Code Page 1252 (拉丁, fixed 1-byte) 

微软还维护了一组与ISO-8859有限兼容性的字符编码，通常表示为“cp####”。微软似乎推动将他们最新的产品发布转向使用一种形式的Unicode，但由于遗留或互操作性原因，您仍然可能会遇到它们。

例如，Windows CP1252 是 ISO-8859-1 的超集，包含0x80-0x9F范围内的其他可打印字符，尤其是欧元符号€和备受诟病的“智能引号”“”。这经常导致不匹配，即8859-1可以在1252上完美显示，而1252可能看起来在8859-1上显示良好，但当其中一个额外符号出现时会出现问题。

#### Shift-JIS, Windows Code Page 932 (日語、漢字, fixed 2-byte) 

**Shift_JIS**是[日本](https://zh.wikipedia.org/wiki/日本)電腦系統常用的編碼表，能容納[全形](https://zh.wikipedia.org/wiki/全形)及[半形](https://zh.wikipedia.org/wiki/半形)[拉丁字母](https://zh.wikipedia.org/wiki/拉丁字母)、[平假名](https://zh.wikipedia.org/wiki/平假名)、[片假名](https://zh.wikipedia.org/wiki/片假名)、[符號](https://zh.wikipedia.org/wiki/符号)及[日語](https://zh.wikipedia.org/wiki/日语)[漢字](https://zh.wikipedia.org/wiki/日本汉字)。命名為Shift_JIS的原因，是在放置全形字元時，要避開原本在0xA1至0xDF放置的[半形假名](https://zh.wikipedia.org/wiki/半角假名)字元。[微軟](https://zh.wikipedia.org/wiki/微软)及[IBM](https://zh.wikipedia.org/wiki/IBM)的日語電腦系統即使用了這編碼表，稱為**CP932**。





## 亂碼如何產生的呢？

亂碼產生的原因主要有兩個，**1. 文本字符編碼過程與解碼過程使用了不同的編碼方式**，**2. 使用了缺少某種字體庫的字符集引起的亂碼**。

### 1. 編碼與解碼使用了不同的編碼方式

亂碼就是 encode 和 decode 采用不同的編碼和解碼格式，包含：

* UTF encode/decode 的不一致:  例如 UTF8, UTF16, UTF32 的不一致
* GBK,  BIG5, UTF 之間 encode/decode 的不一致

**GBK——>UTF-8："鎷涜仒"**

這三字一看就是亂碼，我們使用GBK編碼與UTF-8解碼來修復。

GBK編碼映射結果為：

| 鎷   | E68B |
| ---- | ---- |
| 涜   | 9BE8 |
| 仒   | 8198 |

是按照：E68B 9BE8 8198 四個一組來做拆分

通過上面介紹可知，中文字符在UTF-8中是通過三字節進行編碼的，我們將三個字的GBK編碼按照每三字節為一組重組可得：

E68B9B

E88198

這兩個字符代表什麼呢？查詢UTF-8編碼表：

![img](https://i1.kknews.cc/PcjHry8mY_KI1q2_tElnbXKrSl8giCBXnAWhaJw/0.jpg)

```python
text = '招聘'
u1 = text.encode('utf-8')
print(u1)
print(u1.decode('gbk'))

b'\xe6\x8b\x9b\xe8\x81\x98'
鎷涜仒
```



例子中，用了utf-8編碼，使用了GBK解碼，結果產生了亂碼。因為在utf-8中，一個漢字用三個字節編碼，而GBK中，每個漢字用兩個字節表示，所以產生了亂碼。



### 2. 使用了缺少某種字體庫的字符集

我們知道GB2312是不支持繁體字的，所以使用缺少某種字體庫的字符集編碼，會產生亂碼。



## 亂碼又如何解決呢

使用**支持要展示字體的字符集**編碼，並且**編解碼使用同一種編碼方式**，就可以解決亂碼問題了。

接下來列舉一下亂碼的經典場景與解決方案

### IntelliJ Idea亂碼問題

IDE項目中的中文亂碼問題？File->settings->Editor->File Encodings,設置一下編碼方式utf-8





## 實例

### 1. 英文字母

```python
text = 'a'
print(text.encode('utf-8'))
print(text.encode('utf-16'))
print(text.encode('utf-32'))
print(text.encode('big5'))
print(text.encode('gbk'))
print(text.encode('gb2312'))

b'a'
b'\xff\xfe a\x00'  => why?
b'\xff\xfe\x00\x00 a\x00\x00\x00' => why?
b'a'
b'a'
b'a'
```

- '\xff\xfe' is the BOM (Byte-Order-Mark) for UTF-16 Little Endian, indicating the byte order.

- 'a\x00' could represent the character 'a' followed by a null byte, but it's important to note that UTF-16 encodes characters in two bytes (unless it's a supplementary character).

- '\xff\xfe\x00\x00' is the Byte Order Mark (BOM) for UTF-32 Little Endian, indicating the byte order.

- 'a\x00\x00\x00' represents the character 'a' followed by three null bytes, as UTF-32 uses 4 bytes per character.

  

### 2. 繁體中文

```python
text = '陸'
print(text.encode('utf-8'))
print(text.encode('utf-16'))
print(text.encode('utf-32'))
print(text.encode('big5'))
print(text.encode('gbk'))
print(text.encode('gb2312'))  => Error 因為無法識別繁體字

b'\xe9\x99\xb8'
b'\xff\xfe x\x96'
b'\xff\xfe\x00\x00 x\x96\x00\x00'
b'\xb3\xb0'
b'\xea\x91'
---------------------------------------------------------------------------
UnicodeEncodeError                        Traceback (most recent call last)
----> 7 print(text.encode('gb2312'))
UnicodeEncodeError: 'gb2312' codec can't encode character '\u9678' in position 0: illegal multibyte sequence
```



### 3. 簡體中文

```python
text = '陆'
print(text.encode('utf-8'))
print(text.encode('utf-16'))
print(text.encode('utf-32'))
print(text.encode('gbk'))
print(text.encode('gb2312'))
print(text.encode('big5'))  => Error 因為無法識別簡體字

b'\xe9\x99\x86'
b'\xff\xfe F\x96'
b'\xff\xfe\x00\x00 F\x96\x00\x00'
b'\xc2\xbd'
b'\xc2\xbd'
---------------------------------------------------------------------------
UnicodeEncodeError                        Traceback (most recent call last)
----> 7 print(text.encode('big5'))
UnicodeEncodeError: 'big5' codec can't encode character '\u9646' in position 0: illegal multibyte sequence
```

```python
text = '严'
print(text.encode('utf-8'))
print(text.encode('utf-16'))
print(text.encode('utf-32'))
print(text.encode('gbk'))
print(text.encode('gb2312'))
#print(text.encode('big5'))

b'\xe4\xb8\xa5'
b'\xff\xfe %N'
b'\xff\xfe\x00\x00 %N\x00\x00'
b'\xd1\xcf'
b'\xd1\xcf'
```





## Neural Network Coding Conversion

Constraint Neural Network!

因爲 < 7F  output code 就是 input code!!!!

可是所有其他的 code 都需要計算！







## Appendix

### 
