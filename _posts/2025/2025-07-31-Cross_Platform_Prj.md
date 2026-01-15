---
title: Cross Platform Python Project Management
date: 2025-07-31 23:10:08
typora-root-url: ../../allenlu2009.github.io
link: "[[2022-10-07-Python_Prj]]"
categories:
  - Tools
tags:
  - Claude
  - DevOps
  - Language
  - Python
  - Tools
  - VScode
---



老王 Good youtube to explain conda, uv difference
https://www.youtube.com/watch?v=jd1aRE5pJWc&ab_channel=%E7%A8%8B%E5%BA%8F%E5%91%98%E8%80%81%E7%8E%8B

very good follow-up by 老王
https://www.youtube.com/watch?v=-MSLJKjH8U0&ab_channel=%E7%A8%8B%E5%BA%8F%E5%91%98%E8%80%81%E7%8E%8B

[@liaoPythonImport2020] 指出 import 常常遇到的問題

[@loongProjectStructure2021]  Python project structure 寫的非常好，本文直接引用自己參考。

[GitHub - karpathy/minbpe: Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe)



## (2025/8/8) Conda 轉 Mamba takeaway 
from 5分钟彻底搞懂！Anaconda Miniconda conda-forge miniforge Mamba

1. 以後千萬不要 install anaconda, 只要 install miniforge!
2. 使用 miniforge 的 mamba 取代 anaconda 的 conda! 速度快十倍！
3. 對於已經 install anaconda 
	1. >  conda install mamba -n base -c conda-forge
	2. >  edit ~/.bashrc,  加上 `eval "$(mamba shell hook --shell bash)"`
	3. >  source ~/.bashrc,  以後把 conda command 改成 mamba command!
```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/allen/anaconda3_25/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/allen/anaconda3_25/etc/profile.d/conda.sh" ]; then
        . "/home/allen/anaconda3_25/etc/profile.d/conda.sh"
    else
        export PATH="/home/allen/anaconda3_25/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# >>> mamba initialize >>>
# Initialize mamba shell for bash
eval "$(mamba shell hook --shell bash)"
# <<< mamba initialize <<<
```

![[Pasted image 20250809162554.png]]

## Do > mamba install pip , then use > pip install!



## Modern Python Project

以下我們用 Karpathy 的 minbpe 爲例。
先從 github 下載 minbpe.  
用 tree 看結構。
用 pytest 確認是否 ok.

```
$ git clone https://github.com/karpathy/minbpe.git
$ tree
.
├── minbpe
│   ├── __init__.py
│   ├── base.py
│   ├── basic.py
│   ├── gpt4.py
│   └── regex.py
├── minbpe.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── pyproject.toml
├── requirements.txt
├── setup.py
├── tests
│   ├── __init__.py
│   ├── taylorswift.txt
│   └── test_tokenizer.py
└── train.py

$ pytest
collected 21 items
tests/test_tokenizer.py ........     [100%]
============= 21 passed in 22.04s ===
```

## 緣起 (2025/7/31)

Youtube video 解釋很好。第一個 project management 需求是： 
1. Virtual environment (project 隔離) + 
2. Package management (**萬惡根源！installation 以及不同版號之間的 dependency** )

## Virtual environment 三種選擇 : 用 `uv venv` or `conda create`

1. **Python 內建的在  local virtual environment .venv directory => 不好用 DO NOT use!
- 是 Python 3.3+ 內建的輕量 virtual environment 工具（官方標準庫提供）。  

2. **`uv venv`**  
- 一個結合傳統 local`.venv` 現代輕量 virtual environment 和 package management 工具。  
- 操作快速且用戶體驗好，功能類似於 pip + virtualenv + pyenv 的整合。 **=> 對於動態專案非常有用，例如搭配 Claude Code 很方便。**
- 容易建立和管理 virtual environments，與 Python 生態系統工具整合良好。  

3. **`Conda create`**  
- 一個支援多語言（Python、R、Julia等）且功能完整的環境及套件管理器。  
- 較 uv重且使用資源較多，**global environment (~/anaconda3) 不 tie to specific project。**  
- 非常適合資料科學專案、穩定且需要多語言或複雜依賴的專案。套件管理不只限 Python，還包含系統函式庫及執行檔。

| 特性             | `.venv` (builtin)      | `uv venv`                | `Conda create`        |
| -------------- | ---------------------- | ------------------------ | --------------------- |
| **範圍 (Scope)** | Python，local prj/.venv | Python，local prj/.venv   | 多語言，global ~/anaconda |
| **安裝複雜度**      | 簡單，內建工具                | 簡單且現代化                   | 較複雜且資源需求較高            |
| **套件管理**       | pip（獨立工具）              | 與 pip 整合                 | 整合 conda              |
| **效能**         | 輕量且基本                  | 快速且高效                    | 較慢且資源密集               |
| **適用場景**       | 簡單 isolated Python 專案  | 動態專案，快速設置                | 資料科學，穩定或多環境需求         |
| **跨版本支援**      | 較有限（單一 Python 版本）      | 支援多 Python 版本            | 支援多語言版本               |
| **進階功能**       | 較少                     | 結合 venv + pyenv + pip 功能 | 豐富，支援 binaries 及系統函式庫 |


## Package management

包含 package installation (安裝) 和 package list/dependency resolution (依賴解決)
### Package Installation 建議用 `uv pip install`

Installation 基本很單純：`pip install` or `uv pip install` or   `conda install`.   

pip 是 Python 官方推薦的標準套件安裝工具，適用於一般軟體開發。而 Conda 主要針對資料科學和多語言需求，並且能安裝一些系統層級依賴或非 Python 套件。

- **pip install 是最普遍的方式。**  
  大部分 Python package（尤其在 PyPI 上的）都能用 pip 安裝，且版本更新快。  
- **pip 通用於所有 virtual environment（.venv, uv, Conda env 等），而且大多數工具都是用 pip 作為安裝工具。**  
- **Conda install 只能用在 Conda 環境中，且只能安裝 Conda repository 中的 packages（也可以透過 conda-forge 獲取較多套件）。**  
- **此外，pip 支援從多種來源安裝（PyPI、Git repo、local package 等），彈性很高。**



**`uv pip install` 和傳統的 `pip install` 在功能上類似，都是用來安裝 Python 套件，但兩者的實現和行為有顯著差異：**

| 差異點              | `uv pip install`                                   | `pip install`                                |
|--------------------|--------------------------------------------------|---------------------------------------------|
| **性能與速度**       | 基於 Rust 實作，支持平行下載與安裝，速度快 10-100 倍 | 實現較老，單線程下載，速度較慢               |
| **虛擬環境管理**     | uv 內建虛擬環境管理，安裝時會自動識別並使用專案虛擬環境 | 需手動切換激活虛擬環境，pip 本身不管理環境    |
| **依賴解析**         | 使用優化過的依賴解決算法，錯誤訊息更清晰，衝突處理較好  | 解析能力較弱，衝突時錯誤不易理解             |
| **安裝行為**         | 會嚴格執行 `--only-binary` 等限制，對直接 URL 安裝更嚴格  | 對 URL 安裝較寬鬆，有時從源碼編譯             |
| **工具整合性**       | 與 uv 生態系統深度整合，支持虛擬環境、依賴鎖定、版本管理  | 只負責單純安裝套件，環境及鎖定需其他工具協助   |
| **錯誤處理與訊息**   | 清晰且具體，便於排錯                                | 較一般，錯誤提示有時不夠詳細                   |
| **安裝目標環境**     | 可針對指定虛擬環境安裝（非必須當前環境）               | 僅在當前激活的環境中安裝                      |

**重點說明：**

- `uv pip install` 實際上是 uv 對 pip 安裝功能的包裝與加強版，保留 pip 的用法與相容性，但加快速度並完善依賴解決和環境管理[1][2][6][5]。  
- `pip install` 是 Python 標準的套件安裝工具，廣受使用但功能專注且性能有限，需要搭配其他工具才能完善開發流程[1][5]。  
- 使用 `uv pip install` 在 uv 管理的專案中能更好地與虛擬環境和鎖定檔同步，適合追求效率和一致性的現代開發流程[1][6]。

如你主要在使用 uv 作為專案工具，建議改用 `uv pip install` 來享受快速且整合的體驗；但在傳統環境或無 uv 支援時，仍可用 `pip install`。


### 進階： Package Dependency 建議用 pyproject.toml + uv

Python 最令人詬病的地方是長期以來沒有一套官方完整且優良的 dependency management，造成很多包版本衝突及不可控問題。

**requirements.txt**:  
-  簡單易用以前最常用的方法，是過去最常見記錄依賴的格式。但它僅列出已安裝的 package 名稱與版本（通常手動維護），不描述 package 之間的依賴樹狀關係。 
- requirements.txt 是靜態列出安裝套件名及版本的檔案，無法描述完整依賴樹，卸載時無法自動清理依賴，且無法做依賴衝突解析；
- 適用作快速部署的“套件快照”，而非規劃完全版的依賴樹。  

**pyproject.toml**:  
- pyproject.toml 為 Python 官方新標準，可清楚聲明專案的直接依賴及建置工具版本。  
- 現代工具如 Poetry、PDM、Flit 等都使用 pyproject.toml，能有效管理依賴並支持依賴衝突解決，使環境乾淨且可重現。  
- 建議現代專案採用 pyproject.toml 結合專門依賴管理工具，以提升可維護性與穩定性。

### 建議說明
- **pip 自身並沒有內建依賴解決功能**（至少到 pip 22.x 版本前是如此，近期已有部分改進，但仍不及 Poetry 這類工具）。依賴解決常由外部管理工具（Poetry、uv、PDM、pip-tools）來輔助，而這些工具多是以 `pyproject.toml` 為標準依據。  
- **Conda 內建依賴解決，能安裝非 Python 套件和系統依賴，適合科學計算、跨語言專案。**  
- **pip + requirements.txt 仍然是大部分已有專案和 CI/CD 主要使用的方式，但逐漸被 pyproject.toml + Poetry/uv 類工具取代。**

## uv 建立並管理虛擬環境與 Python 專案 (含 pyproject.toml)

結合 **uv** 進行管理虛擬環境並建立 Python 專案（含 `pyproject.toml`）的完整流程：

1. **安裝 uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或在 macOS/Linux 用 brew 安裝
brew install uv
```

2. **建立新的 Python 專案（包含 pyproject.toml）**

```bash
uv init my_project
cd my_project
```

此時 uv 會建立以下基本專案結構（含 `pyproject.toml`）：

```
my_project/
├── .gitignore
├── .python-version
├── README.md
├── main.py
└── pyproject.toml
```

3. **建立虛擬環境**

```bash
uv venv
```

- 這會在專案目錄下建立 `.venv` 虛擬環境資料夾（名稱可自訂，預設是 `.venv`）。
- uv 也支援使用指定 Python 版本（如果沒裝，會自動下載）：
  
  ```bash
  uv venv --python 3.12.0
  ```

4. **安裝或新增套件（並自動更新 `pyproject.toml`）**

例如安裝 `requests`：

```bash
uv add requests
```

- 這會在虛擬環境中安裝 requests，同時將依賴加入 `pyproject.toml` 的 dependencies。

5. **執行 Python 腳本**

```bash
uv run main.py
```

- 會自動使用專案裡的虛擬環境執行。

6. **鎖定依賴版本**

```bash
uv lock
```

- 生成或更新 `uv.lock` 文件，確保版本一致。

7. **同步安裝所有依賴**

若其他人拉取專案，或剛從版本庫複製專案，可用：

```bash
uv sync
```

- 根據 `uv.lock` 與 `pyproject.toml` 安裝所有需依賴。

### 流程總結示意

```bash
# 安裝 uv（若需）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 新建專案
uv init my_project
cd my_project

# 建立虛擬環境（可指定 Python 版本）
uv venv --python 3.12.0

# 新增依賴
uv add requests flask

# 鎖定版本
uv lock

# 執行程序
uv run main.py

# 同步安裝依賴（他人複製代碼後用）
uv sync
```

### 補充說明

- **uv init** 會自動幫你生成 `pyproject.toml`，並載入一份簡單的專案結構。
- 虛擬環境 `.venv` 不會在 `uv init` 時自動建立，需要用 `uv venv` 指令生成。
- `uv add` 會同時安裝套件及更新 `pyproject.toml`，保持依賴同步管理。
- `uv run` 會自動找到並使用虛擬環境，免去手動 activate 的麻煩。
- `uv lock`和`uv sync`分別用於鎖版本和同步安裝，非常適用協作時保持環境一致性。


 ### **`uv add package`** 還是 **`uv pip install package`**?

- **`uv add package`**  
  - 是一個「高階的專案管理 API」。  
  - 它會將該套件自動加入你的專案的 `pyproject.toml` 中的依賴清單，同時安裝套件。  
  - 適合你有使用 `pyproject.toml` 來聲明專案依賴的情況。  
  - 這樣做有利於保持依賴的有序管理與版本鎖定，進而方便團隊同步。  
  - `uv add` 執行時會產生或更新 `uv.lock` 鎖定檔，確保環境跨機相容。  
  - 簡言之，是 uv 推薦的「專案依賴管理」方式。  

- **`uv pip install package`**  
  - 是一個「低階的 pip 兼容 API」，等同於傳統的 `pip install`。  
  - 僅僅在當前虛擬環境中安裝套件，不會改動 `pyproject.toml`。  
  - 適合臨時安裝或不使用 `pyproject.toml` 管理依賴的場景。  
  - 這種用法類似傳統直接操作環境的方式，缺少專案層級的依賴追蹤。  

### 簡單比喻：

| 功能          | uv add                         | uv pip install                 |
|---------------|--------------------------------|-------------------------------|
| 依賴管理       | 自動更新 pyproject.toml         | 不更新配置文件                  |
| 配合 pyproject.toml | 是（適合有專案依賴聲明）         | 否（手動操作當前環境）           |
| 適合用途       | 正規專案依賴管理與團隊協作       | 臨時、低階操作或快速安裝         |
| 產生鎖定檔     | 更新 uv.lock                    | 不產生鎖定檔                   |

- `uv add` 會保證跨平台（Windows、Linux 等）相容性，選擇版本時更謹慎，能確保其他人也可以無差異環境重現。  
- `uv pip install` 則通常安裝最新版本（無跨平台版本限制），有時可能帶來不一致版本風險。  
- `uv add` 是「專案 API」，而 `uv pip install` 是「低階 pip 命令集」的替代方案。  

**結論建議：**

- 如果你在用 `pyproject.toml` 管理專案，請優先使用 **`uv add package`**。這能讓專案依賴管理更規範且適合團隊合作。  
- 如果只是臨時在某環境裝包，或不想改動依賴聲明，可以用 **`uv pip install package`**。  

這樣能兼顧依賴管理的乾淨與裝包靈活度。



## uv 設定

`uv` stores packages is important for cross-platform development. Here's how `uv` works:

## Where uv stores packages:

### 1. **Virtual Environment (.venv directory)**
```bash
your-project/
├── .venv/           # Virtual environment (should NOT be synced)
├── pyproject.toml   # Project config (SHOULD be synced)
├── uv.lock         # Lock file (SHOULD be synced)
└── your-code/
```

### 2. **Global package cache** (shared across projects)
- **Linux/Mac**: `~/.cache/uv/`
- **Windows**: `%LOCALAPPDATA%\uv\cache\`

## Best practices for Google Drive sync:

### ✅ DO sync these files:
```bash
# Add to your project (sync these)
pyproject.toml      # Project dependencies and config
uv.lock            # Exact versions for reproducibility
requirements.txt    # If you're using this instead
```

### ❌ DON'T sync these:
```bash
# Add to .gitignore (don't sync these)
.venv/             # Virtual environment
__pycache__/       # Python cache
*.pyc              # Compiled Python files
.pytest_cache/     # Pytest cache
```

## Recommended setup for cross-platform:

### 1. Create a `.gitignore` file in your project:
```bash
# Create .gitignore to prevent syncing unwanted files
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
*.egg-info/
EOF
```

### 2. Use `pyproject.toml` for dependency management:
```toml
# pyproject.toml (this gets synced)
[project]
name = "claude_mt_bench"
dependencies = [
    "torch",
    "pytest",
    "accelerate",
    # other dependencies
]

[tool.uv]
dev-dependencies = [
    "pytest",
]
```

### 3. On each computer, recreate the environment:
```bash
# On any computer (PC/Mac), run:
cd ~/gdrive/github/claude_mt_bench
uv venv        # Creates local .venv (not synced)
uv sync        # Installs dependencies from pyproject.toml
```

## For your current situation:

Since you're already using Google Drive sync, here's what to do:

```bash
# 1. Make sure .venv is not being synced
echo ".venv/" >> .gitignore

# 2. If .venv already exists and is being synced, delete it
rm -rf .venv/

# 3. Create fresh environment on each machine
uv venv
uv sync  # or uv add your-dependencies

# 4. Run tests
uv run pytest tests/ -v
```

This way, each computer maintains its own `.venv` directory locally, but your project configuration is synced via `pyproject.toml`. The environments are isolated and won't conflict between different operating systems.



## Cross-platform 挑戰

1. Share 同一個 project with different PCs and Mac.  有些 PC 有 GPU with different CUDA 版本。 Mac 則需要 run CPU 版本。


如果你有兩台電腦共用同一個專案目錄，一台用 CPU 版本的 PyTorch，另一台用 CUDA 版本，要如何在 requirements.txt、pyproject.toml 或其他方式管理這種差異，核心問題是「依賴版本與來源有所不同怎麼條件安裝並共存」。

整理並參考最新的實務建議與工具支持，說明如下：

## 1. requirements.txt

- 傳統 `requirements.txt` 只是單純列出套件名和版本，**不支援根據硬體或環境條件切換版本**。
    
- 你可以寫兩份 `requirements-cpu.txt` 與 `requirements-gpu.txt`，兩台電腦各自用對應檔案安裝套件：
    
    text
    
    `# requirements-cpu.txt torch==2.0.1+cpu torchvision==0.15.2+cpu`
    
    text
    
    `# requirements-gpu.txt torch==2.0.1+cu118 torchvision==0.15.2+cu118`
    
- 安裝時指定檔案，例如：`pip install -r requirements-cpu.txt`。但只能靠手動選擇檔案，不會自動根據硬體條件切換。
    

## 2. pyproject.toml + Poetry 或 uv

現在多數現代 Python 專案用 `pyproject.toml` 配合管理工具（Poetry、uv、PDM）管理依賴，更靈活支持根據系統條件安裝版本。

## a. Poetry 支援「依賴群組」與「條件裝載」示例

text

`[tool.poetry.dependencies] python = "^3.8" [tool.poetry.group.cpu] optional = true [tool.poetry.group.cpu.dependencies] torch = { version = "^2.0.1", source = "pypi", markers = "sys_platform != 'linux' and sys_platform != 'win32'" } torchvision = { version = "^0.15.2", source = "pypi", markers = "sys_platform != 'linux' and sys_platform != 'win32'" } [tool.poetry.group.gpu] optional = true [tool.poetry.group.gpu.dependencies] torch = { version = "^2.0.1+cu118", source = "torch_cuda", markers = "sys_platform == 'linux' or sys_platform == 'win32'" } torchvision = { version = "^0.15.2+cu118", source = "torch_cuda", markers = "sys_platform == 'linux' or sys_platform == 'win32'" } [[tool.poetry.source]] name = "torch_cuda" url = "https://download.pytorch.org/whl/cu118" priority = "explicit"`

- 你可用命令選擇性安裝：
    
    bash
    
    `poetry install --with cpu poetry install --with gpu`
    
- 但實務上系統判斷和混合安裝稍複雜，需要人工選擇。
    

## b. uv 配置範例（根據系統條件切換索引源與版本）

`pyproject.toml` 範例（摘自 uv 文檔）：

text

`[project] name = "my_project" version = "0.1.0" requires-python = ">=3.12.0" dependencies = [   "torch>=2.7.0",  "torchvision>=0.22.0",  "pytorch-triton-rocm>=3.3.0 ; sys_platform == 'linux'", ] [tool.uv.sources] torch = [   { index = "pytorch-rocm", marker = "sys_platform == 'linux'" }, ] [[tool.uv.index]] name = "pytorch-rocm" url = "https://download.pytorch.org/whl/rocm6.3" explicit = true`

- 可針對各平臺分別設定不同索引和版本條件安裝。
    
- uv 能根據作業系統自動選擇對應版本，較適合跨平台 GPU/CPU 自動處理。
    

## 3. 用環境變數或安裝腳本動態決定依賴

- 實務中，有些團隊會撰寫安裝腳本判斷是否有 GPU，然後用不同指令安裝相應版本，避免在套件定義檔內複雜判斷。
    
- 例如 bash 下：
    
    bash
    
    `if nvidia-smi >/dev/null 2>&1; then   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html else   pip install torch==2.0.1+cpu torchvision==0.15.2+cpu fi`
    

## 總結建議

|方法|優點|缺點|適用情境|
|---|---|---|---|
|兩份 requirements.txt|簡單直接，技術門檻低|安裝時需手動選檔，不自動判斷GPU|小型專案或無專案管理工具時使用|
|pyproject.toml + Poetry|可設定依賴群組，條件化安裝，方便版本鎖定|設定稍複雜，安裝需明確指定群組|現代複雜專案，自動化和版本控管需求|
|pyproject.toml + uv|支援根據平臺條件切換索引與版本，自動化較好|仍在成長中，設定較新，需習慣工具|跨平台、高度自動化需求|
|安裝腳本判定與安裝|靈活，具體判斷GPU存在與否|需要額外腳本管理，不納入依賴管理工具|需要最大彈性或混合環境使用情境|

## 補充說明

- CPU 與 GPU 版本 PyTorch 通常是不同的 wheel 套件，需要從不同索引源（如 PyTorch 官方 CPU 或 CUDA wheel 源）安裝。這使得在同一依賴定義檔內自動判斷更不容易實作。
    
- `pyproject.toml` 目前屬於靜態依賴定義，對於跨硬體需求仍依賴工具的擴展（Poetry群組、uv索引條件）或外部判斷腳本配合。
    






### Method 1:  pip install -e .

如果沒有 pyproject.toml 或是 setupy.py, 必須先產生一個。
此處以 setup.py 爲例。

```python
from setuptools import setup, find_packages

setup(
    name='minbpe',
    version='0.1',
    packages=['minbpe'],  # Explicitly specify the package
    install_requires=[
        # List any dependencies here
    ],
)
```

接下來就執行下式，產生 minbpe 0.1 version.
```
$ pip install -e .
```



### 使用方法

```python
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

def test_encode_decode_identity(tokenizer, text):
    text = unpack(text)
    #tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded

def unpack(text):
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

tokenizer = GPT4Tokenizer()

test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]

for text in test_strings:
    test_encode_decode_identity(tokenizer, text)



```


## Reference