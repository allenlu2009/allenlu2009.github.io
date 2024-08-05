---
title: Ollama Llama3
date: 2024-06-16 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
description: LLM Output Token Rate
typora-root-url: ../../allenlu2009.github.io


---





## Source

[Running Ollama on Google Colab (Free Tier): A Step-by-Step Guide | by Anoop Maurya | May, 2024 | Medium](https://medium.com/@mauryaanoop3/running-ollama-on-google-colab-free-tier-a-step-by-step-guide-9ef74b1f8f7a)





## Introduction





## Colab Ollama Llama3

Ref:   [Running Ollama on Google Colab (Free Tier): A Step-by-Step Guide | by Anoop Maurya | May, 2024 | Medium](https://medium.com/@mauryaanoop3/running-ollama-on-google-colab-free-tier-a-step-by-step-guide-9ef74b1f8f7a)



### Step 1: Installing package and loading the extension:

```
!pip install colab-xterm #https://pypi.org/project/colab-xterm/
%load_ext colabxterm
```



### Step 2: Opening Terminal

```
%xterm
```

在 terminal window

```
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & ollama pull llama3
```



有非常多的 models!

<img src="/media/image-20240616231117021.png" alt="image-20240616231117021" style="zoom:70%;" />



### 使用 Langchain

```python
!pip install langchain-community 

# Import Ollama module from Langchain
from langchain_community.llms import Ollama

# Initialize an instance of the Ollama model
llm = Ollama(model="llama2")

# Invoke the model to generate responses
response = llm.invoke("Tell me a joke")
print(response)
```



#### 使用 Ollama API

```python
from importlib.metadata import version

pkgs = ["tqdm",    # Progress bar
       ]

for p in pkgs:
    print(f"{p} version: {version(p)}")
    
    
import urllib.request
import json

def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "seed":123,        # for deterministic responses
        "temperature":0,   # for deterministic responses
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


result = query_model("What do Llamas eat?")
print(result)
```



















