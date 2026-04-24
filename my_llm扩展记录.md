# 扩展 HelloAgentsLLM 支持多 Provider

## 背景

父类 `HelloAgentsLLM`（位于 `../Base-Agent/AgentLLM.py`）是一个 OpenAI 兼容的 LLM 客户端，通过设置不同的 `base_url` 可以对接不同的云服务。但它的签名是 `(model, apiKey, baseUrl, timeout)`，没有 `provider` 概念。

目标：通过 `MyLLM` 继承扩展，支持通过 `provider` 参数切换不同服务商的逻辑（Ollama 本地、ModelScope 云端等），并在凭证不全时自动降级。

## 遇到的问题与修复

### 1. 参数名不匹配导致父类调用失败

子类传的是 `api_key`、`base_url`（蛇形），父类接收的是 `apiKey`、`baseUrl`（驼峰）。修复了两处 `super().__init__()` 的参数名。

### 2. `self._client` vs `self.client` 变量名不一致

ModelScope 分支把客户端存在 `self._client`，但父类 `think()` 方法用的是 `self.client`。修复为统一使用 `self.client`。

### 3. 凭证缺失时没有 fallback

`provider="modelscope"` 但 `MODELSCOPE_API_KEY` 未设置时直接 `raise ValueError`。改为自动降级走父类逻辑。

### 4. Windows 上 httpx 与 Ollama 不兼容

OpenAI SDK 底层用 httpx，在 Windows 上请求 `localhost:11434` 时返回 502 错误。发现 `requests` 库可以正常工作，因此为 Ollama 写了独立的 `OllamaClient` 类，用 `requests` 替代 OpenAI SDK，并将非流式响应拆分为逐字符生成器模拟流式体验。

## 最终代码

### `my_llm.py`

```python
import os
import sys
import json
import requests
from typing import Optional, List, Dict, Iterator, Any
from openai import OpenAI

# 引入 Base-Agent 项目中的 HelloAgentsLLM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Base-Agent'))
from AgentLLM import HelloAgentsLLM  # type: ignore[import-untyped]


class _OllamaChunk:
    """模拟 OpenAI 流式响应的 chunk 对象"""
    def __init__(self, content: str):
        self.choices = [type('obj', (), {'delta': type('obj', (), {'content': content})()})()]


class OllamaClient:
    """用 requests 替代 OpenAI SDK，解决 Windows 上 httpx 与 Ollama 不兼容的问题"""
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    class ChatCompletions:
        def __init__(self, client):
            self._client = client

        def create(self, model: str, messages: List[Dict[str, str]], temperature: float = 0, stream: bool = True, **kw):
            url = f"{self._client.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,  # httpx 流式在 Windows 上有 bug，强制非流式
            }
            resp = requests.post(url, json=payload, timeout=self._client.timeout)
            resp.raise_for_status()
            data = resp.json()
            # 将非流式响应拆分为逐字符的生成器，模拟流式体验
            content = data["choices"][0]["message"]["content"]
            for char in content:
                yield _OllamaChunk(char)

    @property
    def chat(self):
        return type('obj', (), {'completions': self.ChatCompletions(self)})()


class MyLLM(HelloAgentsLLM):
    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str]= None,
            base_url: Optional[str] = None,
            provider: Optional[str] = "auto",
            **kwargs
    ):
        # 检查 provider 是否为 'ollama'
        if provider == "ollama":
            print("正在使用 Ollama Provider (本地模型)")
            self.provider = "ollama"
            self.base_url = base_url or "http://localhost:11434/v1"
            self.model = model or os.getenv("LLM_MODEL_ID") or "llama3.2:1b"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            self.client = OllamaClient(base_url=self.base_url, timeout=self.timeout)

        # 检查 provider 是否为 'modelscope'
        elif provider == "modelscope":
            api_key_resolved = api_key or os.getenv("MODELSCOPE_API_KEY")
            if api_key_resolved:
                print("正在使用自定义的 ModelScope Provider")
                self.provider = "modelscope"
                self.api_key = api_key_resolved
                self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"
                self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
                self.temperature = kwargs.get('temperature', 0.7)
                self.max_tokens = kwargs.get('max_tokens')
                self.timeout = kwargs.get('timeout', 60)
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            else:
                print("ModelScope API key 未设置，降级使用父类逻辑")
                super().__init__(
                    model=model,
                    apiKey=api_key,
                    baseUrl=base_url,
                    timeout=kwargs.get('timeout')
                )

        else:
            # 如果不是 modelscope, 则完全使用父类的原始逻辑来处理
            super().__init__(
                model=model,
                apiKey=api_key,
                baseUrl=base_url,
                timeout=kwargs.get('timeout')
            )
```

### `my_main.py`

```python
from my_llm import MyLLM

# 实例化我们重写的客户端，并指定 ollama provider
llm = MyLLM(
    provider="ollama",
    model="llama3.2:1b",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 准备消息
messages =[{"role": "user", "content": "你好，请介绍一下你自己。"}]

# 发起调用
response_stream = llm.think(messages)
# 打印响应
print("ModelScope Response:")
for chunk in response_stream:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    pass
```

## 架构说明

### 支持的 Provider

| provider 值 | 行为 | 依赖 |
|---|---|---|
| `ollama` | 本地 Ollama 服务（`requests` 调用） | `ollama serve` 运行中 |
| `modelscope` | 魔搭云 API（OpenAI SDK 调用） | `MODELSCOPE_API_KEY` 环境变量 |
| 其他（`auto`等） | 降级到父类，读 `.env` | `.env` 中的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID` |

### 降级链路

```
provider="modelscope"
  ├── MODELSCOPE_API_KEY 已设置 → ModelScope 云端
  └── MODELSCOPE_API_KEY 未设置 → 降级到父类（.env 配置的 DashScope）

provider="ollama"
  └── 直接走 OllamaClient（本地）

provider="auto" 或其他
  └── 父类 HelloAgentsLLM（.env 配置）
```

### 为什么 Ollama 不能用 OpenAI SDK？

Windows 上 OpenAI SDK 底层使用的 httpx 库在请求 `localhost:11434` 时返回 502 错误，但同地址用 `requests` 库或 `curl` 完全正常。这是已知的兼容性问题。`OllamaClient` 用 `requests` 绕过此问题，并将非流式响应拆解为逐字符生成器以兼容父类 `think()` 的流式消费逻辑。

## 术语解释

### 本地模型

将模型文件下载到本地硬盘，用你自己的 CPU/显卡运行。对比云 API（模型在别人的服务器上跑）：

| | 云 API | 本地模型 |
|---|---|---|
| 模型运行位置 | 云服务器 | 你自己的电脑 |
| 费用 | 按 token 收费 | 免费（消耗自己的电和硬件） |
| 延迟 | 受网络影响 | 本地，无网络延迟 |
| 隐私 | 请求数据发送到云端 | 数据不离开本机 |
| 硬件要求 | 无 | 需要足够的内存/显存 |

### Ollama 是什么

一个"本地模型一键运行工具"，做了三件事：

1. **下载模型** — `ollama pull llama3.2:1b` 把模型拉到本地
2. **启动服务** — 在本地 11434 端口开 HTTP 服务，自动加载模型
3. **提供 API** — 暴露 OpenAI 兼容的接口

类比：云 API 像去餐馆吃饭，Ollama 像自己在家做。

### "换个 URL 就行"的原理

所有 OpenAI 兼容的服务都遵循统一的请求/返回格式。所以父类 `HelloAgentsLLM` 只需要知道一个格式，`base_url` 换成谁就找谁：

| base_url | 实际服务方 |
|---|---|
| `https://dashscope.aliyuncs.com/compatible-mode/v1` | 阿里云百炼 |
| `https://api-inference.modelscope.cn/v1/` | 魔搭 |
| `http://localhost:11434/v1` | 本地 Ollama |

## 环境信息

- OS: Windows 10 Education
- GPU: AMD Radeon RX550/550 Series（~4GB，不支持 vLLM）
- Python: 3.13.9
- Ollama 模型: `llama3.2:1b`（1.3 GB）
