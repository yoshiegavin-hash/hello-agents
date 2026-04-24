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
