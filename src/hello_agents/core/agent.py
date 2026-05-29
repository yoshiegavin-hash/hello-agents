"""Agent基类"""
from abc import ABC, abstractclassmethod
from typing import Optional, Any
from .message import Message
from .my_llm import MyLLM
from .config import Config

class Agent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: MyLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None):
        self.name = name
        self.llm=llm
        self.system_prompt=system_prompt
        self.config=config
        self._history: list[Message] = []

    
    @abstractclassmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message:Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()
    

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"