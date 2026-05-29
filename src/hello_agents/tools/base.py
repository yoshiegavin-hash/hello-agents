from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel


class Tool:
    """工具参数定义"""
    def __init__(self, name: str, param_type: str, description: str, required: bool = True):
        self.name = name
        self.param_type = param_type
        self.description = description
        self.required = required


class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数是否符合工具定义"""
        params = self.get_parameters()
        if not params:
            return True
        required = {p.name for p in params if p.required}
        return required.issubset(set(parameters.keys()))


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None