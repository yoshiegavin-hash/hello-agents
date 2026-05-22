from __future__ import annotations
from typing import Any, Callable, Dict
from .base import Tool

class ToolRegistry:
    """HelloAgents工具注册表"""
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}
    
    def register_tool(self, tool:Tool):
        """注册Tool对象"""
        if tool.name in self._tools:
            print(f"⚠ 警告：工具 '{tool.name}'已存在，将被覆盖")
        self._tools[tool.name] = tool
        print(f"工具'{tool.name}'已注册。")
    

    def registry_function(self, name: str, description: str, func:Callable[[str],str]):
        """
        直接注册函数作为工具(便捷方式)

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符结果
        """
        if name in self._functions:
            print(f"警告：工具'{name}'已存在，将被覆盖")
        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"工具 '{name}'已注册。")

    def get_tools_description(self) -> str:
        """获取所有可用工具的格式化描述字符串"""
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        # 函数工具描述
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "暂无可用的工具"

    def execute_tool(self, name: str, params: str) -> str:
        """执行指定工具，返回结果字符串"""
        tool = self._tools.get(name)
        if tool is not None:
            return tool.run({"expression": params})
        func_info = self._functions.get(name)
        if func_info is not None:
            return func_info["func"](params)
        return f"错误：未找到工具 '{name}'"
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换为 OPENAI function calling schema格式
        用于FunctionCallAgent，使工具能够被OpenAI原生function calling使用

        Returns:
            符合OpenAI function calling 标准的schema
        """
        parameters = self.get_parameters()

        # 构建properties
        properties = {}
        required = []

        for param in parameters:
            # 基础属性定义
            prop = {
                "type": param.type,
                "description": param.description
            }

            # 如果有默认值，添加到描述中(OpenAI schema 不支持default字段)
            if param.default is not None:
                prop["description"] = f"{param.description} (默认： {param.default})"

            # 如果是数组类型，添加items定义
            if param.type == "array":
                prop["items"] = {"type": "string"}
            
            properties[param.name] = prop

            # 手机必需参数
            if param.required:
                required.append(param.name)
        
        return{
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }