from typing import List, Dict, Any, Optional
from registry import ToolRegistry

class ToolChain:
    """工具链 - 支持多个工具的顺序执行"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []
    
    def add_step(self, tool_name: str, input_template:str, output_key: str=None):
        """
        添加工具执行步骤
        Args:
            tool_name: 工具名称
            input_template: 输入模板，支持变量替换
            output_key: 输出结果的键名，用于后续步骤引用
        """
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        })

    
    def execute(self, registry: ToolRegistry, initial_input: str, context:Dict[str, Any] = None) -> str:
        """执行工具链"""
        context = context or {}
        context['input'] = initial_input

        print(f"🔗 开始执行工具链: {self.name}")

        for i, step in enumerate(self.steps, 1):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]

            # 替换模板中