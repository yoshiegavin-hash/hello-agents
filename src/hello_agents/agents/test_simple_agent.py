import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from core.my_llm import MyLLM
from tools.registry import ToolRegistry
from tools.buildtin.calculator import MyCalculator
from simple_agent import MySimpleAgent

# 加载环境变量
load_dotenv()

# 创建LLM实例
llm = MyLLM()

# 测试1：基础对话Agent（无工具）
print("=====测试1：基础对话====")
basic_agent = MySimpleAgent(
    name="基础助手",
    llm=llm,
    system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
)

response1 = basic_agent.run("你好，请介绍一下自己")
print(f"基础对话响应: {response1}\n")

# 测试2：带工具的Agent
print("=== 测试2：工具增强对话 ===")
tool_registry = ToolRegistry()
calculator = MyCalculator()
tool_registry.register_tool(calculator)

enhanced_agent = MySimpleAgent(
    name= "增强助手",
    llm = llm,
    system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
    tool_registry=tool_registry,
    enable_tool_calling=True
)

response2 = enhanced_agent.run("请帮我计算 15*8+32")
print(f"工具增强响应：{response2}\n")

# 测试3：流式响应
print("=== 测试3:流式响应 ===")
print("流式响应： ", end="")
for chunk in basic_agent.stream_run("请解释什么是人工智能"):
    pass

