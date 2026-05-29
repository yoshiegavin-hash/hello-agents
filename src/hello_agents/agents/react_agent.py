MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。
## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought：你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action：你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动：
"""
from typing import Optional, List
from base_agent.react import ReActAgent
from hello_agents.core.my_llm import MyLLM
from hello_agents.core.config import Config
from hello_agents.core.message import Message
from hello_agents.tools.registry import ToolRegistry


class _ToolExecutorAdapter:
    """适配 ToolRegistry 到 ReActAgent 需要的 ToolExecutor 接口"""
    def __init__(self, registry):
        self._registry = registry

    def getAvailableTools(self) -> str:
        return self._registry.get_tools_description()

    def getTool(self, name: str):
        tool = self._registry._tools.get(name)
        if tool is not None:
            return lambda params: tool.run({"expression": params} if hasattr(tool, 'get_parameters') else params)
        func_info = self._registry._functions.get(name)
        if func_info is not None:
            return func_info["func"]
        return None


class MyReActAgent(ReActAgent):
    """重写的ReAct Agent - 推理与行动结合的智能体"""
    def __init__(
            self,
            name: str,
            llm: MyLLM,
            tool_registry: ToolRegistry,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            max_steps: int = 5,
            custom_prompt: Optional[str] = None
    ):
        tool_executor = _ToolExecutorAdapter(tool_registry)
        super().__init__(llm, tool_executor, max_steps)
        self.name = name
        self.llm = llm
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.system_prompt = system_prompt
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行ReAct Agent"""
        self.current_history = []
        current_step = 0
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)

            thought, action = self._parse_output(response_text)

            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            if action:
                tool_name, tool_input = self._parse_action(action)
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")

        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

    def add_message(self, message):
        """添加消息到历史记录（兼容 Agent 基类接口）"""
        if not hasattr(self, '_history'):
            self._history = []
        self._history.append(message)

    def get_history(self):
        """获取历史记录（兼容 Agent 基类接口）"""
        if not hasattr(self, '_history'):
            return []
        return self._history.copy()
