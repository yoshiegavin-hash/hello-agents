from typing import Optional, Dict
from core.agent import Agent
from core.my_llm import MyLLM
from core.config import Config
from core.message import Message


# --- 记忆模块 ---
class _ReflectionMemory:
    """短期记忆模块，存储执行与反思的迭代轨迹"""
    def __init__(self):
        self.records: list[dict] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None


# --- 默认提示词模板 ---
_DEFAULT_INITIAL_PROMPT = """你是一位专业的内容创作者和分析师。请根据以下任务要求，生成高质量的内容。

任务要求: {task}

请直接输出你的创作内容，不要包含额外的解释或说明。
"""

_DEFAULT_REFLECT_PROMPT = """你是一位极其严格的评审专家。你的任务是审查以下内容，并专注于找出其中的问题、不足或可改进之处。

# 原始任务:
{task}

# 待审查内容:
{content}

请仔细分析内容质量，指出主要问题和改进建议。如果内容已经很好，无明显问题，请回答"无需改进"。

请直接输出你的反馈，不要包含额外的解释。
"""

_DEFAULT_REFINE_PROMPT = """你是一位专业的内容创作者。请根据评审专家的反馈，对上一版本内容进行优化。

# 原始任务:
{task}

# 上一版本内容:
{last_content}

# 评审员反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本内容。请直接输出优化后的内容，不要包含额外的解释。
"""


class MyReflectionAgent(Agent):
    """
    通用反思型智能体 - 适用于文本生成、分析、创作等多种场景

    工作流程:
    1. 初始生成 - 根据任务要求生成第一版内容
    2. 反思循环 - 评审 → 如果无需改进则停止 → 否则优化
    3. 输出最终结果

    支持通过 custom_prompts 参数深度定制三个阶段的提示词:
        custom_prompts = {
            "initial": "你的初始生成提示词模板，使用 {{task}} 占位",
            "reflect": "你的反思提示词模板，使用 {{task}}, {{content}} 占位",
            "refine": "你的优化提示词模板，使用 {{task}}, {{last_content}}, {{feedback}} 占位"
        }
    """

    def __init__(
            self,
            name: str,
            llm: MyLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            max_iterations: int = 3,
            custom_prompts: Optional[Dict[str, str]] = None,
            stop_keyword: str = "无需改进"
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory = _ReflectionMemory()
        self.stop_keyword = stop_keyword

        # 提示词模板：优先使用 custom_prompts，否则使用默认
        self.initial_prompt = custom_prompts.get("initial", _DEFAULT_INITIAL_PROMPT) if custom_prompts else _DEFAULT_INITIAL_PROMPT
        self.reflect_prompt = custom_prompts.get("reflect", _DEFAULT_REFLECT_PROMPT) if custom_prompts else _DEFAULT_REFLECT_PROMPT
        self.refine_prompt = custom_prompts.get("refine", _DEFAULT_REFINE_PROMPT) if custom_prompts else _DEFAULT_REFINE_PROMPT

        print(f"✅ {name} 初始化完成，最大迭代次数: {max_iterations}")

    def run(self, task: str, **kwargs) -> str:
        """
        运行反思型智能体

        Args:
            task: 任务描述，如"写一篇关于AI的短文"、"分析以下数据的趋势"等
            **kwargs: 传递给 LLM 的额外参数（如 temperature）

        Returns:
            最终生成的内容
        """
        print(f"\n🤖 {self.name} 开始处理任务: {task}")

        # 1. 初始生成
        print("\n--- 正在进行初始尝试 ---")
        prompt = self.initial_prompt.format(task=task)
        initial_content = self._llm_call(prompt, **kwargs)
        self.memory.add_record("execution", initial_content)

        # 2. 迭代循环：反思与优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i + 1}/{self.max_iterations} 轮迭代 ---")

            # 反思
            print("\n-> 正在进行反思...")
            last_content = self.memory.get_last_execution()
            prompt = self.reflect_prompt.format(task=task, content=last_content)
            feedback = self._llm_call(prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # 检查停止条件
            if self.stop_keyword in feedback:
                print("\n✅ 反思认为内容已无需改进，任务完成。")
                break

            # 优化
            print("\n-> 正在进行优化...")
            prompt = self.refine_prompt.format(
                task=task,
                last_content=last_content,
                feedback=feedback
            )
            refined_content = self._llm_call(prompt, **kwargs)
            self.memory.add_record("execution", refined_content)

        # 保存历史记录
        final_result = self.memory.get_last_execution()
        self.add_message(Message(f"任务: {task}", "user"))
        self.add_message(Message(final_result, "assistant"))

        print(f"\n--- 任务完成 ---")
        print(f"✅ {self.name} 响应完成")
        return final_result

    def stream_run(self, task: str, **kwargs):
        """
        流式运行（仅流式输出最终优化结果）
        """
        print(f"\n🌊 {self.name} 开始流式处理: {task}")

        # 非流式完成迭代过程
        final_result = self.run(task, **kwargs)

        # 流式输出最终结果
        for chunk in final_result:
            yield chunk

    def _llm_call(self, prompt: str, **kwargs) -> str:
        """调用 LLM 并返回完整响应"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""
