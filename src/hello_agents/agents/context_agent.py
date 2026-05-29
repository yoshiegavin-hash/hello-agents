from typing import Optional, List
from core.agent import Agent
from core.my_llm import MyLLM
from core.config import Config
from core.message import Message
from tools.buildtin.memory_tool import MemoryTool
from tools.buildtin.rag_tool import RAGTool
from tools.buildtin.note_tool import NoteTool
from context.builder import ContextBuilder, ContextConfig, ContextPacket
from datetime import datetime


class ContextAgent(Agent):
    """基于 ContextBuilder 构建上下文的智能体

    与 SimpleAgent 的区别：
    - 使用 ContextBuilder 的 GSSC 流水线构建上下文
    - 自动从记忆系统和 RAG 中检索相关信息
    - 支持 token 预算管理和上下文压缩
    """

    def __init__(
            self,
            name: str,
            llm: MyLLM,
            memory_tool: Optional[MemoryTool] = None,
            rag_tool: Optional[RAGTool] = None,
            note_tool: Optional[NoteTool] = None,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            context_config: Optional[ContextConfig] = None,
            enable_context_builder: bool = True):
        super().__init__(name, llm, system_prompt, config)
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.note_tool = note_tool
        self.enable_context_builder = enable_context_builder

        self.context_builder = ContextBuilder(
            memory_tool=memory_tool,
            rag_tool=rag_tool,
            config=context_config or ContextConfig()
        )

        mode = "启用" if self.enable_context_builder else "禁用"
        print(f"✅ {name} 初始化完成，ContextBuilder: {mode}")

    def run(self, input_text: str, note_as_action: bool = False, **kwargs) -> str:
        """运行 ContextAgent

        如果启用了 ContextBuilder，则使用 GSSC 流水线构建上下文；
        否则回退到 SimpleAgent 的标准对话逻辑。

        Args:
            input_text: 用户输入
            note_as_action: 是否自动将本次交互保存为笔记
        """
        if not self.enable_context_builder:
            return self._run_simple(input_text, **kwargs)

        print(f"\U0001f916 {self.name} 使用 ContextBuilder 处理: {input_text}")

        # 从笔记系统检索并转换为 ContextPacket
        note_packets = self._build_note_packets(input_text)

        # 使用 ContextBuilder 构建结构化上下文
        context = self.context_builder.build(
            user_query=input_text,
            conversation_history=self.get_history(),
            system_instructions=self.system_prompt,
            additional_packets=note_packets
        )

        # 调用 LLM
        messages = [{"role": "user", "content": context}]
        response = self.llm.invoke(messages, **kwargs)

        # 保存对话历史
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

        # 自动保存为笔记
        if note_as_action and self.note_tool:
            self._save_as_note(input_text, response)

        return response

    def _run_simple(self, input_text: str, **kwargs) -> str:
        """回退到简单对话逻辑"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": input_text})

        response = self.llm.invoke(messages, **kwargs)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        return response

    def stream_run(self, input_text: str, note_as_action: bool = False, **kwargs):
        """流式运行"""
        if not self.enable_context_builder:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            for msg in self._history:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": input_text})

            full_response = ""
            for chunk in self.llm.stream_invoke(messages, **kwargs):
                full_response += chunk
                yield chunk

            self.add_message(Message(input_text, "user"))
            self.add_message(Message(full_response, "assistant"))
            return

        # 使用 ContextBuilder 的流式版本
        note_packets = self._build_note_packets(input_text)
        context = self.context_builder.build(
            user_query=input_text,
            conversation_history=self.get_history(),
            system_instructions=self.system_prompt,
            additional_packets=note_packets
        )

        messages = [{"role": "user", "content": context}]
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))

        # 自动保存为笔记
        if note_as_action and self.note_tool:
            self._save_as_note(input_text, full_response)

    def _build_note_packets(self, query: str, limit: int = 3) -> List[ContextPacket]:
        """检索笔记并转换为 ContextPacket

        从 NoteTool 检索相关笔记，封装为 ContextPacket 后
        通过 additional_packets 注入 ContextBuilder。
        """
        if not self.note_tool:
            return []

        packets = []
        try:
            # 搜索匹配的笔记
            search_results = self.note_tool.run({
                "action": "search",
                "query": query,
                "limit": limit
            })

            if search_results and "未找到" not in search_results:
                # 从解析结果中提取笔记ID列表
                import re

                note_lines = re.findall(r"ID:\s*(note_\S+)", search_results)
                for note_id in note_lines[:limit]:
                    note_path = self.note_tool._get_note_path(note_id)
                    if note_path.exists():
                        with open(note_path, 'r', encoding='utf-8') as f:
                            markdown_text = f.read()
                        note = self.note_tool._markdown_to_note(markdown_text)

                        content = f"[笔记:{note['title']}]\n{note['content']}"
                        packets.append(ContextPacket(
                            content=content,
                            timestamp=datetime.fromisoformat(note.get('updated_at', datetime.now().isoformat())),
                            relevance_score=0.75,
                            metadata={
                                "type": "notes",
                                "note_type": note.get('type', 'general'),
                                "note_id": note_id
                            }
                        ))

                if not note_lines:
                    return []
            else:
                return []

        except Exception as e:
            print(f"⚠️ 笔记检索失败: {e}")

        return packets

    def _save_as_note(self, user_input: str, response: str) -> str:
        """将本次交互自动保存为笔记

        根据内容关键词自动判断笔记类型：
        - blocker: 问题、阻塞
        - action: 计划、下一步
        - conclusion: 其他（关键结论）
        """
        if not self.note_tool:
            return ""

        # 根据关键词自动判断类型
        note_type = "conclusion"
        if any(kw in user_input for kw in ["问题", "阻塞", "失败", "报错", "bug"]):
            note_type = "blocker"
        elif any(kw in user_input for kw in ["计划", "下一步", "行动", "安排", "todo"]):
            note_type = "action"

        try:
            result = self.note_tool.run({
                "action": "create",
                "title": user_input[:30] + ("..." if len(user_input) > 30 else ""),
                "content": f"## 问题\n{user_input}\n\n## 分析\n{response}",
                "note_type": note_type,
                "tags": ["auto"]
            })
            print(f"[NoteTool] {result}")
            return result
        except Exception as e:
            print(f"⚠️ 自动保存笔记失败: {e}")
            return ""
