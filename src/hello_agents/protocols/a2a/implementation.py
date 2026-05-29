"""
基于官方a2a-sdk库的A2A协议实现

使用官方a2a-sdk库实现Agent-to-Agent Protocol功能。
官方仓库: https://github.com/a2aproject/a2a-python
安装: pip install a2a-sdk
"""
from typing import Dict, Any, List, Optional
import asyncio
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    from a2a.client import A2AClient
    from a2a.types import Message
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    A2AClient = None
    Message = None

class A2AServer:
    """A2A服务器(使用Flask提供HTTP API)"""
    def __init__(
            self,
            name: str,
            description: str,
            version: str = "1.0.0",
            capabilities: Optional[Dict[str, Any]] = None
    ):
        """
        初始化A2A服务器

        Args:
            name:Agent名称
            description:Agent描述
            version: Agent版本
            capabilities: Agent能力描述
        """
        self.name = name
        self.description = description
        self.version = version
        self.capabilities = capabilities or {}
        self.skills: Dict[str, Dict[str, Any]] = {}

    def add_skill(self, skill_name: str, func, description: Optional[str] = None):
        """添加技能到服务器

        Args:
            skill_name: 技能名称
            func: 技能函数
            description: 可选的技能描述（用于LLM路由器更精准地选择技能）
        """
        self.skills[skill_name] = {
            "func": func,
            "description": description or (func.__doc__ or "").strip() or skill_name
        }
        return func

    def skill(self, skill_name: str, description: Optional[str] = None):
        """装饰器方式添加技能"""
        def decorator(func):
            self.add_skill(skill_name, func, description=description)
            return func
        return decorator
    
    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """运行服务器(使用Flask提供HTTP API)"""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError(
                "A2A server requires Flask.Install it with:pip install flask"
            )

        app = Flask(self.name)

        # 禁用Flask的日志输出(可选)
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/info',methods=['GET'])
        def get_info():
            """获取Agent信息"""
            return jsonify(self.get_info())

        @app.route('/skills', methods=['GET'])
        def list_skills():
            """列出所有技能"""
            skill_list = []
            for name, info in self.skills.items():
                skill_list.append({
                    "name": name,
                    "description": info["description"]
                })
            return jsonify({
                "skills": skill_list,
                "count": len(self.skills)
            })

        @app.route('/execute/<skill_name>', methods = ['POST'])
        def execute_skill(skill_name):
            """执行指定技能"""
            if skill_name not in self.skills:
                return jsonify({
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": list(self.skills.keys())
                }),404

            try:
                data = request.get_json() or {}
                text = data.get('text', data.get('query',''))

                # 调用技能函数
                result = self.skills[skill_name]["func"](text)

                return jsonify({
                    "skill": skill_name,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "skill": skill_name,
                    "status": "error"
                }),500

        @app.route('/ask', methods = ['POST'])
        def ask():
            """通过 LLM 路由器自动、精准地选择并执行技能"""
            try:
                data = request.get_json() or {}
                question = data.get('question', data.get('text', ''))

                if not question:
                    return jsonify({"error": "Question cannot be empty", "status": "error"}), 400

                # 1. 构建可用技能清单（包含描述）
                available_skills_info = []
                for name, info in self.skills.items():
                    available_skills_info.append(f"- 技能名: {name}, 技能描述: {info['description']}")
                skills_manifest = "\n".join(available_skills_info)

                # 2. 调用 LLM 路由器进行技能选择
                try:
                    from core.my_llm import MyLLM

                    llm = MyLLM()

                    system_prompt = f"""你是一个高效的 AI Agent 路由网关。
你的任务是根据用户的输入（question），从下方可用的技能列表中，选择【最合适】的一个技能来处理。

【可用技能列表】：
{skills_manifest}

【严格输出要求】：
你只能输出最终选中的【技能名】，不要包含任何解释、不要标点符号、不要换行。
如果没有任何技能可以匹配，请严格输出: NONE"""

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ]

                    chosen_skill = llm.invoke(messages, temperature=0.0).strip()

                except Exception as llm_err:
                    return jsonify({
                        "error": f"LLM 路由器路由失败: {str(llm_err)}",
                        "status": "router_error"
                    }), 500

                # 3. 根据 LLM 决策进行分流执行
                if chosen_skill == "NONE" or chosen_skill not in self.skills:
                    return jsonify({
                        "answer": "抱歉，我的大模型路由器评估后发现，目前我拥有的技能无法解决您的问题。",
                        "status": "no_match"
                    })

                # 4. 精准调用选中的技能
                try:
                    selected_func = self.skills[chosen_skill]["func"]
                    result = selected_func(question)

                    return jsonify({
                        "answer": result,
                        "skill_used": chosen_skill,
                        "status": "success"
                    })
                except Exception as exec_err:
                    return jsonify({
                        "error": f"技能 [{chosen_skill}] 执行时内部报错: {str(exec_err)}",
                        "status": "execution_error"
                    }), 500

            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500

        @app.route('/health', methods=['GET'])
        def health():
            """健康检查"""
            return jsonify({"status": "healthy", "agent": self.name})

        # 启动服务器
        print(f"🚀 A2A 服务器 '{self.name}' 启动在 {host}:{port}")
        print(f"📋 描述: {self.description}")
        print(f"🛠️  可用技能: {list(self.skills.keys())}")
        print(f"📡 API 端点:")
        print(f"   - GET  {host}:{port}/info - 获取 Agent 信息")
        print(f"   - GET  {host}:{port}/skills - 列出技能")
        print(f"   - POST {host}:{port}/execute/<skill> - 执行技能")
        print(f"   - POST {host}:{port}/ask - 通用问答")
        print(f"   - GET  {host}:{port}/health - 健康检查")
        print()

        app.run(host=host, port=port, debug=False, threaded=True)

    def build_app(self):
        """构建 Flask app 对象（不启动），供外部嵌入到其他 WSGI 服务器"""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError(
                "A2A server requires Flask.Install it with:pip install flask"
            )

        app = Flask(self.name)

        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        @app.route('/info',methods=['GET'])
        def get_info():
            return jsonify(self.get_info())

        @app.route('/skills', methods=['GET'])
        def list_skills():
            skill_list = []
            for name, info in self.skills.items():
                skill_list.append({
                    "name": name,
                    "description": info["description"]
                })
            return jsonify({
                "skills": skill_list,
                "count": len(self.skills)
            })

        @app.route('/execute/<skill_name>', methods = ['POST'])
        def execute_skill(skill_name):
            if skill_name not in self.skills:
                return jsonify({
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": list(self.skills.keys())
                }),404
            try:
                data = request.get_json() or {}
                text = data.get('text', data.get('query',''))
                result = self.skills[skill_name]["func"](text)
                return jsonify({
                    "skill": skill_name,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "skill": skill_name,
                    "status": "error"
                }),500

        @app.route('/ask', methods = ['POST'])
        def ask():
            try:
                data = request.get_json() or {}
                question = data.get('question', data.get('text', ''))
                if not question:
                    return jsonify({"error": "Question cannot be empty", "status": "error"}), 400

                available_skills_info = []
                for name, info in self.skills.items():
                    available_skills_info.append(f"- 技能名: {name}, 技能描述: {info['description']}")
                skills_manifest = "\n".join(available_skills_info)

                try:
                    from core.my_llm import MyLLM
                    llm = MyLLM()
                    system_prompt = f"""你是一个高效的 AI Agent 路由网关。
你的任务是根据用户的输入（question），从下方可用的技能列表中，选择【最合适】的一个技能来处理。

【可用技能列表】：
{skills_manifest}

【严格输出要求】：
你只能输出最终选中的【技能名】，不要包含任何解释、不要标点符号、不要换行。
如果没有任何技能可以匹配，请严格输出: NONE"""
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ]
                    chosen_skill = llm.invoke(messages, temperature=0.0).strip()
                except Exception as llm_err:
                    return jsonify({
                        "error": f"LLM 路由器路由失败: {str(llm_err)}",
                        "status": "router_error"
                    }), 500

                if chosen_skill == "NONE" or chosen_skill not in self.skills:
                    return jsonify({
                        "answer": "抱歉，我的大模型路由器评估后发现，目前我拥有的技能无法解决您的问题。",
                        "status": "no_match"
                    })

                try:
                    selected_func = self.skills[chosen_skill]["func"]
                    result = selected_func(question)
                    return jsonify({
                        "answer": result,
                        "skill_used": chosen_skill,
                        "status": "success"
                    })
                except Exception as exec_err:
                    return jsonify({
                        "error": f"技能 [{chosen_skill}] 执行时内部报错: {str(exec_err)}",
                        "status": "execution_error"
                    }), 500
            except Exception as e:
                return jsonify({"error": str(e), "status": "error"}), 500

        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy", "agent": self.name})

        return app

    def get_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "protocol": "A2A",
            "skills": list(self.skills.keys())
        }

    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """获取指定技能的详细信息"""
        if skill_name not in self.skills:
            return None
        info = self.skills[skill_name]
        return {
            "name": skill_name,
            "description": info["description"]
        }

class A2AClient:
    """A2A 客户端（通过 HTTP 与 A2AServer 通信）"""

    def __init__(self, server_url: str):
        """
        初始化 A2A 客户端

        Args:
            server_url: 服务器 URL（例如：http://localhost:5000）
        """
        self.server_url = server_url.rstrip('/')

    def ask(self, question: str) -> str:
        """
        向 Agent 提问（通用接口）

        Args:
            question: 问题文本

        Returns:
            Agent 的回答
        """
        try:
            import requests
            response = requests.post(
                f"{self.server_url}/ask",
                json={"question": question},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("answer", "No response")
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"

    def execute_skill(self, skill_name: str, text: str = "") -> Dict[str, Any]:
        """
        执行指定技能

        Args:
            skill_name: 技能名称
            text: 输入文本

        Returns:
            执行结果
        """
        try:
            import requests
            response = requests.post(
                f"{self.server_url}/execute/{skill_name}",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to execute skill: {str(e)}", "status": "error"}

    def get_info(self) -> Dict[str, Any]:
        """获取 Agent 信息"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Failed to get agent info: {str(e)}"}

    def list_skills(self) -> List[str]:
        """列出 Agent 的技能"""
        try:
            import requests
            response = requests.get(f"{self.server_url}/skills", timeout=10)
            response.raise_for_status()
            return response.json().get("skills", [])
        except Exception as e:
            return []
