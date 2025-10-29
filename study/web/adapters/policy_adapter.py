"""
惠企政策问答适配器 - 对接 demo-惠企政策问答.py
"""
import sys
import os
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path

# 添加项目根目录到路径，以便导入其他模块
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from study.web.adapters.base_adapter import BaseAdapter
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver


class PolicyAdapter(BaseAdapter):
    """惠企政策问答适配器"""
    
    def __init__(self):
        super().__init__(
            name="惠企政策问答",
            description="帮助企业判断是否符合政策申请资格的智能顾问"
        )
        self.app = None
        self.memory = None
        self.default_thread_id = "default-user-session"
        
    def initialize(self, **kwargs) -> None:
        """初始化惠企政策问答应用"""
        # 由于文件名包含中文，使用 importlib 动态导入
        import importlib.util
        import sys
        
        # 获取原文件路径
        policy_file = project_root / "study" / "langgraph" / "advance" / "demo-惠企政策问答.py"
        
        if not policy_file.exists():
            raise FileNotFoundError(f"找不到文件: {policy_file}")
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("policy_module", policy_file)
        policy_module = importlib.util.module_from_spec(spec)
        sys.modules["policy_module"] = policy_module
        spec.loader.exec_module(policy_module)
        
        # 获取 build_agent_app 函数
        build_agent_app = policy_module.build_agent_app
        
        # 初始化记忆存储
        self.memory = MemorySaver()
        
        # 构建并编译应用
        self.app = build_agent_app().compile(checkpointer=self.memory)
        
        print(f"[PolicyAdapter] 已初始化 {self.name}")
    
    def _stream_app_response(self, inputs: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Iterator[Dict[str, Any]]:
        """
        处理 app.stream 的通用流式逻辑
        """
        from langchain_core.messages import AIMessage, ToolMessage
        last_ai_message = None
        last_content = ""
        has_yielded = False
        
        try:
            for event in self.app.stream(inputs, config=config, stream_mode="values"):
                if isinstance(event, dict) and "messages" in event:
                    messages: List[BaseMessage] = event["messages"]
                    if messages:
                        last_message = messages[-1]
                        
                        if isinstance(last_message, AIMessage):
                            current_content = last_message.content if hasattr(last_message, 'content') and last_message.content else ""
                            if current_content and current_content != last_content:
                                last_content = current_content
                                last_ai_message = last_message
                                has_yielded = True
                                yield {
                                    "type": "message",
                                    "content": current_content
                                }
                        elif isinstance(last_message, ToolMessage):
                            if kwargs.get("show_tool_calls", True):
                                yield {
                                    "type": "tool",
                                    "content": f"[工具执行]: {last_message.name}\n{last_message.content}",
                                    "tool_name": last_message.name
                                }
        except Exception as e:
            print(f"[PolicyAdapter] 流式处理出错: {e}")
            import traceback
            traceback.print_exc()
        
        if not has_yielded:
            try:
                final_state = self.app.invoke(inputs, config=config)
                if isinstance(final_state, dict) and "messages" in final_state:
                    messages: List[BaseMessage] = final_state["messages"]
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            last_ai_message = msg
                            yield {
                                "type": "message",
                                "content": msg.content
                            }
                            break
            except Exception as e:
                print(f"[PolicyAdapter] 获取最终状态失败: {e}")
        
        if last_ai_message:
            yield {
                "type": "end",
                "messages": [self.format_message(last_ai_message)]
            }
            
    def process_message(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        处理用户消息
        
        Args:
            user_input: 用户输入
            thread_id: 对话线程ID
            
        Yields:
            处理结果字典
        """
        if self.app is None:
            raise RuntimeError("适配器未初始化，请先调用 initialize()")
        
        current_thread_id = thread_id or self.default_thread_id
        config = {"configurable": {"thread_id": current_thread_id}}
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        yield from self._stream_app_response(inputs, config, **kwargs)

    def regenerate(self, thread_id: Optional[str] = None, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        重新生成最后一条 AI 回复
        """
        if self.app is None:
            raise RuntimeError("适配器未初始化，请先调用 initialize()")

        current_thread_id = thread_id or self.default_thread_id
        config = {"configurable": {"thread_id": current_thread_id}}

        checkpoint = self.memory.get(config)
        if not checkpoint:
            yield {"type": "message", "content": "没有历史记录可以重新生成。"}
            return

        messages = checkpoint.get("channel_values", {}).get("messages", [])
        if not messages:
            yield {"type": "message", "content": "没有历史记录可以重新生成。"}
            return

        last_human_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_index = i
                break

        if last_human_index == -1:
            yield {"type": "message", "content": "找不到可以重新生成的用户消息。"}
            return

        # 截断历史记录到最后一个用户消息之前
        pruned_messages = messages[:last_human_index]

        # 使用 update_state 安全地更新后端记忆
        self.app.update_state(config, {"messages": pruned_messages})

        # 获取需要重新运行的用户消息
        message_to_rerun = messages[last_human_index]
        inputs = {"messages": [message_to_rerun]}
        
        # 使用更新后的状态，重新触发流式响应
        yield from self._stream_app_response(inputs, config, **kwargs)
    
    def clear_history(self, thread_id: Optional[str] = None) -> bool:
        """
        清空对话历史
        
        Args:
            thread_id: 要清空的线程ID，如果为 None 则使用默认线程ID
        """
        try:
            current_thread_id = thread_id or self.default_thread_id
            # 由于使用 MemorySaver，我们需要重新初始化或使用新的 thread_id
            # 简单做法：将 thread_id 改为新的，旧的会被遗忘
            import uuid
            self.default_thread_id = f"default-user-session-{uuid.uuid4().hex[:8]}"
            return True
        except Exception as e:
            print(f"[PolicyAdapter] 清空历史失败: {e}")
            return False
    
    def get_history(self, thread_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            thread_id: 对话线程ID
        """
        if self.app is None or self.memory is None:
            return []
        
        try:
            current_thread_id = thread_id or self.default_thread_id
            config = {"configurable": {"thread_id": current_thread_id}}
            
            # 从记忆中获取状态
            state = self.memory.get(config)
            if state and "messages" in state:
                messages = state["messages"]
                # 过滤掉系统消息
                formatted = []
                for msg in messages:
                    formatted_msg = self.format_message(msg)
                    # 不显示系统消息和工具消息给用户
                    if formatted_msg["role"] not in ["system", "tool"]:
                        formatted.append(formatted_msg)
                return formatted
            return []
        except Exception as e:
            print(f"[PolicyAdapter] 获取历史失败: {e}")
            return []

