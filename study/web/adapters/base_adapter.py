"""
基础适配器接口 - 定义所有适配器需要实现的接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Iterator
from langchain_core.messages import BaseMessage


class BaseAdapter(ABC):
    """基础适配器接口，所有 AI 功能适配器都应继承此类"""
    
    def __init__(self, name: str, description: str = ""):
        """
        初始化适配器
        
        Args:
            name: 适配器名称
            description: 适配器描述
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        初始化 AI 应用（如 LangGraph 图、模型等）
        
        Args:
            **kwargs: 初始化参数
        """
        pass
    
    @abstractmethod
    def process_message(
        self, 
        user_input: str, 
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        处理用户消息并返回响应
        
        Args:
            user_input: 用户输入的消息
            thread_id: 对话线程ID，用于管理多轮对话
            **kwargs: 其他参数
            
        Yields:
            字典，包含以下可能的键：
            - 'type': 'message' | 'tool' | 'end'
            - 'content': 消息内容或工具执行结果
            - 'messages': 完整的消息列表（仅在 type='end' 时）
        """
        pass
    
    @abstractmethod
    def regenerate(self, thread_id: Optional[str] = None, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        重新生成最后一条 AI 回复

        Args:
            thread_id: 对话线程ID
            **kwargs: 其他参数

        Yields:
            与 process_message 相同的字典结构
        """
        pass
    
    @abstractmethod
    def clear_history(self, thread_id: Optional[str] = None) -> bool:
        """
        清空对话历史
        
        Args:
            thread_id: 要清空的对话线程ID，如果为 None 则清空所有历史
            
        Returns:
            是否清空成功
        """
        pass
    
    @abstractmethod
    def get_history(self, thread_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            thread_id: 对话线程ID，如果为 None 则获取默认历史
            
        Returns:
            消息历史列表，每个元素包含 'role' 和 'content'
        """
        pass
    
    def format_message(self, message: BaseMessage) -> Dict[str, str]:
        """
        格式化 LangChain 消息为字典
        
        Args:
            message: LangChain 消息对象
            
        Returns:
            包含 'role' 和 'content' 的字典
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        
        if isinstance(message, HumanMessage):
            role = "user"
            content = message.content
        elif isinstance(message, AIMessage):
            role = "assistant"
            content = message.content
        elif isinstance(message, SystemMessage):
            role = "system"
            content = message.content
        elif isinstance(message, ToolMessage):
            role = "tool"
            content = f"[工具: {message.name}]\n{message.content}"
        else:
            role = "unknown"
            content = str(message.content) if hasattr(message, 'content') else str(message)
        
        return {
            "role": role,
            "content": content
        }

