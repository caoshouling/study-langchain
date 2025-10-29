"""
基础 Gradio 应用框架 - 可复用的智能问答界面
"""
import gradio as gr
from typing import Optional, List, Dict, Any
from study.web.adapters.base_adapter import BaseAdapter


class ChatApp:
    """基于 Gradio 的聊天应用"""
    
    def __init__(self, adapter: BaseAdapter, title: str = "智能问答系统"):
        """
        初始化聊天应用
        
        Args:
            adapter: AI 功能适配器实例
            title: 应用标题
        """
        self.adapter = adapter
        self.title = title
        self.chat_history: List[Dict[str, str]] = []  # Gradio 需要的格式（messages类型）: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        self.current_thread_id: Optional[str] = None
        
        # 初始化适配器
        try:
            self.adapter.initialize()
            print(f"[ChatApp] {adapter.name} 适配器初始化成功")
        except Exception as e:
            print(f"[ChatApp] 适配器初始化失败: {e}")
            raise
    
    def chat_fn(self, message: str, history: List[Dict[str, str]]):
        """
        处理用户消息的核心函数（流式输出）
        
        Args:
            message: 用户输入的消息
            history: 当前对话历史（Gradio messages 格式）
            
        Yields:
            (更新的历史记录, 清空的输入框) - 逐步更新，实现流式输出
        """
        if not message or not message.strip():
            yield history, ""
            return
        
        # 添加用户消息到历史（使用新的消息格式）
        history.append({"role": "user", "content": message})
        
        # 先 yield 一次，显示用户消息并清空输入框
        yield history, ""
        
        # 初始化助手消息
        assistant_message_index = len(history)
        history.append({"role": "assistant", "content": ""})
        
        try:
            # 流式处理消息
            for result in self.adapter.process_message(
                user_input=message,
                thread_id=self.current_thread_id
            ):
                if result["type"] == "message":
                    content = result.get("content", "")
                    # 实时更新助手消息内容
                    if content:
                        history[assistant_message_index]["content"] = content
                        # yield 更新后的历史，实现流式输出
                        yield history, ""
                elif result["type"] == "end" and "messages" in result:
                    # 使用最终消息（如果有）
                    final_messages = result.get("messages", [])
                    if final_messages:
                        final_content = final_messages[-1].get("content", "")
                        if final_content:
                            history[assistant_message_index]["content"] = final_content
                            yield history, ""
            
            # 如果最终没有内容，添加默认消息
            if not history[assistant_message_index]["content"]:
                history[assistant_message_index]["content"] = "抱歉，没有收到回复。"
                yield history, ""
            
        except Exception as e:
            error_msg = f"处理消息时出错: {str(e)}"
            print(f"[ChatApp] {error_msg}")
            import traceback
            traceback.print_exc()
            # 更新错误消息到历史
            history[assistant_message_index]["content"] = error_msg
            yield history, ""
    
    def regenerate_fn(self, history: List[Dict[str, str]]):
        """
        重新生成最后一条 AI 回复
        
        Args:
            history: 当前对话历史
            
        Returns:
            (更新的历史记录, 输入框内容)
        """
        if not history or len(history) == 0:
            yield history, ""
            return
        
        # 找到最后一条 AI 消息和对应的用户消息
        last_assistant_idx = None
        last_user_idx = None
        
        # 从后往前找最后一条 assistant 消息
        for i in range(len(history) - 1, -1, -1):
            if history[i].get("role") == "assistant":
                last_assistant_idx = i
                # 找对应的用户消息（应该在前一条）
                if i > 0 and history[i-1].get("role") == "user":
                    last_user_idx = i - 1
                break
        
        if last_assistant_idx is None:
             yield history, "" # no assistant message to regenerate
             return
        
        # Prune the history to just before the last assistant message
        history = history[:last_assistant_idx]
        
        # Now stream from adapter.regenerate
        try:
            assistant_message_index = len(history)
            history.append({"role": "assistant", "content": ""})
            
            yield history, "" # Show the history with the empty assistant message bubble

            for result in self.adapter.regenerate(thread_id=self.current_thread_id):
                if result["type"] == "message":
                    content = result.get("content", "")
                    if content:
                        history[assistant_message_index]["content"] = content
                        # yield 更新后的历史，实现流式输出
                        yield history, ""
                elif result["type"] == "end" and "messages" in result:
                    final_messages = result.get("messages", [])
                    if final_messages:
                        final_content = final_messages[-1].get("content", "")
                        if final_content:
                            history[assistant_message_index]["content"] = final_content
                            yield history, ""
            
            # 如果最终没有内容，添加默认消息
            if not history[assistant_message_index]["content"]:
                history[assistant_message_index]["content"] = "抱歉，没有收到回复。"
                yield history, ""
            
        except Exception as e:
            error_msg = f"重新生成时出错: {str(e)}"
            print(f"[ChatApp] {error_msg}")
            import traceback
            traceback.print_exc()
            if assistant_message_index < len(history):
                history[assistant_message_index]["content"] = error_msg
            else:
                history.append({"role": "assistant", "content": error_msg})
            yield history, ""
    
    def clear_history_fn(self) -> tuple:
        """
        清空对话历史
        
        Returns:
            (清空的历史, 清空的输入框)
        """
        try:
            # 调用适配器清空历史
            self.adapter.clear_history(thread_id=self.current_thread_id)
            print(f"[ChatApp] 已清空历史")
        except Exception as e:
            print(f"[ChatApp] 清空历史失败: {e}")
        
        # 清空 Gradio 的历史
        self.chat_history = []
        return [], ""
    
    def build_interface(self) -> gr.Blocks:
        """
        构建 Gradio 界面
        
        Returns:
            Gradio Blocks 对象
        """
        with gr.Blocks(title=self.title, theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# {self.title}\n\n{self.adapter.description}")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=500,
                        show_label=True,
                        avatar_images=(None, None),  # 可以设置头像图片路径
                        container=True,
                        bubble_full_width=False,
                        type='messages'  # 使用新的消息格式
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="输入消息",
                            placeholder="请输入您的问题... (按 Enter 发送，Shift+Enter 换行)",
                            lines=2,
                            scale=4,
                            show_label=False,
                            container=True
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("清空历史", variant="secondary")
                        regenerate_btn = gr.Button("🔄 重新生成", variant="secondary", visible=True, elem_id="regenerate_btn")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 使用说明")
                    gr.Markdown(f"""
**功能**: {self.adapter.name}

**说明**: {self.adapter.description}

**操作**:
- 在输入框中输入您的问题
- 点击"发送"按钮或按 Enter 键发送消息
- 点击"清空历史"按钮清除所有对话记录
                    """)
            
            # 绑定事件
            # 点击发送按钮触发
            submit_btn.click(
                fn=self.chat_fn,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input],
                show_progress=False
            )
            
            # 按 Enter 键触发（多行输入框中，Enter 提交，Shift+Enter 换行）
            msg_input.submit(
                fn=self.chat_fn,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input],
                show_progress=False
            )
            
            # 清空历史按钮
            clear_btn.click(
                fn=self.clear_history_fn,
                inputs=[],
                outputs=[chatbot, msg_input],
                show_progress=False
            )
            
            # 重新生成按钮（支持流式输出）
            regenerate_btn.click(
                fn=self.regenerate_fn,
                inputs=[chatbot],
                outputs=[chatbot, msg_input],
                show_progress=False
            )
        
        return app
    
    def launch(self, server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False):
        """
        启动 Gradio 应用
        
        Args:
            server_name: 服务器地址
            server_port: 服务器端口
            share: 是否创建公开分享链接
        """
        app = self.build_interface()
        app.launch(
            server_name=server_name,
            server_port=server_port,
            share=share
        )

