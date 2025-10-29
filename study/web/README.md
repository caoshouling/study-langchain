# Web 智能问答界面

这是一个基于 Gradio 的可复用智能问答 Web 界面框架，可以快速对接不同的 AI 功能。

## 功能特性

- ✅ 现代化的聊天界面，支持消息输入和历史记录
- ✅ 可清空对话历史
- ✅ 可复用的适配器架构，方便对接新的 AI 功能
- ✅ 基于 Gradio，界面美观易用

## 安装依赖

确保已安装以下依赖：

```bash
pip install gradio langchain langgraph langchain-openai
```

或者在项目根目录运行：

```bash
pip install -r requirements.txt
pip install gradio
```

## 项目结构

```
study/web/
├── __init__.py              # 包初始化文件
├── base_app.py              # 基础 Gradio 应用框架
├── app.py                   # 主应用入口
├── README.md                # 使用说明
└── adapters/                # 适配器目录
    ├── __init__.py
    ├── base_adapter.py      # 基础适配器接口
    └── policy_adapter.py    # 惠企政策问答适配器
```

## 快速开始

### 启动惠企政策问答系统

```bash
# 默认配置（127.0.0.1:7860）
python -m study.web.app

# 指定端口
python -m study.web.app --port 8080

# 创建公开分享链接
python -m study.web.app --share
```

### 使用方式

1. 启动应用后，浏览器会自动打开 Web 界面
2. 在输入框中输入您的问题
3. 点击"发送"按钮或按 Enter 键发送消息
4. 查看 AI 的回复
5. 点击"清空历史"按钮清除所有对话记录

## 如何对接新的 AI 功能

### 步骤 1: 创建适配器

在 `adapters/` 目录下创建新的适配器类，继承 `BaseAdapter`：

```python
from study.web.adapters.base_adapter import BaseAdapter
from typing import Dict, Any, Iterator, Optional, List

class MyCustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__(
            name="我的AI功能",
            description="功能描述"
        )
        # 初始化你的AI应用
        self.my_app = None
    
    def initialize(self, **kwargs) -> None:
        """初始化你的AI应用"""
        # 在这里初始化你的 LangGraph、模型等
        pass
    
    def process_message(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """处理用户消息"""
        # 调用你的AI应用，处理消息
        # yield 返回结果
        pass
    
    def clear_history(self, thread_id: Optional[str] = None) -> bool:
        """清空对话历史"""
        # 实现清空逻辑
        pass
    
    def get_history(self, thread_id: Optional[str] = None) -> List[Dict[str, str]]:
        """获取对话历史"""
        # 返回历史记录
        pass
```

### 步骤 2: 注册适配器

在 `app.py` 中添加你的适配器：

```python
from study.web.adapters.my_custom_adapter import MyCustomAdapter

def create_my_custom_app():
    adapter = MyCustomAdapter()
    app = ChatApp(adapter, title="我的AI功能")
    return app

# 在 main 中添加选择项
if args.adapter == "my_custom":
    app = create_my_custom_app()
```

### 步骤 3: 启动应用

```bash
python -m study.web.app --adapter my_custom
```

## 适配器接口说明

所有适配器必须实现 `BaseAdapter` 接口中的以下方法：

- `initialize(**kwargs)`: 初始化 AI 应用
- `process_message(user_input, thread_id, **kwargs)`: 处理用户消息，返回迭代器
- `clear_history(thread_id)`: 清空对话历史
- `get_history(thread_id)`: 获取对话历史

`process_message` 方法应该 yield 字典，包含以下键：
- `type`: `"message"` | `"tool"` | `"end"`
- `content`: 消息内容
- `messages`: 完整消息列表（`type="end"` 时可选）

## 注意事项

1. 确保环境变量已设置（如 `DASHSCOPE_API_KEY`）
2. 适配器的 `initialize` 方法会在创建 `ChatApp` 时自动调用
3. 对话历史通过 `thread_id` 管理，每次清空历史会生成新的 thread_id
4. 适配器应该处理异常，避免崩溃

## 示例：惠企政策问答

惠企政策问答适配器已经实现，展示了如何对接 LangGraph 应用：

- 使用动态导入加载原始的 `demo-惠企政策问答.py`
- 使用 `MemorySaver` 管理对话历史
- 流式处理消息并返回给用户

