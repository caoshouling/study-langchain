"""
主应用入口 - 启动智能问答 Web 界面
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from study.web.base_app import ChatApp
from study.web.adapters.policy_adapter import PolicyAdapter


def create_policy_app():
    """创建惠企政策问答应用"""
    adapter = PolicyAdapter()
    app = ChatApp(adapter, title="惠企政策智能问答系统")
    return app


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="启动智能问答 Web 界面")
    parser.add_argument(
        "--adapter",
        type=str,
        default="policy",
        choices=["policy"],
        help="选择要使用的适配器 (默认: policy)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务器地址 (默认: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务器端口 (默认: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建公开分享链接"
    )
    
    args = parser.parse_args()
    
    # 根据参数创建对应的应用
    if args.adapter == "policy":
        print("正在初始化惠企政策问答系统...")
        app = create_policy_app()
    else:
        print(f"未知的适配器: {args.adapter}")
        sys.exit(1)
    
    # 启动应用
    print(f"正在启动 Web 界面: http://{args.host}:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )

