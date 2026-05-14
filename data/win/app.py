"""
“腺”而易见 —— MGD 智能辅助诊疗系统 V2.0
主入口文件
"""
import gradio as gr
from ui import create_app

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
