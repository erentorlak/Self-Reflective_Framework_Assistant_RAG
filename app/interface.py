# app/interface.py
import gradio as gr
from app.inference import InferenceEngine

class Interface:
    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine

    def launch_interface(self):
        demo = gr.ChatInterface(
            self.inference_engine.inference,
            chatbot=gr.Chatbot(height=600,type="messages"), 
            textbox=gr.Textbox(placeholder="Ask a question", container=False, scale=7),
            title="Self Reflective Framework Assistant",
            description="Ask me anything about LangGraph.",
            type="messages"
        )
        demo.launch()
