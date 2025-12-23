
import gradio as gr
from gradio_unified3d import Unified3D


with gr.Blocks() as demo:
    Unified3D()


if __name__ == "__main__":
    demo.launch()
