import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION

# Required for registration
from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="XrayGPT Gradio Demo")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--options", nargs="+", help="Optional config overrides (deprecated).")
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ========== Initialize Model ==========
print("Initializing Chat...")
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(f"cuda:{args.gpu_id}")

vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device=f"cuda:{args.gpu_id}")
print("Initialization Finished.")

# ========== Gradio UI Functions ==========

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state, []

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if not user_message.strip():
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot.append({"role": "user", "content": user_message})
    chatbot.append({"role": "assistant", "content": None})
    return '', chatbot, chat_state

def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state, img_list=img_list, num_beams=num_beams, temperature=temperature, max_new_tokens=300, max_length=2000)[0]
    for msg in reversed(chatbot):
        if msg["role"] == "assistant" and msg["content"] is None:
            msg["content"] = llm_message
            break
    return chatbot, chat_state, img_list

# ========== UI Definitions ==========

title = """<h1 align="center">🩻 XrayGPT: X-Ray Image Querying with AI</h1>"""
description = """<h3 align="center">Upload a chest X-ray and ask questions about it using AI.</h3>"""
disclaimer = """
<hr>
<h4>Terms of Use:</h4>
<ul>
<li>This tool is for research and educational purposes only.</li>
<li>It does not provide medical advice, diagnosis, or treatment.</li>
<li>Designed by IVAL Lab, MBZUAI.</li>
</ul>
"""

def set_example_xray(example):
    return gr.Image.update(value=example[0])

def set_example_text_input(example_text):
    return gr.Textbox.update(value=example_text[0])

# ========== Gradio Layout ==========

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Upload Chest X-Ray")
            upload_button = gr.Button(value="Upload and Ask Queries", variant="primary")
            clear = gr.Button("Reset")
            num_beams = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Beam Search")
            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label="XrayGPT Chat", type="messages")
            text_input = gr.Textbox(label="Your Question", placeholder="Please upload your X-Ray image first.", interactive=False)

    with gr.Row():
        example_xrays = gr.Dataset(components=[image], label="📂 X-Ray Examples", samples=[
            [os.path.join(os.path.dirname(__file__), "images/example_test_images/img1.png")],
            [os.path.join(os.path.dirname(__file__), "images/example_test_images/img2.png")],
            [os.path.join(os.path.dirname(__file__), "images/example_test_images/img3.png")],
            [os.path.join(os.path.dirname(__file__), "images/example_test_images/img4.png")],
            [os.path.join(os.path.dirname(__file__), "images/example_test_images/img5.png")],
        ])

        example_texts = gr.Dataset(components=[gr.Textbox(visible=False)], label="💬 Prompt Examples", samples=[
            ["Describe the chest x-ray image in detail."],
            ["Summarize the findings from this image."],
            ["What abnormalities do you see?"],
            ["What is your impression of this image?"],
            ["Could you provide a diagnosis based on this image?"]
        ])

    example_xrays.click(fn=set_example_xray, inputs=example_xrays, outputs=image)

    upload_button.click(upload_img, [image, text_input, chat_state],
                        [image, text_input, upload_button, chat_state, img_list])

    example_texts.click(set_example_text_input, inputs=example_texts, outputs=text_input).then(
        gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]
    ).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state],
                      [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature],
        [chatbot, chat_state, img_list]
    )

    clear.click(gradio_reset, [chat_state, img_list],
                [chatbot, image, text_input, upload_button, chat_state, img_list],
                queue=False)

    gr.Markdown(disclaimer)

# ========== Launch ==========
demo.launch(share=True)
