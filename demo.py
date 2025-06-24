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

from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument("--options", nargs="+", help="Override config values with key=value pairs.")
    return parser.parse_args()

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# ======== Model Initialization ========
print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')

vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')
print('Initialization Finished')

# ======== Gradio Interface ========
def gradio_reset(chat_state, img_list):
    if chat_state:
        chat_state.messages = []
    if img_list:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    _ = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if not user_message:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot.append([user_message, None])
    return '', chatbot, chat_state

def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    response = chat.answer(conv=chat_state, img_list=img_list, num_beams=num_beams, temperature=temperature, max_new_tokens=300, max_length=2000)[0]
    chatbot[-1][1] = response
    return chatbot, chat_state, img_list

def set_example_xray(example):
    return gr.update(value=example[0])

def set_example_text_input(example_text):
    return gr.update(value=example_text[0])

title = "<h1 align='center'>Demo of XrayGPT</h1>"
description = "<h3>Upload your X-Ray images and start asking queries!</h3>"
disclaimer = """
<h1>Terms of Use:</h1>
<ul>
    <li>This service is for research and educational use only and does not replace professional medical advice.</li>
    <li>Results are generated using AI-based analysis and may not be definitive.</li>
    <li>No warranties are made regarding reliability or completeness. This service is under continual improvement.</li>
</ul>
<hr>
<h3 align='center'>Designed and Developed by IVAL Lab, MBZUAI</h3>
"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload and Ask Queries", interactive=True, variant="primary")
            clear = gr.Button("Reset")
            num_beams = gr.Slider(1, 10, value=1, step=1, label="Beam Search")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
        with gr.Column(scale=2):
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='XrayGPT', type='messages')
            text_input = gr.Textbox(label='User', placeholder='Please upload your X-Ray image.', interactive=False)

    example_xrays = gr.Dataset(
        label="X-Ray Examples",
        components=[image],
        samples=[[os.path.join(os.path.dirname(__file__), f"images/example_test_images/img{i}.png")] for i in range(1, 10)]
    )
    example_xrays.click(fn=set_example_xray, inputs=example_xrays, outputs=[image])

    example_texts = gr.Dataset(
        label="Prompt Examples",
        components=[gr.Textbox(visible=False)],
        samples=[
            ["Describe the given chest x-ray image in detail."],
            ["Highlight abnormalities or concerns in this chest x-ray."],
            ["What are the key features of this scan?"],
            ["Could you describe the findings and impression?"]
        ]
    )
    example_texts.click(fn=set_example_text_input, inputs=example_texts, outputs=[text_input]).then(
        gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]
    ).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list])

    gr.Markdown(disclaimer)

demo.launch(share=True, enable_queue=True)
