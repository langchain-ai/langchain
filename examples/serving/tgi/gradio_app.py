# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
import time

import gradio as gr
import requests

import sys
sys.path.insert(0, './')
from conversation import get_conv_template
from fastchat.constants import LOGDIR
from fastchat.utils import (
    build_logger,
)


from langchain_community.llms import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain_community.callbacks import streaming_stdout
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
from intel_extension_for_transformers.langchain.embeddings import HuggingFaceBgeEmbeddings
from intel_extension_for_transformers.langchain.vectorstores import Chroma
import torch
import intel_extension_for_pytorch as ipex

ENDPOINT_URL = "http://localhost:8080"
callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
llm = HuggingFaceEndpoint(
    endpoint_url=ENDPOINT_URL,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    streaming=True,
    callbacks=callbacks
)

embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
embeddings.client= ipex.optimize(embeddings.client.eval(), dtype=torch.bfloat16)
knowledge_base = Chroma.reload(persist_directory='./output', embedding=embeddings)

retriever = VectorStoreRetriever(vectorstore=knowledge_base, search_type='mmr', search_kwargs={'k':1, 'fetch_k':5})
retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)


code_highlight_css = """
#chatbot .hll { background-color: #ffffcc }
#chatbot .c { color: #408080; font-style: italic }
#chatbot .err { border: 1px solid #FF0000 }
#chatbot .k { color: #008000; font-weight: bold }
#chatbot .o { color: #666666 }
#chatbot .ch { color: #408080; font-style: italic }
#chatbot .cm { color: #408080; font-style: italic }
#chatbot .cp { color: #BC7A00 }
#chatbot .cpf { color: #408080; font-style: italic }
#chatbot .c1 { color: #408080; font-style: italic }
#chatbot .cs { color: #408080; font-style: italic }
#chatbot .gd { color: #A00000 }
#chatbot .ge { font-style: italic }
#chatbot .gr { color: #FF0000 }
#chatbot .gh { color: #000080; font-weight: bold }
#chatbot .gi { color: #00A000 }
#chatbot .go { color: #888888 }
#chatbot .gp { color: #000080; font-weight: bold }
#chatbot .gs { font-weight: bold }
#chatbot .gu { color: #800080; font-weight: bold }
#chatbot .gt { color: #0044DD }
#chatbot .kc { color: #008000; font-weight: bold }
#chatbot .kd { color: #008000; font-weight: bold }
#chatbot .kn { color: #008000; font-weight: bold }
#chatbot .kp { color: #008000 }
#chatbot .kr { color: #008000; font-weight: bold }
#chatbot .kt { color: #B00040 }
#chatbot .m { color: #666666 }
#chatbot .s { color: #BA2121 }
#chatbot .na { color: #7D9029 }
#chatbot .nb { color: #008000 }
#chatbot .nc { color: #0000FF; font-weight: bold }
#chatbot .no { color: #880000 }
#chatbot .nd { color: #AA22FF }
#chatbot .ni { color: #999999; font-weight: bold }
#chatbot .ne { color: #D2413A; font-weight: bold }
#chatbot .nf { color: #0000FF }
#chatbot .nl { color: #A0A000 }
#chatbot .nn { color: #0000FF; font-weight: bold }
#chatbot .nt { color: #008000; font-weight: bold }
#chatbot .nv { color: #19177C }
#chatbot .ow { color: #AA22FF; font-weight: bold }
#chatbot .w { color: #bbbbbb }
#chatbot .mb { color: #666666 }
#chatbot .mf { color: #666666 }
#chatbot .mh { color: #666666 }
#chatbot .mi { color: #666666 }
#chatbot .mo { color: #666666 }
#chatbot .sa { color: #BA2121 }
#chatbot .sb { color: #BA2121 }
#chatbot .sc { color: #BA2121 }
#chatbot .dl { color: #BA2121 }
#chatbot .sd { color: #BA2121; font-style: italic }
#chatbot .s2 { color: #BA2121 }
#chatbot .se { color: #BB6622; font-weight: bold }
#chatbot .sh { color: #BA2121 }
#chatbot .si { color: #BB6688; font-weight: bold }
#chatbot .sx { color: #008000 }
#chatbot .sr { color: #BB6688 }
#chatbot .s1 { color: #BA2121 }
#chatbot .ss { color: #19177C }
#chatbot .bp { color: #008000 }
#chatbot .fm { color: #0000FF }
#chatbot .vc { color: #19177C }
#chatbot .vg { color: #19177C }
#chatbot .vi { color: #19177C }
#chatbot .vm { color: #19177C }
#chatbot .il { color: #666666 }
"""

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)
moderation_msg = (
    "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."
)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "NeuralChat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(controller_url):
    ret = requests.post(controller_url + "/v1/models")
    model_data = ret.json()["data"]
    models = [model['id'] for model in model_data]
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);
    return url_params;
    }
"""


def load_demo_single(models, url_params):
    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return load_demo_single(models, url_params)


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "") + (disable_btn,) * 5


def add_text(state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")

    if state is None:
        state = get_conv_template("neural-chat-7b-v2")

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5

    text = text[:2560]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code

def huggingface_api_stream_iter(
    prompt,
    temperature,
    top_p,
    repetition_penalty,
    max_new_tokens
):
    text = ""

    for token in retrievalQA({"query": prompt}):
        text += " " + token
        data = {
            "text": text.strip(),
            "error_code": 0,
        }
        yield data

def http_bot(state, model_selector, temperature, max_new_tokens, topk, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    temperature = float(temperature)
    max_new_tokens = int(max_new_tokens)
    topk = int(topk)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # model conversation name: "mpt-7b-chat", "chatglm", "chatglm2", "llama-2",
        #                          "mistral", "neural-chat-7b-v3-1", "neural-chat-7b-v3",
        #                          "neural-chat-7b-v2", "neural-chat-7b-v1-1"
        # First round of Conversation
        if "Llama-2-7b-chat-hf" in model_name:
            model_name = "llama-2"
        elif "chatglm"  in model_name:
            model_name = model_name.split('-')[0]
        new_state = get_conv_template(model_name.split('/')[-1])
        #new_state.conv_id = uuid.uuid4().hex
        #new_state.model_name = state.model_name or model_selector
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Construct prompt
    prompt = state.get_prompt()
    # print("prompt==============", prompt)

    start_time = time.time()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    # Stream output
    stream_iter = huggingface_api_stream_iter(prompt=prompt,
                                    temperature=temperature,
                                    top_p=0.95,
                                    repetition_penalty = 1.0,
                                    max_new_tokens = max_new_tokens,
                                    )

    try:
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                output = data["text"].strip()
                state.messages[-1][-1] = output + "▌"
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                state.messages[-1][-1] = output
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
            time.sleep(0.005)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time() - start_time
    elapsed_time = "\n✅generation elapsed time: {}s".format(round(finish_tstamp, 4))

    # elapsed_time =  "\n{}s".format(round(finish_tstamp, 4))
    # elapsed_time =  "<p class='time-style'>{}s </p>".format(round(finish_tstamp, 4))

    # state.messages[-1][-1] = state.messages[-1][-1][:-1] + elapsed_time
    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "topk": topk,
            },
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


block_css = (
    code_highlight_css
    + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
#notice_markdown th {
    display: none;
}

#notice_markdown {
    text-align: center;
    background: #2e78c4;
    padding: 1%;
    height: 4.3rem;
    color: #fff !important;
    margin-top: 0;
}

#notice_markdown p{
    color: #fff !important;
}


#notice_markdown h1, #notice_markdown h4 {
    color: #fff;
    margin-top: 0;
}

gradio-app {
    background: linear-gradient(to bottom, #86ccf5, #3273bf) !important;
    padding: 3%;
}

.gradio-container {
    margin: 0 auto !important;
    width: 70% !important;
    padding: 0 !important;
    background: #fff !important;
    border-radius: 5px !important;
}

#chatbot {
    border-style: solid;
    overflow: visible;
    margin: 1% 4%;
    width: 90%;
    box-shadow: 0 15px 15px -5px rgba(0, 0, 0, 0.2);
    border: 1px solid #ddd;
}

#chatbot::before {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 60px;
    height: 60px;
    background-image: url(https://i.postimg.cc/gJzQTQPd/Microsoft-Teams-image-73.png);
    background-repeat: no-repeat;
    background-position: center center;
    background-size: contain;
}

#chatbot::after {
    content: "";
    position: absolute;
    top: 0;
    right: 60px;
    width: 60px;
    height: 60px;
    background-image: url(https://i.postimg.cc/QCBQ45b4/Microsoft-Teams-image-44.png);
    background-repeat: no-repeat;
    background-position: center center;
    background-size: contain;
}

#chatbot .wrap {
    margin-top: 30px !important;
}


#text-box-style, #btn-style {
    width: 90%;
    margin: 1% 4%;
}


.user, .bot {
    width: 80% !important;

}

.bot {
    white-space: pre-wrap !important;
    line-height: 1.3 !important;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;

}

#btn-send-style {
    background: rgb(0, 180, 50);
    color: #fff;
    }

#btn-list-style {
    background: #eee0;
    border: 1px solid #0053f4;
}

.title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff !important;
    display: flex;
    justify-content: center;
}

footer {
    display: none !important;
}

.footer {
    margin-top: 2rem !important;
    text-align: center;
    border-bottom: 1px solid #e5e5e5;
}

.footer>p {
    font-size: .8rem;
    display: inline-block;
    padding: 0 10px;
    transform: translateY(10px);
    background: white;
}

.img-logo {
    width: 3.3rem;
    display: inline-block;
    margin-right: 1rem;
}

.img-logo-style {
    width: 3.5rem;
    float: left;
}

.img-logo-right-style {
    width: 3.5rem;
    display: inline-block !important;
}

.neural-studio-img-style {
     width: 50%;
    height: 20%;
    margin: 0 auto;
}

.acknowledgments {
    margin-bottom: 1rem !important;
    height: 1rem;
}
"""
)


def build_single_model_ui(models):

    notice_markdown = """
<div class="title">
<div style="
    color: #fff;
">Large Language Model <p style="
    font-size: 0.8rem;
">Future Gen Intel® Xeon® (codenamed Granite Rapids) with Intel® AMX</p></div>

</div>
"""
    # <div class="footer">
    #                 <p>Powered by <a href="https://github.com/intel/intel-extension-for-transformers" style="text-decoration: underline;" target="_blank">Intel Extension for Transformers</a> and <a href="https://github.com/intel/intel-extension-for-pytorch" style="text-decoration: underline;" target="_blank">Intel Extension for PyTorch</a>
    #                 <img src='https://i.postimg.cc/Pfv4vV6R/Microsoft-Teams-image-23.png' class='img-logo-right-style'/></p>
    #         </div>
    #         <div class="acknowledgments">
    #         <p></p></div>

    learn_more_markdown =  """<div class="footer">
                    <p>Powered by <a href="https://github.com/intel/intel-extension-for-transformers" style="text-decoration: underline;" target="_blank">Intel Extension for Transformers</a> and <a href="https://github.com/intel/intel-extension-for-pytorch" style="text-decoration: underline;" target="_blank">Intel Extension for PyTorch</a>
                    </p>
            </div>
            <div class="acknowledgments">
            <p></p></div>

        """

    state = gr.State()
    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Row(elem_id="model_selector_row", visible=False):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
        ).style(container=False)

    chatbot = gr.Chatbot(elem_id="chatbot", visible=False).style(height=550)
    with gr.Row(elem_id="text-box-style"):
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False, elem_id="btn-send-style")

    with gr.Accordion("Parameters", open=False, visible=False, elem_id="btn-style") as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.001,
            step=0.1,
            interactive=True,
            label="Temperature",
            visible=False,
        )
        max_output_tokens = gr.Slider(
            minimum=0,
            maximum=1024,
            value=512,
            step=1,
            interactive=True,
            label="Max output tokens",
        )
        topk = gr.Slider(
            minimum=1,
            maximum=10,
            value=1,
            step=1,
            interactive=True,
            label="TOP K",
        )


    with gr.Row(visible=False, elem_id="btn-style") as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False, visible=False, elem_id="btn-list-style")
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False, visible=False, elem_id="btn-list-style")
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False, visible=False, elem_id="btn-list-style")
        # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False, elem_id="btn-list-style")
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False, elem_id="btn-list-style")


    gr.Markdown(learn_more_markdown)

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens, topk],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens, topk],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens, topk],
        [state, chatbot] + btn_list,
    )

    return state, model_selector, chatbot, textbox, send_btn, button_row, parameter_row


def build_demo(models):
    with gr.Blocks(
        title="NeuralChat · Intel",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        (
            state,
            model_selector,
            chatbot,
            textbox,
            send_btn,
            button_row,
            parameter_row,
        ) = build_single_model_ui(models)

        if model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [
                    state,
                    model_selector,
                    chatbot,
                    textbox,
                    send_btn,
                    button_row,
                    parameter_row,
                ],
                _js=get_window_url_params,
            )
        else:
            raise ValueError(f"Unknown model list mode: {model_list_mode}")

    return demo


if __name__ == "__main__":
    host = "0.0.0.0"

    concurrency_count = 10
    model_list_mode = "once"
    share = False

    models = ["Intel/neural-chat-7b-v3-3"]
    demo = build_demo(models)
    demo.queue(
        concurrency_count=concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=host, server_port=80, share=share, max_threads=200
    )
