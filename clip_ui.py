import gradio as gr
import click
from clip_interrogator import Config, Interrogator
from utils import get_image_files
from PIL import Image
import os
from tqdm import tqdm
import torch


MODEL = None


def load_model(
        model_name: str,
        caption_model_name: str = "blip2-2.7b"
) -> str:
    global MODEL
    if MODEL is not None:
        del MODEL.clip_model
        del MODEL.caption_model
        del MODEL
        torch.empty()
    config = Config(
        clip_model_name=model_name,
        caption_model_name=caption_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    config.chunk_size = 2048
    config.flavor_intermediate_count = 512
    config.blip_num_beams = 64
    ci = Interrogator(config=config)

    MODEL = ci
    return "model loaded"


def clip_text_fix_ui():
    pass


def image2text_from_files(file_mode: str, mode: str, input_dir: str, image_path: str):
    fns = get_image_files(input_dir)
    fns.append(image_path)
    datas = image2text(file_mode, mode, fns)
    return gr.update(
        value=datas
    )


def image2text(file_mode: str, mode: str, fns: list[str | os.PathLike]):
    global MODEL

    fmode = "w+"
    if file_mode == 'convert':
        fmode = "w+"
    elif file_mode == 'append':
        fmode = "a+"
    elif file_mode == "ignore":
        pass
    print(f"file mode: {file_mode}")
    images_tags = []
    for fn in tqdm(fns, desc="image2text", total=len(fns)):
        text_fn = f"{os.path.splitext(os.path.basename(fn))[0]}.txt"
        text_fn = os.path.join(os.path.dirname(fn), text_fn)
        if file_mode == "ignore" and os.path.exists(text_fn):
            print(f"ignore {text_fn}")
            continue

        image = Image.open(fn)
        image = image.convert('RGB')

        if mode == 'best':
            prompt = MODEL.interrogate(image)
        elif mode == 'classic':
            prompt = MODEL.interrogate_classic(image)
        elif mode == 'fast':
            prompt = MODEL.interrogate_fast(image)
        elif mode == 'negative':
            prompt = MODEL.interrogate_negative(image)

        print(f"prompt: {prompt}")
        print(f"save to {text_fn}")

        with open(text_fn, fmode, encoding="utf-8") as f:
            f.write(prompt)
        images_tags.append([fn, prompt])
    return images_tags


def clip_image2text_ui():
    with gr.Row():
        blip_model = gr.Dropdown(
            label="blip model",
            value="blip-large",
            choices=[
                "blip-base",
                "blip-large",
                "blip2-2.7b",
                "blip2-flan-t5-xl",
                "git-large-coco"
            ]
        )
        clip_model = gr.Dropdown(
            label="clip model",
            choices=["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k", "ViT-bigG-14/laion2b_s39b_b160k"],
            value="ViT-bigG-14/laion2b_s39b_b160k"
        )
        mode = gr.Dropdown(
            label="mode",
            choices=["best", "classic", "fast", "negative"],
            value="best"
        )
        file_mode = gr.Dropdown(
            label="file mode",
            choices=["append", "convert", "ignore"],
            value="convert"
        )
        status = gr.Textbox(label="status", value="", lines=1)
        load_model_btn = gr.Button(label="load model", value="load model")

    with gr.Row():
        input_dir = gr.Textbox(label="input dir", value="")

    input_image = gr.Image(label="input image", type="filepath")
    datas = gr.Dataframe(
        headers=["image", "text"],
        datatype=["str", "str"],
    )
    gen_btn = gr.Button(label="generate")
    gen_btn.click(
        image2text_from_files,
        inputs=[file_mode, mode, input_dir, input_image],
        outputs=[datas]
    )
    load_model_btn.click(
        load_model,
        inputs=[clip_model, blip_model],
        outputs=[status]
    )


if __name__ == '__main__':
    with gr.Blocks() as app:
        with gr.Tab("image2text"):
            clip_image2text_ui()

    app.launch(enable_queue=False, share=False, inbrowser=True)
