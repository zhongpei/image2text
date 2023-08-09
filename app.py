import gradio as gr

from clip_interrogator import Config, Interrogator
from utils import get_image_files
from PIL import Image
import os
from tqdm import tqdm
import torch

MODEL = None


def load_model(
        model_name: str,
        caption_model_name: str = "blip2-2.7b",
        low_memory: bool = False,
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
    config.cache_path = os.path.join(os.path.dirname(__file__), "cache")
    if not os.path.exists(config.cache_path):
        os.mkdir(config.cache_path)
    if low_memory:
        config.apply_low_vram_defaults()
        config.chunk_size = 1024
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


def image_analysis(image, topk=5):
    global MODEL
    ci = MODEL

    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, topk)
    top_artists = ci.artists.rank(image_features, topk)
    top_movements = ci.movements.rank(image_features, topk)
    top_trendings = ci.trendings.rank(image_features, topk)
    top_flavors = ci.flavors.rank(image_features, topk)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in
                      zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in
                      zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}

    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks


def clip_image2text_ui():
    with gr.Row():
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


if __name__ == '__main__':
    with gr.Blocks() as app:
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
            low_memory = gr.Checkbox(label="low memory", value=False)
            status = gr.Textbox(label="status", value="", lines=1)
            load_model_btn = gr.Button(label="load model", value="load model")

        with gr.Tab("image2text"):
            clip_image2text_ui()
        with gr.Tab("image_analyzer"):
            with gr.Column():
                with gr.Row():
                    topk = gr.Slider(label="topk", min=1, max=20, value=10)
                with gr.Row():
                    image = gr.Image(type='pil', label="Image")
                with gr.Row():
                    medium = gr.Label(label="Medium", num_top_classes=20)
                    artist = gr.Label(label="Artist", num_top_classes=20)
                    movement = gr.Label(label="Movement", num_top_classes=20)
                    trending = gr.Label(label="Trending", num_top_classes=20)
                    flavor = gr.Label(label="Flavor", num_top_classes=20)
            button = gr.Button("Analyze")
            button.click(image_analysis, inputs=[image, topk], outputs=[medium, artist, movement, trending, flavor])
        load_model_btn.click(
            load_model,
            inputs=[clip_model, blip_model, low_memory],
            outputs=[status]
        )
    app.launch(enable_queue=False, share=False, inbrowser=True)
