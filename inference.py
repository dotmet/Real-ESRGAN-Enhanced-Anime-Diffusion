
import os
import random
import glob

import autocuda
from pyabsa.utils.pyabsa_utils import fprint

from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, \
    DPMSolverMultistepScheduler
import gradio as gr
import torch
from PIL import Image
import utils
import datetime
import time
import psutil

from RealESRGANv030.interface import realEsrgan

start_time = time.time()
is_colab = utils.is_google_colab()

device = autocuda.auto_cuda()
dtype = torch.float16 if device != 'cpu' else torch.float32

class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None


models = [
    Model("anything v3", "Linaqruf/anything-v3.0", "anything v3 style"),
]
#  Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
#  Model("Balloon Art", "Fictiverse/Stable_Diffusion_BalloonArt_Model", "BalloonArt "),
#  Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
#  Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy ")
# Model("Pokémon", "lambdalabs/sd-pokemon-diffusers", ""),
# Model("Pony Diffusion", "AstraliteHeart/pony-diffusion", ""),
# Model("Robo Diffusion", "nousr/robo-diffusion", ""),

scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)

custom_model = None
if is_colab:
    models.insert(0, Model("Custom model"))
    custom_model = models[0]

last_mode = "txt2img"
current_model = models[1] if is_colab else models[0]
current_model_path = current_model.path

if is_colab:
    pipe = StableDiffusionPipeline.from_pretrained(current_model.path, torch_dtype=dtype, scheduler=scheduler,
                                                   safety_checker=lambda images, clip_input: (images, False))

else:  # download all models
    print(f"{datetime.datetime.now()} Downloading vae...")
    vae = AutoencoderKL.from_pretrained(current_model.path, subfolder="vae", torch_dtype=dtype)
    for model in models:
        try:
            print(f"{datetime.datetime.now()} Downloading {model.name} model...")
            unet = UNet2DConditionModel.from_pretrained(model.path, subfolder="unet", torch_dtype=dtype)
            model.pipe_t2i = StableDiffusionPipeline.from_pretrained(model.path, unet=unet, vae=vae,
                                                                     torch_dtype=dtype, scheduler=scheduler,
                                                                     safety_checker=None)
            model.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(model.path, unet=unet, vae=vae,
                                                                            torch_dtype=dtype,
                                                                            scheduler=scheduler, safety_checker=None)
        except Exception as e:
            print(f"{datetime.datetime.now()} Failed to load model " + model.name + ": " + str(e))
            models.remove(model)
    pipe = models[0].pipe_t2i

# model.pipe_i2i = torch.compile(model.pipe_i2i)
# model.pipe_t2i = torch.compile(model.pipe_t2i)
if torch.cuda.is_available():
    pipe = pipe.to(device)


# device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"


def error_str(error, title="Error"):
    return f"""#### {title}
            {error}""" if error else ""


def custom_model_changed(path):
    models[0].path = path
    global current_model
    current_model = models[0]


def on_model_change(model_name):
    prefix = "Enter prompt. \"" + next((m.prefix for m in models if m.name == model_name),
                                       None) + "\" is prefixed automatically" if model_name != models[
        0].name else "Don't forget to use the custom model prefix in the prompt!"

    return gr.update(visible=model_name == models[0].name), gr.update(placeholder=prefix)


def inference(model_name, prompt, guidance, steps, width=512, height=512, seed=0, img=None, strength=0.5,
              neg_prompt="", scale_factor=4, tile=200, out_dir='imgs', ext='auto'):

    fprint(psutil.virtual_memory())  # print memory usage
    fprint(f"\nPrompt: {prompt}")
    global current_model
    for model in models:
        if model.name == model_name:
            current_model = model
            model_path = current_model.path

    generator = torch.Generator(device).manual_seed(seed) if seed != 0 else None
    
    if img is not None:
        img = None if len(img.split())==0 else img
    try:
        if img is not None:
            return img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height,
                              generator, scale_factor, tile, out_dir, ext), None
        else:
            return txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator,
                              scale_factor, tile, out_dir, ext), None
    except Exception as e:
        return None, error_str(e)
    # if img is not None:
    #     return img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, width, height,
    #                       generator, scale_factor), None
    # else:
    #     return txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, generator, scale_factor), None


def txt_to_img(model_path, prompt, neg_prompt, guidance, steps, width, height, 
               generator, scale_factor, tile, out_dir, ext='auto'):
    print(f"{datetime.datetime.now()} \ntxt_to_img, model: {current_model.name}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
            pipe = StableDiffusionPipeline.from_pretrained(current_model_path, 
                                                           torch_dtype=dtype,
                                                           scheduler=scheduler,
                                                           safety_checker=lambda images, 
                                                           clip_input: (images, False))
        else:
            pipe = current_model.pipe_t2i

        if torch.cuda.is_available():
            pipe = pipe.to(device)
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt
    result = pipe(
        prompt,
        negative_prompt=neg_prompt,
        # num_images_per_prompt=n_images,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator)
    # result.images[0] = magnifier.magnify(result.images[0], scale_factor=scale_factor)

    # save image
    img_file = "imgs/result-{}.png".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    result.images[0].save(img_file)
    
    # enhance resolution
    if scale_factor>1:
        fp32 = True if device=='cpu' else False
        result.images[0] = realEsrgan(
                            input_dir = img_file,
                            suffix = '',
                            output_dir = out_dir,
                            fp32 = fp32,
                            outscale = scale_factor,
                            tile = tile,
                            out_ext = ext,
        )[0]
        print('Rescale image complete')
    
    return replace_nsfw_images(result)


def img_to_img(model_path, prompt, neg_prompt, img, strength, guidance, steps, 
               width, height, generator, scale_factor, tile, out_dir, ext):
    fprint(f"{datetime.datetime.now()} \nimg_to_img, model: {model_path}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(current_model_path, torch_dtype=dtype,
                                                                  scheduler=scheduler,
                                                                  safety_checker=lambda images, clip_input: (
                                                                      images, False))
        else:
            # pipe = pipe.to("cpu")
            pipe = current_model.pipe_i2i

        if torch.cuda.is_available():
            pipe = pipe.to(device)
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    with Image.open(img) as img:
        ratio = min(height / img.height, width / img.width)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            # num_images_per_prompt=n_images,
            image=img,
            num_inference_steps=int(steps),
            strength=strength,
            guidance_scale=guidance,
            # width=width,
            # height=height,
            generator=generator)
    
    # save image
    img_path = "imgs/result-{}.png".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    result.images[0].save(img_path)
    
    # enhance resolution
    if scale_factor>1:
        fp32 = True if device=='cpu' else False
        result.images[0] = realEsrgan(
                            input_dir = img_path,
                            suffix = '',
                            output_dir= out_dir,
                            fp32 = fp32,
                            outscale = scale_factor,
                            tile = tile,
                            out_ext = ext,)
        print('Complete')
    
    return replace_nsfw_images(result)


def replace_nsfw_images(results):
    if is_colab:
        return results.images[0]
    if hasattr(results, "nsfw_content_detected") and results.nsfw_content_detected:
        for i in range(len(results.images)):
            if results.nsfw_content_detected[i]:
                results.images[i] = Image.open("nsfw.png")
    return results.images[0]
    
def split_text(file=None, text=None, marker='\n'):
    if file is not None:
        if os.path.isfile(file):
            with open(file, 'r') as f:
                text = f.read()
        else:
            text = file
    collection = []
    texts = text.split(marker)
    for txt in texts:
        if len(txt)>0:
            collection.append(txt)
    return collection
    
if __name__ == '__main__':

    args = utils.parse_args()
    
    n = args.n if args.n>0 else 114514
    img = args.image
    if img is not None and len(img.split())!=0:
        if os.path.isfile(img):
            images = [img]
        else:
            images = sorted(glob.blob(os.path.join(img, "*")))
    else:
        images = ['']*n
    
    prompt = split_text(args.words)
    neg_prompt = split_text(args.neg_words)
        
    for i,image in zip(range(n), images):
        if i>=n:
            print('--- Task done ---')
            break
        else:
            print(f'\nGenerating image {i+1} ...\n')
            inference(
                args.model_name,
                random.choice(prompt),
                args.guidance,
                args.gen_steps,
                args.width,
                args.height,
                args.seed,
                image,
                args.strength,
                random.choice(neg_prompt),
                args.scale,
                args.tile,
                args.out_dir,
                args.extension,
            )