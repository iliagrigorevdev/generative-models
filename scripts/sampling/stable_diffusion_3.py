import os
import torch

from fire import Fire
from pathlib import Path
from diffusers import StableDiffusion3Pipeline
from datetime import datetime
from googletrans import Translator

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

def sample(
    text: str,
    negative_text: str = "",
    use_t5xxl_tokenizer: bool = False,
    translate_text: bool = True,
    resize_image: bool = True,
    output_folder: str = "outputs/sd3",
    seed: int = None,
    num_images: int = 1,
    width: int = 576 * 2,
    height: int = 576 * 2,
    resize_width: int = 576,
    resize_height: int = 576,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
):
    if use_t5xxl_tokenizer:
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch.float16
        )

    pipe = pipe.to("cuda")

    if translate_text:
        translator = Translator()
        prompt = translator.translate(text).text
        print("Text: " + text)
        if negative_text:
            negative_prompt = translator.translate(negative_text).text
            print("Negative text: " + negative_text)
        else:
            negative_prompt = negative_text
    else:
        prompt = text
        negative_prompt = negative_text
    print("Prompt: " + prompt)
    if negative_prompt:
        print("Negative prompt: " + negative_prompt)

    generator = torch.Generator(device='cuda')

    for i in range(num_images):
        if i == 0 or seed is None:
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()
            print(f"Seed: {generator.initial_seed()}")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        os.makedirs(output_folder, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d%H%M%S_") + str(generator.initial_seed()) + ".png"
        path = os.path.join(output_folder, name)
        if resize_image:
            image = image.resize((resize_width, resize_height))
        image.save(path)
        image.show()
        print(f"Image: {Path(path).absolute()}")

if __name__ == "__main__":
    Fire(sample)
