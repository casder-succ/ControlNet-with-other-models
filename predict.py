# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from model import Model
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def get_img_by_url(url):
    response = requests.get(url).content
    nparr = np.frombuffer(response, np.uint8)
    # convert to image array
    img = cv2.imdecode(nparr,cv2.IMREAD_UNCHANGED)
    return img

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.baseModel = Model('runwayml/stable-diffusion-v1-5', 'v1-5-pruned.ckpt')

    def predict(
        self,
        style: str = Input(description="Style of model", default='painting'),
        seed: str = Input(description="seed", default="1337"),
        isWarmup: str = Input(description="need be setted false", default="false"),
        prompt: str = Input(description="Model prompt", default=""),
        reference: str = Input(description="reference image", default=""),
    ) -> Path:
        if isWarmup=="true":
            return

        styles_map = {
            'painting': 'Masterpiece, cinematic lighting, photorealistic, realistic, extremely detailed, artgerm, greg rutkowski, alphonse mucha',
            'anime': 'modern anime style art detailed shinkai makoto vibrant Studio animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant digital painting artstation'
        }

        pose_input = get_img_by_url(reference)
        n_prompt = "Ugly, lowres, duplicate, morbid, mutilated, out of frame, extra fingers, extra limbs, extra legs, extra heads, extra arms, extra breasts, extra nipples, extra head, extra digit, poorly drawn hands, poorly drawn face, mutation, mutated hands, bad anatomy, long neck, signature, watermark, username, blurry, artist name, deformed, distorted fingers, distorted limbs, distorted legs, distorted heads, distorted arms, distorted breasts, distorted nipples, distorted head, distorted digit"
        a_prompt = ""
        p_prompt = f"{prompt}, {styles_map[style]}"
        num_samples = 1
        image_resolution = 768
        detect_resolution = 768
        ddim_steps = 20
        scale = 12
        eta = 0

        outputs = self.baseModel.process_depth(pose_input, p_prompt, a_prompt, n_prompt,
                     num_samples, image_resolution, detect_resolution,
                     ddim_steps, scale, seed, eta)
        boy_image = cv2.cvtColor(outputs[1], cv2.COLOR_BGR2RGB)
        cv2.imwrite("output.png", boy_image)
        return Path("output.png")
