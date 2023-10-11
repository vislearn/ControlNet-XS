import scripts.control_utils as cu
import torch
from PIL import Image
import os 
import numpy as np
import cv2


os.chdir("/home/dell/workspace/controlnet/ControlNet-XS")

# path_to_config = 'ControlNet-XS-main/configs/inference/sdxl/sdxl_encD_canny_48m.yaml'
path_to_config = './configs/inference/sdxl/sdxl_encD_canny_48m.yaml'




image_path = 'PATH/TO/IMAGES/Shoe.png'
image_path = '/home/dell/workspace/controlnet/ControlNet-XS/shoes.png'

canny_high_th = 250
canny_low_th = 100
size = 768
num_samples=2

image = cu.get_image(image_path, size=size)
edges = cu.get_canny_edges(image, low_th=canny_low_th, high_th=canny_high_th)

from datasets import load_dataset
ds = load_dataset("/home/dell/workspace/dataset/lambdalabs___pokemon-blip-captions/", split="train")
sample = ds[0]
# display(sample["image"].resize((256, 256)))
print(sample["text"])
image = sample["image"].resize(size = (size, size))
v_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
print(f">> input guiding shape , {v_image.shape}")

model = cu.create_model(path_to_config).to('cuda')

with torch.no_grad():
    samples, controls = cu.get_sdxl_sample(
        # guidance=edges,
        guidance=v_image,
        ddim_steps=10,
        num_samples=num_samples,
        model=model,
        shape=[4, size // 8, size // 8],
        control_scale=0.95,
        prompt='cinematic, shoe in the streets, made from meat, photorealistic shoe, highly detailed',
        n_prompt='lowres, bad anatomy, worst quality, low quality',
    )

print("done!")

Image.fromarray(cu.create_image_grid(samples)).save('SDXL_MyShoe.png')

