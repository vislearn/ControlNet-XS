import scripts.control_utils as cu
import torch
from PIL import Image
import os 
import numpy as np
import cv2

# cp  ~/workspace/models/vit-h/*  ~/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K 

os.chdir("/home/dell/workspace/controlnet/ControlNet-XS")

# path_to_config = 'PATH/TO/CONFIG/sd21_encD_depth_14m.yaml'
path_to_config = './configs/inference/sd/sd21_encD_depth_14m.yaml'
# model = cu.create_model(path_to_config).to('cuda')

# 注入配置，修改模型文件~


model = cu.create_model(path_to_config).to('cuda')

size = 768
image_path = 'PATH/TO/IMAGES/Shoe.png'
from datasets import load_dataset
ds = load_dataset("/home/dell/workspace/dataset/lambdalabs___pokemon-blip-captions/", split="train")
sample = ds[0]
# display(sample["image"].resize((256, 256)))
print(sample["text"])



image = sample["image"].resize(size = (size, size))
# image = cu.get_image(image_path, size=size)
# depth = cu.get_midas_depth(image, max_resolution=size)
# print(f">> type of depth {type(depth)}, shape {depth.shape}")

v_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
print(f">> input guiding shape , {v_image.shape}")

# depth = np.array(sample["image"])
num_samples = 2

samples, controls = cu.get_sd_sample(
    guidance=v_image,
    ddim_steps=10,
    num_samples=num_samples,
    model=model,
    shape=[4, size // 8, size // 8],
    control_scale=0.95,
    prompt='cinematic, advertising shot, shoe in a city street, photorealistic shoe, colourful, highly detailed',
    n_prompt='low quality, bad quality, sketches'
)


Image.fromarray(cu.create_image_grid(samples)).save('pokeman_o2o.png')




# ------------------------------------------
#           train   SD 21 controlnet XS  -- required  args ----------
name = "sd21_t"           #  n help="postfix for logdir",
no_date = False           #  true  skip date generate for logdir
resume = False            # r resume from logdir or checkpoint in logdir"
base = "base_config.yaml"     # b "paths to base configs. Loaded from left-to-right.
train=True                    # t
no_test = True                # disable test
p       =    ""                 # project     name of new or path to existing project
d      = True                 # debug
s       = 1                   # seed    
f       = "x"                 # postfix   post-postfix for default name",
projectname = "stablediffusion"        # 
l           =  "logs"                  # "directory for logging dat shit",
scale_lr    =  0.001                   # scale base-lr by ngpu * batch_size * n_accumulate",
legacy_naming = True                   #    name run based on config file name if true, else by whole path"如果为True，基于配置文件名命名运行；否则，基于整个路径命名
enable_tf32   = True                   #  nables the TensorFloat32 format both for matmuls and cuDNN for pytorch 1.12",

startup       =   ""                   #   Startuptime from distributed script

wandb =  True                          #
no_base_name       =   True             #    log to wandb"
resume_from_checkpoint     =    None    #   single checkpoint file to resume from",
