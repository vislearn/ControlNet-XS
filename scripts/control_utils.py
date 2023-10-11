import sys
import os
import copy
from functools import partial
import einops

import torch
import torchvision.transforms as tt
from pytorch_lightning import seed_everything

import numpy as np
from PIL import Image


sys.path.append('../ldm')

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from annotator.util import HWC3
from annotator.canny import CannyDetector
from ldm.models.diffusion.ddim import DDIMSampler
from annotator.midas import MidasDetector
from ldm.data.util import resize_image_pil


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(
            torch.load(
                ckpt_path, map_location=torch.device(location)
            )
        )
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def create_image_grid(images: np.ndarray, grid_size=None):
    """
    Create a grid with the fed images
    Args:
        images (np.array): array of images
        grid_size (tuple(int)): size of grid (grid_width, grid_height)
    Returns:
        grid (np.array): image grid of size grid_size
    """
    # Sanity check
    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, c = images.shape
    # If user specifies the grid shape, use it
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
        # If one of the sides is None, then we must infer it (this was divine inspiration)
        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)

    # Otherwise, we can infer it by the number of images (priority is given to grid_w)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    # Sanity check
    assert grid_w * grid_h >= num, 'Number of rows and columns must be greater than the number of images!'
    # Get the grid
    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)
    # Paste each image in the grid
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


def get_edge_hint(
    image,    # source image
    version='cv2',   # diffusion version [1.5, 2.1]
    size=512,
    low_th=50,
    high_th=300,
):
    assert version in ('cv2', 'skimage'), f'<version> has to be cv2 or skimage and not {version}.'

    # if type(image).__name__ != 'PngImageFile':
    #     if image.dtype not in (np.uint8, torch.uint8):
    #         image =  image * 128 + 128

    image = np.array(image).astype(np.uint8)[..., :3]
    min_size = min(image.shape[:2])
    trafo_edges = tt.Compose([tt.ToPILImage(), tt.CenterCrop(min_size), tt.Resize(size)])

    im_edges = np.array(image).astype(np.uint8)
    hint_model = CannyDetector()

    detected_map = hint_model(im_edges, low_threshold=low_th, high_threshold=high_th)
    detected_map = np.array(trafo_edges(detected_map)).astype(np.uint8)
    hint = HWC3(detected_map)
    hint = hint / 255.0

    return hint


def get_image(path, size=512):
    image = Image.open(path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = tt.CenterCrop(size)(tt.Resize(size)(image))
    return image


def get_midas_depth(image, midas=MidasDetector(), max_resolution=512, size=512):
    image = np.array(image).astype(np.uint8)
    image = resize_image_pil(image, resolution=min(min(*image.shape[:-1]), max_resolution))
    depth, _ = midas(image)
    return depth


def get_canny_edges(image, size=512, low_th=50, high_th=200):
    image = np.array(image).astype(np.uint8)

    low_th = low_th or np.random.randint(50, 100)
    high_th = high_th or np.random.randint(200, 350)
    edges = CannyDetector()(image, low_th, high_th)  # original sized greyscale edges
    edges = edges / 255.
    return edges


def get_batch(ds_instance, num_samples):
    batch = dict()
    for key in ds_instance:
        instance = ds_instance[key]
        if isinstance(instance, str):
            batch[key] = [instance] * num_samples
        elif isinstance(instance, np.ndarray):
            batch[key] = instance[None, ...].repeat(num_samples, 0)
        elif isinstance(instance, torch.Tensor):
            batch[key] = instance[None, ...].repeat(num_samples, *[1] * len(instance.shape))

    return batch


@torch.no_grad()
def get_sdxl_sample(
        guidance,
        model,
        num_samples=2,
        seed=None,
        scale=9.5,
        eta=0.5,
        ddim_steps=25,
        prompt='',
        idx=None,
        control_scale=1.,
        shape=[4, 64, 64],
        n_prompt='longbody, lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality',
):
    ds = {
        'hint': guidance,
        'crop_coords_top_left': torch.tensor([0, 0]),
        'original_size_as_tuple': torch.tensor(guidance.shape[-2:]),
        'target_size_as_tuple': torch.tensor(guidance.shape[-2:]),
    }
    model.sampler.num_steps = ddim_steps
    model.sampler.s_noise = eta
    scale_schedule = lambda scale, sigma: scale  # independent of step
    model.sampler.guider.scale_schedule = partial(scale_schedule, scale)
    # print(f'[GENERATING SAMPLE SET WITH {ddim_steps} STEPS AND ETA {eta}]')

    if float(control_scale) != 1.0:
        model.model.scale_list *= control_scale
        print(f'[CONTROL CORRECTION OF {type(model).__name__} SCALED WITH {control_scale}]')

    control = torch.stack(
            [
                tt.ToTensor()(
                    # detected_map[..., None].repeat(3, 2)  # single map -> (512, 512, 3) 3channle->(512, 512, 9, 1)
                    guidance
                )
            ] * 2
        ).float().to('cuda')
    if len(guidance.shape) < 3:
        control = torch.stack([tt.ToTensor()(guidance[..., None].repeat(3, 2))] * num_samples).float().to('cuda')
    # control = torch.stack([tt.ToTensor()(ds['hint'][..., None].repeat(3, 2))] * num_samples).float().to('cuda')
    print(f">> input sdxl_hint_shape {control.shape}")
    sampling_kwargs = {'hint': control}

    if seed is None:
        seed = 1999158951
    seed_everything(seed)

    batch = get_batch(ds_instance=ds, num_samples=num_samples)
    batch['caption'] = [prompt or ds['caption']] * num_samples

    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to('cuda')

    batch_uc = copy.deepcopy(batch)
    batch_uc['caption'] = [n_prompt] * num_samples

    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=None
        if len(model.conditioner.embedders) > 0
        else [],
    )

    for k in c:
        if isinstance(c[k], torch.Tensor):
            c[k], uc[k] = map(lambda y: y[k].to('cuda'), (c, uc))

    with model.ema_scope("Plotting"):
        samples = model.sample(
            c, shape=shape, uc=uc, batch_size=num_samples, **sampling_kwargs
        )

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(
        x_samples, 'b c h w -> b h w c'
    ) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    x_samples
    control

    #  reset scales
    model.model.scale_list = model.model.scale_list * 0. + 1.

    return x_samples, control


def get_sd_sample(
        guidance,
        num_samples=3,
        model=None,
        seed=None,
        scale=9.5,
        eta=0.5,
        ddim_steps=25,
        prompt='',
        control_scale=1.,
        shape=[4, 64, 64],
        n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
):
    sampler = DDIMSampler

    if float(control_scale) != 1.0:
        model.control_model.scale_list *= control_scale
        print(f'[CONTROL CORRECTION OF {type(model).__name__} SCALED WITH {control_scale}]')
    ddim_sampler = sampler(model)

    detected_map = guidance
    if detected_map.shape[0] == 3:
        detected_map = einops.rearrange(detected_map, 'c h w -> h w c')
        if isinstance(detected_map, torch.Tensor):
            detected_map = np.array(detected_map)
    
    control = torch.stack(
            [
                tt.ToTensor()(
                    # detected_map[..., None].repeat(3, 2)  # single map -> (512, 512, 3) 3channle->(512, 512, 9, 1)
                    detected_map
                )
            ] * 2
        ).float().to('cuda')
    if len(detected_map.shape) < 3:
        control = torch.stack([tt.ToTensor()(guidance[..., None].repeat(3, 2))] * num_samples).float().to('cuda')

    if seed is None:
        seed = 1999158951
    seed_everything(seed)

    cond = {
        "c_concat": [control[..., :shape[-2] * 8, :shape[-1] * 8]],
        "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
    un_cond = {
        "c_concat": [control[..., :shape[-2] * 8, :shape[-1] * 8]],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

    samples, _ = ddim_sampler.sample(
        ddim_steps, num_samples,
        shape, cond, verbose=False, eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    #  reset scales
    model.control_model.scale_list = model.control_model.scale_list * 0. + 1.

    return x_samples, control
