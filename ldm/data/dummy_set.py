import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from ldm.data.util import resize_image_pil

import torchvision.transforms as tt
import torch
import einops


from annotator.canny import CannyDetector
from annotator.midas import MidasDetector


class DummyBase(Dataset):
    """
    Dummy dataset for testing purposes. This dataset is a simple dataset that loads images from a folder.
    Currently, Canny Edges, Midas Depth or the original image can be used as control hints.
    It is advised to
        - load hints from a folder for hints with intense compute like MiDaS
        - to add captions to prevent semantic bias
    """
    def __init__(
        self,
        data_root,                          # path to image folder
        size=512,                           # size of the image
        interpolation="bilinear",           # interpolation method
        center_crop=True,                   # whether to center crop or random crop
        control_mode='canny',               # control mode
        np_format=True,                     # whether to return numpy ordering [H W C] or torch ordering [C H W]
        original_size_as_tuple=False,       # whether to return original size as tuple - only needed for sdxl training
        crop_coords_top_left=False,         # whether to return crop coordinates top left - only needed for sdxl training
        target_size_as_tuple=False,         # whether to return target size as tuple - only needed for sdxl training
        ):

        assert control_mode in ('canny', 'midas', 'image'), f'[<control_mode> {control_mode} not implemented in this dummy set]'

        self.np_format = np_format
        self.original_size_as_tuple = original_size_as_tuple
        self.crop_coords_top_left = crop_coords_top_left
        self.target_size_as_tuple = target_size_as_tuple
        # self.key_phrases = None
        self.control_mode = control_mode
        self.data_root = data_root
        self.center_crop = center_crop

        self.control_mode = control_mode
        if control_mode == 'canny':
            self.hint_model = CannyDetector()
        elif control_mode == 'midas':
            self.hint_model = MidasDetector()
        elif control_mode == 'image':
            self.hint_model = None
        else:
            raise NotImplementedError()

        self.image_paths = os.listdir(data_root)
        self.image_paths = [path for path in self.image_paths if ".png" in path or 'jpg' in path]

        # this dummy dataset has no captions, we advice to use captions to prevent sementic bias (if wanted)
        self.captions = ['' for _ in self.image_paths]

        self._length = len(self.image_paths)

        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "caption": self.captions
        }

        size = None if size is not None and size <= 0 else size
        self.size = size

        ########## PILLOW RESIZE VERSION ##########
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": tt.InterpolationMode.NEAREST,
                "bilinear": tt.InterpolationMode.BILINEAR,
                "bicubic": tt.InterpolationMode.BICUBIC,
                "lanczos": tt.InterpolationMode.LANCZOS}[self.interpolation]
            self.image_rescaler = tt.Resize(
                size=self.size,
                interpolation=self.interpolation,
                antialias=True
                )

            if self.center_crop:
                self.cropper = tt.CenterCrop(size=self.size)
            else:
                self.cropper = tt.RandomCrop(size=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i, low_th=None, high_th=None):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        # load imge
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # get control image
        if self.control_mode == 'canny':
            low_th = low_th or np.random.randint(50, 100)
            high_th = high_th or np.random.randint(200, 350)
            detected_map = self.hint_model(image, low_th, high_th)  # original sized greyscale edges

        elif self.control_mode == 'midas':
            max_resolution = 1024
            image = resize_image_pil(image, resolution=min(min(*image.shape[:-1]), max_resolution))
            detected_map, _ = self.hint_model(image)

        elif self.control_mode == 'image':
            detected_map = image

        else:
            raise NotImplementedError()

        image_hint = np.concatenate([
                        image,
                        detected_map if self.control_mode == 'image' else detected_map[..., None]
                        ], axis=2)
        image_hint = tt.ToTensor()(image_hint)

        if self.size is not None:
            image_hint = self.image_rescaler(image_hint)
        orig_size = image_hint.shape[-2:]

        if self.size is not None:
            processed = self.preprocessor(image_hint)
        else:
            processed = image_hint
        target_size = processed.shape[-2:]

        processed = np.array(einops.rearrange(processed, 'c h w -> h w c'))

        example["image"] = (processed[..., :3]*2 - 1.0).astype(np.float32)
        if self.control_mode == 'canny':
            example['hint'] = processed[..., 3:].repeat(3, 2)
        elif self.control_mode == 'midas':
            example['hint'] = processed[..., 3:].repeat(3, 2) * 2. - 1.
        elif self.control_mode == 'image':
            example['hint'] = processed[..., 3:] * 255
        else:
            raise NotImplementedError()

        if self.crop_coords_top_left:
            example['crop_coords_top_left'] = torch.tensor([0, 0])
        if self.original_size_as_tuple:
            example['original_size_as_tuple'] = torch.tensor(orig_size)
        if self.target_size_as_tuple:
            example['target_size_as_tuple'] = torch.tensor(target_size)

        if not self.np_format:
            example['image'] = einops.rearrange(example['image'], 'h w c -> c h w')
            example['hint'] = einops.rearrange(example['hint'], 'h w c -> c h w')

        return example


if __name__ == "__main__":
    dset = DummyBase(size=256)
    ex = dset[0]
    for k in ["image", "caption"
              ]:
        print(type(ex[k]))
        try:
            print(ex[k].shape)
        except:
            print(ex[k])
