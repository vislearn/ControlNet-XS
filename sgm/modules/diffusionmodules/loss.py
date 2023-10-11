from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig

from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        # print(f"conditioner, {type(conditioner)}, batch.keys() {batch.keys()}")       # sgm.modules.encoders.modules.GeneralConditioner
        additional_model_inputs = {   # 求两个dict中的交集
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        # cond crossattention like x or hint, vector, emb additional_model_inputs {} emptydc noise_storch.Size([2, 4, 64, 64]), sigmas torch.Size([2])
        # for k in cond.keys():   # text encoder 2, 77 , 2048  crossattention    2, 2048    vector
        #     print(">> cond_shape_dbg",cond[k].shape)
        print(f">> stabddiffuseion_denoiser cond {cond.keys()}, additional_model_inputs{additional_model_inputs}, noise_s{noise.shape},sigmas{sigmas.shape} ")
        hint = batch["image"]  # self.input_key  v_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # v_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # for k in batch.keys():
        #     if isinstance(batch[k], list):
        #         print(f">> cond_shape_dbg {k}", len( batch[k]), batch[k])
        #     else:
        #         print(f">> cond_shape_dbg {k}", batch[k].shape)

        # control = torch.stack(
        #         [
        #             tt.ToTensor()(
        #                 # detected_map[..., None].repeat(3, 2)  # single map -> (512, 512, 3) 3channle->(512, 512, 9, 1)
        #                 guidance
        #             )
        #         ] * 2
        #     ).float().to('cuda')
        model_output = denoiser(
            network, noised_input, sigmas, cond, hint, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
