import torch
import torch as th
import torch.nn as nn

from functools import partial
from typing import Iterable

from sgm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
)

from einops import rearrange

from ...modules.attention import BasicTransformerBlock
from ...modules.diffusionmodules.openaimodel import (
    UNetModel,
    Timestep,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
)

from ...util import default, exists

from annotator.midas import MidasDetector
from torchvision import transforms as tt
import numpy as np


class PseudoModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return 0


class TwoStreamControlNet(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            adm_in_channels=None,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            spatial_transformer_attn_type="softmax",
            use_linear_in_transformer=False,
            num_classes=None,
            infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
            infusion2base='add',            # how to infuse intermediate information into the base net? {'add', 'cat'}
            guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
            two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
            control_model_ratio=1.0,        # ratio of the control model size compared to the base model. [0, 1]
            base_model=None,
            learn_embedding=False,
            control_mode='canny',
            prune_until=None,
            fixed=False,
    ):
        assert infusion2control in ('cat', 'add', None), f'infusion2control needs to be cat, add or None, but not {infusion2control}'
        assert infusion2base == 'add', f'infusion2base only defined for add, but not {infusion2base}'
        assert guiding in ('encoder', 'encoder_double', 'full'), f'guiding has to be encoder, encoder_double or full, but not {guiding}'
        assert two_stream_mode in ('cross', 'sequential'), f'two_stream_mode has to be either cross or sequential, but not {two_stream_mode}'

        super().__init__()

        self.control_mode = control_mode
        self.learn_embedding = learn_embedding
        self.infusion2control = infusion2control
        self.infusion2base = infusion2base
        self.in_ch_factor = 1 if infusion2control == 'add' else 2
        self.guiding = guiding
        self.two_stream_mode = two_stream_mode
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.no_control = False
        self.control_scale = 1.0
        self.prune_until = prune_until
        self.fixed = fixed

        if control_mode == 'midas':
            self.hint_model = MidasDetector()
        else:
            self.hint_model = None

        ################# start control model variations #################
        if base_model is None:
            base_model = UNetModel(
                adm_in_channels=adm_in_channels, num_classes=num_classes, use_checkpoint=use_checkpoint,
                in_channels=in_channels, out_channels=out_channels, model_channels=model_channels,
                attention_resolutions=attention_resolutions, num_res_blocks=num_res_blocks,
                channel_mult=channel_mult, num_head_channels=num_head_channels, use_spatial_transformer=use_spatial_transformer,
                use_linear_in_transformer=use_linear_in_transformer, transformer_depth=transformer_depth,
                context_dim=context_dim, spatial_transformer_attn_type=spatial_transformer_attn_type,
                legacy=legacy, dropout=dropout,
                conv_resample=conv_resample, dims=dims, use_fp16=use_fp16, num_heads=num_heads,
                num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
                resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
                n_embed=n_embed,
                # disable_self_attentions=disable_self_attentions,
                # num_attention_blocks=num_attention_blocks,
                # disable_middle_self_attn=disable_middle_self_attn,
            )  # initialise control model from base model
        self.control_model = ControlledXLUNetModelFixed(
            adm_in_channels=adm_in_channels, num_classes=num_classes, use_checkpoint=use_checkpoint,
            in_channels=in_channels, out_channels=out_channels, model_channels=model_channels,
            attention_resolutions=attention_resolutions, num_res_blocks=num_res_blocks,
            channel_mult=channel_mult, num_head_channels=num_head_channels, use_spatial_transformer=use_spatial_transformer,
            use_linear_in_transformer=use_linear_in_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, spatial_transformer_attn_type=spatial_transformer_attn_type,
            legacy=legacy, dropout=dropout,
            conv_resample=conv_resample, dims=dims, use_fp16=use_fp16, num_heads=num_heads,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            n_embed=n_embed, fixed=fixed,
            # disable_self_attentions=disable_self_attentions,
            # num_attention_blocks=num_attention_blocks,
            # disable_middle_self_attn=disable_middle_self_attn,
            infusion2control=infusion2control,
            guiding=guiding, two_stream_mode=two_stream_mode, control_model_ratio=control_model_ratio,
        )  # initialise pretrained model

        self.diffusion_model = base_model

        ################# end control model variations #################

        self.enc_zero_convs_out = nn.ModuleList([])
        self.enc_zero_convs_in = nn.ModuleList([])

        self.middle_block_out = nn.ModuleList([])
        self.middle_block_in = nn.ModuleList([])

        self.dec_zero_convs_out = nn.ModuleList([])
        self.dec_zero_convs_in = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        ################# Gather Channel Sizes #################
        for module in self.control_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

        ch_inout_ctr['mid'].append((self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
        ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

        if guiding not in ('encoder', 'encoder_double'):
            for module in self.control_model.output_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_ctr['dec'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_ctr['dec'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[-1], Upsample):
                    ch_inout_ctr['dec'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        ################# Build zero convolutions #################
        if two_stream_mode == 'cross':
            ################# cross infusion #################
            # infusion2control
            # add
            if infusion2control == 'add':
                for i in range(len(ch_inout_base['enc'])):
                    self.enc_zero_convs_in.append(self.make_zero_conv(
                        in_channels=ch_inout_base['enc'][i][1], out_channels=ch_inout_ctr['enc'][i][1])
                    )

                if guiding == 'full':
                    self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_ctr['mid'][-1][1])
                    for i in range(len(ch_inout_base['dec']) - 1):
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_inout_base['dec'][i][1], out_channels=ch_inout_ctr['dec'][i][1])
                        )

                # cat - processing full concatenation (all output layers are concatenated without "slimming")
            if infusion2control == 'cat':
                for ch_io_base in ch_inout_base['enc']:
                    self.enc_zero_convs_in.append(self.make_zero_conv(
                        in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                    )

                if guiding == 'full':
                    self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_base['mid'][-1][1])
                    for ch_io_base in ch_inout_base['dec']:
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                        )

                # None - no changes

            # infusion2base - consider all three guidings
                # add
            if infusion2base == 'add':
                self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

                if guiding in ('encoder', 'encoder_double'):
                    self.dec_zero_convs_out.append(
                        self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
                    )
                    for i in range(1, len(ch_inout_ctr['enc'])):
                        self.dec_zero_convs_out.append(
                            self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
                        )
                if guiding in ('encoder_double', 'full'):
                    for i in range(len(ch_inout_ctr['enc'])):
                        self.enc_zero_convs_out.append(self.make_zero_conv(
                            in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
                        )

                if guiding == 'full':
                    for i in range(len(ch_inout_ctr['dec'])):
                        self.dec_zero_convs_out.append(self.make_zero_conv(
                            in_channels=ch_inout_ctr['dec'][i][1], out_channels=ch_inout_base['dec'][i][1])
                        )

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, int(model_channels * control_model_ratio), 3, padding=1))
        )

        if self.prune_until is not None:
            self.prune_last(prune_until)
        elif not self.learn_embedding:
            if not self.learn_embedding:
                self.control_model.time_embed = PseudoModule()
        self.control_model.label_emb = PseudoModule()

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list))

    def prune_last(self, end_layer):
        for n in range(len(self.control_model.input_blocks)):
            if n >= end_layer:
                self.control_model.input_blocks[n] = PseudoModule()

        self.control_model.middle_block = PseudoModule()
        if not self.learn_embedding:
            self.control_model.time_embed = PseudoModule()
        self.control_model.label_emb = PseudoModule()

        self.enc_zero_convs_in = nn.ModuleList(self.enc_zero_convs_in[:end_layer] + [PseudoModule()] * len(self.enc_zero_convs_in[end_layer:]))
        self.enc_zero_convs_out = nn.ModuleList(self.enc_zero_convs_out[:end_layer] + [PseudoModule()] * len(self.enc_zero_convs_out[end_layer:]))
        self.middle_block_in = self.middle_block_out = PseudoModule()
        self.dec_zero_convs_in = nn.ModuleList([PseudoModule()] * len(self.dec_zero_convs_in))
        self.dec_zero_convs_out = nn.ModuleList([PseudoModule()] * len(self.dec_zero_convs_out))

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        )

    def infuse(self, stream, infusion, mlp, variant, emb, scale=1.0):
        if variant == 'add':
            stream = stream + mlp(infusion, emb) * scale
        elif variant == 'cat':
            stream = torch.cat([stream, mlp(infusion, emb) * scale], dim=1)

        return stream

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, hint: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        # in case of classifier free guidance:
        if x.size(0) // 2 == hint.size(0):
            hint = torch.cat([hint, hint], dim=0)
        return self.forward_(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            hint=hint,
            base_model=self.diffusion_model,
            compute_hint=True if self.control_mode == 'midas' else False,
            **kwargs,
        )

    def forward_(
        self, x, hint, timesteps, context,
        base_model=None, y=None, precomputed_hint=False,
        no_control=False, compute_hint=False, **kwargs
    ):

        if base_model is None:
            base_model = self.diffusion_model

        if no_control or self.no_control:
            return base_model(x=x, timesteps=timesteps, context=context, y=y, **kwargs)

        if compute_hint:
            hints = []
            for inp in hint:
                hint_processed, _ = self.hint_model(np.array(inp.cpu().permute(1, 2, 0)))
                hints.append(tt.ToTensor()(hint_processed[..., None].repeat(3, 2)))
            hint_processed = torch.stack(hints).to(x.device)
            hint = hint_processed.to(memory_format=torch.contiguous_format).float()

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if self.learn_embedding:
            emb = self.control_model.time_embed(t_emb) * self.control_scale ** 0.3 + base_model.time_embed(t_emb) * (1 - self.control_scale ** 0.3)
        else:
            emb = base_model.time_embed(t_emb)

        if y is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + base_model.label_emb(y)

        if precomputed_hint:
            guided_hint = hint
        else:
            guided_hint = self.input_hint_block(hint, emb, context)

        h_ctr = h_base = x
        hs_base = []
        hs_ctr = []
        it_enc_convs_in = iter(self.enc_zero_convs_in)
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_in = iter(self.dec_zero_convs_in)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list)

        ###################### Cross Control        ######################
        if self.two_stream_mode == 'cross':
            #  input blocks (encoder)
            for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
                h_base = module_base(h_base, emb, context)
                if "PseudoModule" not in str(type(module_ctr)):
                    h_ctr = module_ctr(h_ctr, emb, context)
                if guided_hint is not None:
                    h_ctr = h_ctr + guided_hint
                    guided_hint = None

                if self.guiding in ('encoder_double', 'full'):
                    h_base = self.infuse(h_base, h_ctr, next(it_enc_convs_out), self.infusion2base, emb, scale=next(scales))

                hs_base.append(h_base)
                hs_ctr.append(h_ctr)

                if "PseudoModule" not in str(type(module_ctr)):
                    h_ctr = self.infuse(h_ctr, h_base, next(it_enc_convs_in), self.infusion2control, emb)

            # mid blocks (bottleneck)
            h_base = base_model.middle_block(h_base, emb, context)
            h_ctr = self.control_model.middle_block(h_ctr, emb, context)

            h_base = self.infuse(h_base, h_ctr, self.middle_block_out, self.infusion2base, emb, scale=next(scales))

            if self.guiding == 'full':
                h_ctr = self.infuse(h_ctr, h_base, self.middle_block_in, self.infusion2control, emb)

            # output blocks (decoder)
            for module_base, module_ctr in zip(
                    base_model.output_blocks,
                    self.control_model.output_blocks if hasattr(
                        self.control_model, 'output_blocks') else [None] * len(base_model.output_blocks)
            ):

                if self.guiding != 'full':
                    h_base = self.infuse(h_base, hs_ctr.pop(), next(it_dec_convs_out), self.infusion2base, emb, scale=next(scales))

                h_base = th.cat([h_base, hs_base.pop()], dim=1)
                h_base = module_base(h_base, emb, context)

                ##### Quick and dirty way of fixing "full" with not applying corrections to the last layer #####
                if self.guiding == 'full':
                    h_ctr = th.cat([h_ctr, hs_ctr.pop()], dim=1)
                    h_ctr = module_ctr(h_ctr, emb, context)
                    if module_base != base_model.output_blocks[-1]:
                        h_base = self.infuse(h_base, h_ctr, next(it_dec_convs_out), self.infusion2base, emb, scale=next(scales))
                        h_ctr = self.infuse(h_ctr, h_base, next(it_dec_convs_in), self.infusion2control, emb)

        h_base = h_base.type(x.dtype)
        return base_model.out(h_base)


class ControlledXLUNetModelFixed(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        spatial_transformer_attn_type="softmax",
        adm_in_channels=None,
        use_fairscale_checkpoint=False,
        offload_to_cpu=False,
        transformer_depth_middle=None,
        infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
        guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
        two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
        control_model_ratio=1.0,
        fixed=False,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.infusion2control = infusion2control
        infusion_factor = 1 / control_model_ratio
        if not fixed:
            infusion_factor = int(infusion_factor)
        cat_infusion = 1 if infusion2control == 'cat' else 0

        self.guiding = guiding
        self.two_stage_mode = two_stream_mode
        seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        if use_fp16:
            print("WARNING: use_fp16 was dropped and has no effect anymore.")
        # self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        assert use_fairscale_checkpoint != use_checkpoint or not (
            use_checkpoint or use_fairscale_checkpoint
        )

        self.use_fairscale_checkpoint = False
        checkpoint_wrapper_fn = (
            partial(checkpoint_wrapper, offload_to_cpu=offload_to_cpu)
            if self.use_fairscale_checkpoint
            else lambda x: x
        )

        time_embed_dim = model_channels * 4
        self.time_embed = checkpoint_wrapper_fn(
            nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        )

        ch_orig = model_channels
        model_channels_orig = model_channels
        model_channels = max(1, int(model_channels * control_model_ratio))
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = checkpoint_wrapper_fn(
                    nn.Sequential(
                        Timestep(model_channels),
                        nn.Sequential(
                            linear(model_channels, time_embed_dim),
                            nn.SiLU(),
                            linear(time_embed_dim, time_embed_dim),
                        ),
                    )
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    checkpoint_wrapper_fn(
                        ResBlock(
                            # int(ch * (1 + cat_infusion * infusion_factor)),
                            int(ch) + cat_infusion * ch_orig,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )
                ]
                ch = mult * model_channels
                ch_orig = mult * model_channels_orig
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        # custom code for smaller models - start
                        num_head_channels = find_denominator(ch, self.num_head_channels)
                        # custom code for smaller models - end
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            checkpoint_wrapper_fn(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                )
                            )
                            if not use_spatial_transformer
                            else checkpoint_wrapper_fn(
                                SpatialTransformer(
                                    ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth[level],
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_transformer,
                                    attn_type=spatial_transformer_attn_type,
                                    use_checkpoint=use_checkpoint,
                                )
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                out_ch_orig = ch_orig
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        checkpoint_wrapper_fn(
                            ResBlock(
                                int(ch) + (cat_infusion - seq_factor) * ch_orig,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                        )
                        if resblock_updown
                        else Downsample(
                            int(ch) + (cat_infusion - seq_factor) * ch_orig,
                            conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                ch_orig = out_ch_orig
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            checkpoint_wrapper_fn(
                ResBlock(
                    int(ch) + cat_infusion * ch_orig,
                    time_embed_dim,
                    dropout,
                    out_channels=ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ),
            checkpoint_wrapper_fn(
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
            )
            if not use_spatial_transformer
            else checkpoint_wrapper_fn(
                SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth_middle,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    attn_type=spatial_transformer_attn_type,
                    use_checkpoint=use_checkpoint,
                )
            ),
            checkpoint_wrapper_fn(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ),
        )
        self._feature_size += ch

        if guiding == 'full':
            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(self.num_res_blocks[level] + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ich + ch if level and i == num_res_blocks and two_stream_mode == 'sequential' else int(ich + ch * (1 + cat_infusion * infusion_factor)),
                                time_embed_dim,
                                dropout,
                                out_channels=model_channels * mult,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                            )
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            # num_heads = 1
                            dim_head = (
                                ch // num_heads
                                if use_spatial_transformer
                                else num_head_channels
                            )
                        if exists(disable_self_attentions):
                            disabled_sa = disable_self_attentions[level]
                        else:
                            disabled_sa = False

                        if (
                            not exists(num_attention_blocks)
                            or i < num_attention_blocks[level]
                        ):
                            layers.append(
                                checkpoint_wrapper_fn(
                                    AttentionBlock(
                                        ch,
                                        use_checkpoint=use_checkpoint,
                                        num_heads=num_heads_upsample,
                                        num_head_channels=dim_head,
                                        use_new_attention_order=use_new_attention_order,
                                    )
                                )
                                if not use_spatial_transformer
                                else checkpoint_wrapper_fn(
                                    SpatialTransformer(
                                        ch,
                                        num_heads,
                                        dim_head,
                                        depth=transformer_depth[level],
                                        context_dim=context_dim,
                                        disable_self_attn=disabled_sa,
                                        use_linear=use_linear_in_transformer,
                                        attn_type=spatial_transformer_attn_type,
                                        use_checkpoint=use_checkpoint,
                                    )
                                )
                            )
                    if level and i == self.num_res_blocks[level]:
                        out_ch = ch
                        layers.append(
                            checkpoint_wrapper_fn(
                                ResBlock(
                                    ch,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=out_ch,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm,
                                    up=True,
                                )
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch


class ControlledXLUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        spatial_transformer_attn_type="softmax",
        adm_in_channels=None,
        use_fairscale_checkpoint=False,
        offload_to_cpu=False,
        transformer_depth_middle=None,
        infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
        guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
        two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
        control_model_ratio=1.0,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.infusion2control = infusion2control
        infusion_factor = int(1 / control_model_ratio)
        cat_infusion = 1 if infusion2control == 'cat' else 0

        self.guiding = guiding
        self.two_stage_mode = two_stream_mode
        seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        if use_fp16:
            print("WARNING: use_fp16 was dropped and has no effect anymore.")
        # self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        assert use_fairscale_checkpoint != use_checkpoint or not (
            use_checkpoint or use_fairscale_checkpoint
        )

        self.use_fairscale_checkpoint = False
        checkpoint_wrapper_fn = (
            partial(checkpoint_wrapper, offload_to_cpu=offload_to_cpu)
            if self.use_fairscale_checkpoint
            else lambda x: x
        )

        time_embed_dim = model_channels * 4
        self.time_embed = checkpoint_wrapper_fn(
            nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        )

        model_channels = int(model_channels * control_model_ratio)
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = checkpoint_wrapper_fn(
                    nn.Sequential(
                        Timestep(model_channels),
                        nn.Sequential(
                            linear(model_channels, time_embed_dim),
                            nn.SiLU(),
                            linear(time_embed_dim, time_embed_dim),
                        ),
                    )
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    checkpoint_wrapper_fn(
                        ResBlock(
                            ch * (1 + cat_infusion * infusion_factor),
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_head_channels = find_denominator(ch, self.num_head_channels)
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            checkpoint_wrapper_fn(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                )
                            )
                            if not use_spatial_transformer
                            else checkpoint_wrapper_fn(
                                SpatialTransformer(
                                    ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth[level],
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_transformer,
                                    attn_type=spatial_transformer_attn_type,
                                    use_checkpoint=use_checkpoint,
                                )
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch * (1 + (cat_infusion - seq_factor) * infusion_factor),
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                        )
                        if resblock_updown
                        else Downsample(
                            ch * (1 + (cat_infusion - seq_factor) * infusion_factor),
                            conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            checkpoint_wrapper_fn(
                ResBlock(
                    ch * (1 + cat_infusion * infusion_factor),
                    time_embed_dim,
                    dropout,
                    out_channels=ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ),
            checkpoint_wrapper_fn(
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
            )
            if not use_spatial_transformer
            else checkpoint_wrapper_fn(
                SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth_middle,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    attn_type=spatial_transformer_attn_type,
                    use_checkpoint=use_checkpoint,
                )
            ),
            checkpoint_wrapper_fn(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ),
        )
        self._feature_size += ch

        if guiding == 'full':
            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(self.num_res_blocks[level] + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ich + ch if level and i == num_res_blocks and two_stream_mode == 'sequential' else ich + ch * (1 + cat_infusion * infusion_factor),
                                time_embed_dim,
                                dropout,
                                out_channels=model_channels * mult,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                            )
                        )
                    ]
                    ch = model_channels * mult
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            dim_head = (
                                ch // num_heads
                                if use_spatial_transformer
                                else num_head_channels
                            )
                        if exists(disable_self_attentions):
                            disabled_sa = disable_self_attentions[level]
                        else:
                            disabled_sa = False

                        if (
                            not exists(num_attention_blocks)
                            or i < num_attention_blocks[level]
                        ):
                            layers.append(
                                checkpoint_wrapper_fn(
                                    AttentionBlock(
                                        ch,
                                        use_checkpoint=use_checkpoint,
                                        num_heads=num_heads_upsample,
                                        num_head_channels=dim_head,
                                        use_new_attention_order=use_new_attention_order,
                                    )
                                )
                                if not use_spatial_transformer
                                else checkpoint_wrapper_fn(
                                    SpatialTransformer(
                                        ch,
                                        num_heads,
                                        dim_head,
                                        depth=transformer_depth[level],
                                        context_dim=context_dim,
                                        disable_self_attn=disabled_sa,
                                        use_linear=use_linear_in_transformer,
                                        attn_type=spatial_transformer_attn_type,
                                        use_checkpoint=use_checkpoint,
                                    )
                                )
                            )
                    if level and i == self.num_res_blocks[level]:
                        out_ch = ch
                        layers.append(
                            checkpoint_wrapper_fn(
                                ResBlock(
                                    ch,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=out_ch,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm,
                                    up=True,
                                )
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                    self._feature_size += ch


def find_denominator(number, start):
    if start >= number:
        return number
    while (start != 0):
        residual = number % start
        if residual == 0:
            return start
        start -= 1


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if find_denominator(channels, 32) < 32:
        print(f'[USING GROUPNORM OVER LESS CHANNELS ({find_denominator(channels, 32)}) FOR {channels} CHANNELS]')
    return GroupNorm_leq32(find_denominator(channels, 32), channels)


class GroupNorm_leq32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = th.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        attn_type="softmax",
        use_checkpoint=True,
        # sdp_backend=SDPBackend.FLASH_ATTENTION
        sdp_backend=None,
    ):
        super().__init__()
        print(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )
        from omegaconf import ListConfig

        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in