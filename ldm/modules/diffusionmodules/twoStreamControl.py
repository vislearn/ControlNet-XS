import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
)


from einops import rearrange
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
)
from ldm.util import exists


class TwoStreamControlNet(nn.Module):
    def __init__(
            self,
            image_size,
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
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            use_linear_in_transformer=False,
            infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
            infusion2base='add',            # how to infuse intermediate information into the base net? {'add', 'cat'}
            guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
            two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
            control_model_ratio=1.0,        # ratio of the control model size compared to the base model. [0, 1]
            learn_embedding=True,
    ):
        assert infusion2control in ('cat', 'add', None), f'infusion2control needs to be cat, add or None, but not {infusion2control}'
        assert infusion2base == 'add', f'infusion2base only defined for add, but not {infusion2base}'
        assert guiding in ('encoder', 'encoder_double', 'full'), f'guiding has to be encoder, encoder_double or full, but not {guiding}'
        assert two_stream_mode in ('cross', 'sequential'), f'two_stream_mode has to be either cross or sequential, but not {two_stream_mode}'

        super().__init__()

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

        ################# start control model variations #################
        base_model = UNetModel(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
        )  # initialise control model from base model
        self.control_model = ControlledUNetModel(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
            infusion2control=infusion2control,
            guiding=guiding, two_stream_mode=two_stream_mode, control_model_ratio=control_model_ratio,
        )  # initialise pretrained model

        # if guiding in ('encoder', 'encoder_double'):
        #     self.control_model.output_blocks = None
        ################# end control model variations #################

        self.enc_zero_convs_out = nn.ModuleList([])
        self.enc_zero_convs_in = nn.ModuleList([])

        self.middle_block_out = None
        self.middle_block_in = None

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

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list))

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, base_model, precomputed_hint=False, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if self.learn_embedding:
            emb = self.control_model.time_embed(t_emb)
        else:
            emb = base_model.time_embed(t_emb)

        if precomputed_hint:
            guided_hint = hint
        else:
            guided_hint = self.input_hint_block(hint, emb, context)
            # print('guided_hint_mean:', guided_hint.mean())

        h_ctr = h_base = x.type(base_model.dtype)
        hs_base = []
        hs_ctr = []
        it_enc_convs_in = iter(self.enc_zero_convs_in)
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_in = iter(self.dec_zero_convs_in)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list)

        ###################### Cross Control        ######################

        if self.two_stream_mode == 'cross':
            # input blocks (encoder)
            for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
                h_base = module_base(h_base, emb, context)
                h_ctr = module_ctr(h_ctr, emb, context)
                if guided_hint is not None:
                    h_ctr = h_ctr + guided_hint
                    guided_hint = None

                if self.guiding in ('encoder_double', 'full'):
                    if self.infusion2base == 'add':
                        h_base = h_base + next(it_enc_convs_out)(h_ctr, emb) * next(scales)
                    elif self.infusion2base == 'cat':
                        raise NotImplementedError()

                hs_base.append(h_base)
                hs_ctr.append(h_ctr)

                if self.infusion2control == 'add':
                    h_ctr = h_ctr + next(it_enc_convs_in)(h_base, emb)
                elif self.infusion2control == 'cat':
                    h_ctr = th.cat([h_ctr, next(it_enc_convs_in)(h_base, emb)], dim=1)

            # mid blocks (bottleneck)
            h_base = base_model.middle_block(h_base, emb, context)
            h_ctr = self.control_model.middle_block(h_ctr, emb, context)

            if self.infusion2base == 'add':
                h_base = h_base + self.middle_block_out(h_ctr, emb) * next(scales)
            elif self.infusion2base == 'cat':
                raise NotImplementedError()

            if self.guiding == 'full':
                if self.infusion2control == 'add':
                    h_ctr = h_ctr + self.middle_block_in(h_base, emb)
                elif self.infusion2control == 'cat':
                    h_ctr = th.cat([h_ctr, self.middle_block_in(h_base, emb)], dim=1)

            # output blocks (decoder)
            for module_base, module_ctr in zip(
                    base_model.output_blocks,
                    self.control_model.output_blocks if hasattr(
                    self.control_model, 'output_blocks') else [None] * len(base_model.output_blocks)
            ):

                if self.guiding != 'full':
                    if self.infusion2base == 'add':
                        h_base = h_base + next(it_dec_convs_out)(hs_ctr.pop(), emb) * next(scales)
                    elif self.infusion2base == 'cat':
                        raise NotImplementedError()

                h_base = th.cat([h_base, hs_base.pop()], dim=1)
                h_base = module_base(h_base, emb, context)

                if self.guiding == 'full':
                    h_ctr = th.cat([h_ctr, hs_ctr.pop()], dim=1)
                    h_ctr = module_ctr(h_ctr, emb, context)
                    if module_base != base_model.output_blocks[-1]:
                        if self.infusion2base == 'add':
                            h_base = h_base + next(it_dec_convs_out)(h_ctr, emb) * next(scales)
                        elif self.infusion2base == 'cat':
                            raise NotImplementedError()

                        if self.infusion2control == 'add':
                            h_ctr = h_ctr + next(it_dec_convs_in)(h_base, emb)
                        elif self.infusion2control == 'cat':
                            h_ctr = th.cat([h_ctr, next(it_dec_convs_in)(h_base, emb)], dim=1)

        return base_model.out(h_base)


class ControlledUNetModel(nn.Module):
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
        image_size,
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
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
        guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
        two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
        control_model_ratio=1.0,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        self.infusion2control = infusion2control
        infusion_factor = int(1 / control_model_ratio)
        cat_infusion = 1 if infusion2control == 'cat' else 0

        self.guiding = guiding
        self.two_stage_mode = two_stream_mode
        seq_factor = 1 if two_stream_mode == 'sequential' and infusion2control == 'cat' else 0

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
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
                    ResBlock(
                        ch * (1 + cat_infusion * infusion_factor),
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
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
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
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
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch * (1 + cat_infusion * infusion_factor),
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        if guiding == 'full':
            self.output_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(self.num_res_blocks[level] + 1):
                    ich = input_block_chans.pop()
                    layers = [
                        ResBlock(
                            ich + ch if level and i == num_res_blocks and two_stream_mode == 'sequential' else ich + ch * (1 + cat_infusion * infusion_factor),
                            time_embed_dim,
                            dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
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
                            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                        if exists(disable_self_attentions):
                            disabled_sa = disable_self_attentions[level]
                        else:
                            disabled_sa = False

                        if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                            layers.append(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads_upsample,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                ) if not use_spatial_transformer else SpatialTransformer(
                                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                    use_checkpoint=use_checkpoint
                                )
                            )
                    if level and i == self.num_res_blocks[level]:
                        out_ch = ch
                        layers.append(
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
    # if find_denominator(channels, 32) < 32:
    #     print(f'[USING GROUPNORM OVER LESS CHANNELS ({find_denominator(channels, 32)}) FOR {channels} CHANNELS]')
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
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
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

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
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
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialTransformertest(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
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
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
