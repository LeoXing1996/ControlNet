import abc
import os.path as osp
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import xformers
import xformers.ops
from einops import rearrange, repeat
from PIL import Image
from torch import einsum

from ldm.modules.attention import (BasicTransformerBlock, CrossAttention,
                                   MemoryEfficientCrossAttention,
                                   SpatialTransformer, default, exists)


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        # self.cur_att_layer += 1
        # if self.cur_att_layer == self.num_att_layers:
        #     self.cur_att_layer = 0
        #     self.cur_step += 1
        #     self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": []
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # if attn.shape[1] <= 16**2:  # avoid memory overhead
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn.clone().cpu())
        return attn

    def between_steps(self):
        # NOTE: different from the official code, we call this function
        # manually during the sampling
        if len(self.attention_store) == 0:
            # self.attention_store = self.step_store
            self.attention_store = {
                k: [[attn] for attn in attns]
                for k, attns in self.step_store.items()
            }
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    # self.attention_store[key][i] += self.step_store[key][i]
                    # NOTE: we store the full attention map
                    # NOTE: this operation took a lot of memory,
                    # maybe save them is a pickle
                    self.attention_store[key][i].append(
                        self.step_store[key][i])
        self.step_store = self.get_empty_store()
        self.cur_step += 1

    def get_average_attention(self):
        # NOTE: this function may not work.
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def aggregate_attention(prompts, attention_store: AttentionStore, res: int,
                        from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res**2
    for location in from_where:
        for item in attention_maps[
                f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res,
                                          item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def text_under_image(image: np.ndarray,
                     text: str,
                     text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 0.8, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8)
              for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones(
        (h * num_rows + offset * (num_rows - 1), w * num_cols + offset *
         (num_cols - 1), 3),
        dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset):i * (h + offset) + h:, j * (w + offset):j *
                   (w + offset) + w] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img


def vis_single_attn(prompts: List, attn_map: torch.Tensor, res: Tuple[int,
                                                                      int]):
    vis_list = []
    for i, prompt in enumerate(prompts):
        image = attn_map[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res[1], res[0])))
        image = text_under_image(image, prompt)
        vis_list.append(image)
    vis_res = np.stack(vis_list, axis=0)
    return view_images(vis_res, num_rows=4, offset_ratio=0.02)
    # return Image.fromarray(vis_res)


def vis_attention(prompts,
                  a_prompts,
                  n_prompts,
                  attention_store: AttentionStore,
                  res,
                  tokenizer,
                  save_dir,
                  seed,
                  keys=None,
                  timesteps='avg'):
    """
    timesteps/head_idx: None to vis all, 'average' to average
    """
    cond_tokens = prompts + ',' + a_prompts
    cond_p_id = tokenizer.encode(cond_tokens)
    cond_p = [tokenizer.decode([p]) for p in cond_p_id]

    uncond_tokens = n_prompts
    uncond_p_id = tokenizer.encode(uncond_tokens)
    uncond_p = [tokenizer.decode([p]) for p in uncond_p_id]

    vis_res = (res[0] // 4, res[1] // 4)
    if keys is None:
        keys = attention_store.attention_store.keys()

    for k in keys:
        # NOTE: ignore SA
        if 'self' in k:
            continue

        attn_maps = attention_store.attention_store[k]

        if not attn_maps:
            # NOTE: this attention map is empty
            continue

        num_layers = len(attn_maps)
        num_timesteps = len(attn_maps[0])
        if timesteps is None or timesteps == 'avg':
            vis_timesteps = [idx for idx in range(num_timesteps)]
        else:
            vis_timesteps = timesteps

        for layer_attn_maps in attn_maps:
            b, hw, n_prom = layer_attn_maps[0].shape
            scale_factor = (res[0] * res[1]) // hw
            if scale_factor * hw != res[0] * res[1]:
                print(f"Warning: {res} is not divisible by {hw}")
            scale_factor = int(np.sqrt(scale_factor))
            # cond_attn_map, uncond_attn_map = layer_attn_maps.split(b//2)

            cond_vis_attn_map, uncond_vis_attn_map = [], []
            for t in vis_timesteps:
                # t = len(num_timesteps) - t  # NOTE: reverse
                vis_attn_map = layer_attn_maps[t]
                cond_attn_map, uncond_attn_map = vis_attn_map.split(b // 2)
                cond_vis_attn_map.append(cond_attn_map)
                uncond_vis_attn_map.append(uncond_attn_map)

            if timesteps == 'avg':
                cond_vis_attn_map = sum(cond_vis_attn_map) / num_timesteps
                uncond_vis_attn_map = sum(uncond_vis_attn_map) / num_timesteps

                cond_vis_attn_map = cond_vis_attn_map.reshape(
                    -1, res[0] // scale_factor, res[1] // scale_factor, 77)
                uncond_vis_attn_map = uncond_vis_attn_map.reshape(
                    -1, res[0] // scale_factor, res[1] // scale_factor, 77)

                cond_vis_attn_map = cond_vis_attn_map.sum(dim=0) / b
                uncond_vis_attn_map = uncond_vis_attn_map.sum(dim=0) / b

                cond_vis_res = vis_single_attn(cond_p, cond_vis_attn_map,
                                               vis_res)
                uncond_vis_res = vis_single_attn(uncond_p, uncond_vis_attn_map,
                                                 vis_res)

                name_prefix = f'seed{seed}_{k}_scale{scale_factor}_tavg'
                cond_vis_res.save(osp.join(save_dir,
                                           name_prefix + '_cond.png'))
                uncond_vis_res.save(
                    osp.join(save_dir, name_prefix + '_uncond.png'))

            else:
                for t in vis_timesteps:
                    index = num_timesteps - t - 1
                    cond_attn_map = cond_vis_attn_map[index]
                    uncond_attn_map = uncond_vis_attn_map[index]

                    cond_attn_map = cond_attn_map.reshape(
                        -1, res[0] // scale_factor, res[1] // scale_factor, 77)
                    uncond_attn_map = uncond_attn_map.reshape(
                        -1, res[0] // scale_factor, res[1] // scale_factor, 77)

                    cond_attn_map = cond_attn_map.sum(dim=0) / b
                    uncond_attn_map = uncond_attn_map.sum(dim=0) / b

                    cond_vis_res = vis_single_attn(cond_p, cond_attn_map,
                                                   vis_res)
                    uncond_vis_res = vis_single_attn(uncond_p, uncond_attn_map,
                                                     vis_res)

                    name_prefix = f'seed{seed}_{k}_scale{scale_factor}_t{t}'
                    cond_vis_res.save(
                        osp.join(save_dir, name_prefix + '_cond.png'))
                    uncond_vis_res.save(
                        osp.join(save_dir, name_prefix + '_uncond.png'))


def hook_attn(attn_block, controller, place_in_unet):

    def ca_forward_xfor(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            is_cross = context is not None

            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[
                    1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                        b * self.heads, t.shape[1], self.dim_head).contiguous(
                        ),
                (q, k, v),
            )

            # q/k/v: [b * heads, seq_len, dim_head] -->
            # * SA: q/k/v: [2*heads, H*W, dim_head]
            # * CA: q: [2*heads, H*W, dim_head], k/v: [2*heads, 77, dim_head]
            out = xformers.ops.memory_efficient_attention(q,
                                                          k,
                                                          v,
                                                          attn_bias=None,
                                                          op=self.attention_op)

            if exists(mask):
                raise NotImplementedError
            out = (out.unsqueeze(0).reshape(
                b, self.heads, out.shape[1],
                self.dim_head).permute(0, 2, 1,
                                       3).reshape(b, out.shape[1],
                                                  self.heads * self.dim_head))

            # NOTE: calculate attention map manually,
            # but this is not efficient
            # >>> code for save attention,
            # * SA: [2*heads, H*W, H*W]
            # * CA: [2*heads, H*W, 77]
            with torch.no_grad():
                # NOTE: this is very important
                scale = self.dim_head**-0.5
                attn_map = einsum('b i d, b j d -> b i j', q, k) * scale
                attn_map = torch.softmax(attn_map, dim=-1)
                controller(attn_map, is_cross, place_in_unet)
            # <<< code for save attention,

            return self.to_out(out)

        return forward

    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):

            is_cross = context is not None
            h = self.heads

            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(
                lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                (q, k, v))

            # force cast to fp32 to avoid overflowing
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            del q, k

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            sim = sim.softmax(dim=-1)

            sim = controller(sim, is_cross, place_in_unet)
            out = einsum('b i j, b j d -> b i d', sim, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        return forward

    if isinstance(attn_block, MemoryEfficientCrossAttention):
        attn_block.forward = ca_forward_xfor(attn_block, place_in_unet)
    elif isinstance(attn_block, CrossAttention):
        attn_block.forward = ca_forward(attn_block, place_in_unet)


def replace_ca(model, index, controller, place_in_unet):
    for m in model.children():
        if isinstance(m, SpatialTransformer):
            trans_blocks = m.transformer_blocks
            for block in trans_blocks:
                block: BasicTransformerBlock
                attn1 = block.attn1
                attn2 = block.attn2
                hook_attn(attn1, controller, place_in_unet)
                hook_attn(attn2, controller, place_in_unet)
                index += 1
        else:
            index = replace_ca(m, index, controller, place_in_unet)
    return index


def registry_baseUnet(model, controller):
    unet = model.model.diffusion_model
    mid_blocks = unet.middle_block
    inp_blocks = unet.input_blocks
    out_blocks = unet.output_blocks

    index = 0
    index = replace_ca(inp_blocks, index, controller, 'down')
    index = replace_ca(mid_blocks, index, controller, 'mid')
    index = replace_ca(out_blocks, index, controller, 'up')
    controller.num_att_layers = index


def registry_controlUnet(model, controller):
    unet = model.model
    mid_blocks = unet.middle_block
    inp_blocks = unet.input_blocks

    index = 0
    index = replace_ca(inp_blocks, index, controller, 'down')
    index = replace_ca(mid_blocks, index, controller, 'mid')
    controller.num_att_layers = index
