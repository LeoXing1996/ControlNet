import os.path as osp
import os
import hashlib
from copy import deepcopy
from collections import defaultdict

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from PIL import Image
from safetensors import safe_open
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Resize)

from annotator.midas import apply_midas
from annotator.openpose import apply_openpose
from annotator.util import HWC3, resize_image
from cldm.model import create_model, load_state_dict
from hooks import AttentionStore, registry_baseUnet, registry_controlUnet
# from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddim_ours import DDIMSampler
from share import *


def parse_list(var):
    if var is None:
        return None

    if not isinstance(var, list):
        return [var]

    if Ellipsis in var:
        var_list = []
        for idx, v in enumerate(var):
            if v == Ellipsis:
                if idx == 0:
                    assert var[1] != Ellipsis
                    var_list += [i for i in range(0, var[1] + 1)]
                else:
                    # not the list element
                    assert idx != len(var) - 1
                    # not
                    assert var[idx - 1] != Ellipsis and var[idx +
                                                            1] != Ellipsis
                    var_list += [
                        i for i in range(var[idx - 1], var[idx + 1] + 1)
                    ]

        return var_list
    else:
        return var


class BlendLoop():

    def __init__(self, base_model_cfgs, checkpoint, control_cfgs,
                 change_base_model, avg_control_weight, store_attention,
                 save_control_output):
        self.model = self.load_base_model(base_model_cfgs, checkpoint)

        if control_cfgs is not None:
            self.controlnets = BlendedControlNet(control_cfgs,
                                                 avg_control_weight,
                                                 save_control_output)
            if change_base_model:
                self.controlnets.change_base_model(self.model, basemodel=None)
        else:
            self.controlnets = None

        self.model.cuda()

        if store_attention:
            self.attn_store = AttentionStore()
            registry_baseUnet(self.model, self.attn_store)
            # registry store for controlnets
            if self.controlnets is not None:
                num_controls = len(self.controlnets.controlnet_list)
                self.control_attn_stores = [
                    AttentionStore() for _ in range(num_controls)
                ]
                for store, controlnet in zip(self.control_attn_stores,
                                             self.controlnets.controlnet_list):
                    registry_controlUnet(controlnet, store)
        else:
            self.attn_store = self.control_attn_stores = None

        self.sampler = self.load_sampler(self.model, self.controlnets,
                                         self.attn_store,
                                         self.control_attn_stores)

    def load_base_model(self, base_model_cfgs, checkpoint):
        # NOTE: we do not support load ema
        model = create_model(base_model_cfgs)
        state_dict = dict()
        with safe_open(checkpoint, framework="pt", device="cpu") as file:
            for k in file.keys():
                state_dict[k] = file.get_tensor(k)
        model.load_state_dict(state_dict, strict=False)
        return model

    def load_sampler(self, model, controlnets, attn_store, control_stores):
        sampler = DDIMSampler(model,
                              controlnets=controlnets,
                              attn_store=attn_store,
                              control_stores=control_stores)
        return sampler

    def random_sample(self, image_resolution, num_samples=1):
        # NOTE: sample images by AutoEncoder-KL
        # NOTE: this is not f***king work, because the latent space for
        # AutoEncoderKL is not gaussian distribution
        H, W = image_resolution
        shape = (4, H // 8, W // 8)

        latent = torch.randn(num_samples, *shape).cuda()
        x_samples = self.model.decode_first_stage(latent)
        x_samples_np = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples_np[i] for i in range(num_samples)]
        return results

    def sample(self,
               prompt,
               a_prompt,
               n_prompt,
               scale,
               image_resolution,
               ddim_steps,
               num_samples=1,
               do_hire=True,
               hire_upscale=1,
               hire_strength=0.7,
               no_control=False,
               noise_inject_scale=0):

        # ddim_steps = 20
        # num_samples = 1
        eta = 0.0
        H, W = image_resolution
        shape = (4, H // 8, W // 8)

        cond_prompt = [prompt + ', ' + a_prompt]
        # [num_samples, 77, 768] for sd15
        cond_c_crossattn = self.model.get_learned_conditioning(cond_prompt *
                                                               num_samples)
        uncond_c_crossattn = self.model.get_learned_conditioning([n_prompt] *
                                                                 num_samples)

        cond = {
            # "c_concat": [control],
            "c_crossattn": [cond_c_crossattn],
        }
        un_cond = {
            # "c_concat": [control],
            "c_crossattn": [uncond_c_crossattn]
        }

        # set const noise
        if noise_inject_scale != 0:
            generator = torch.Generator()
            generator.manual_seed(99889)
            noise_const = torch.randn(num_samples, *shape,
                                      generator=generator).cuda()
            noise_const = noise_const * noise_inject_scale
            self.sampler.set_noise(noise_const)

        if self.controlnets is not None:
            self.controlnets.set_cond_tensor((H, W))
        samples, intermediates = self.sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            no_control=no_control)
        x_samples = self.model.decode_first_stage(samples)
        x_samples_np = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples_np[i] for i in range(num_samples)]

        # do hire / img2img
        if do_hire:
            steps, t_enc = self.get_hire_tstart(ddim_steps, hire_strength)
            init_latent = samples

            if hire_upscale != 1:
                h, w = init_latent.shape[2:]
                new_h = int(h * hire_upscale // 8) * 8
                new_w = int(w * hire_upscale // 8) * 8
                init_latent = F.interpolate(init_latent,
                                            size=(new_h, new_w),
                                            mode='bicubic',
                                            antialias=True)
            # set const noise
            if noise_inject_scale != 0:
                generator = torch.Generator()
                generator.manual_seed(99889)
                noise_const = torch.randn(num_samples,
                                          4,
                                          new_h,
                                          new_w,
                                          generator=generator).cuda()
                noise_const = noise_const * noise_inject_scale
                self.sampler.set_noise(noise_const)

            if self.controlnets is not None:
                self.controlnets.set_cond_tensor((new_h * 8, new_w * 8))

            self.sampler.make_schedule(ddim_num_steps=steps,
                                       is_hires=True,
                                       t_enc=t_enc)
            x1 = self.sampler.stochastic_encode(
                init_latent,
                torch.tensor([t_enc] * int(init_latent.shape[0])).to(
                    init_latent.device),
                noise=torch.randn_like(init_latent))

            samples_hire = self.sampler.decode(
                x1,
                cond,
                t_enc,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                no_control=no_control)
            x_samples_hires = self.model.decode_first_stage(samples_hire)
            x_samples_hires = (
                einops.rearrange(x_samples_hires, 'b c h w -> b h w c') * 127.5
                + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results_hire = [x_samples_hires[i] for i in range(num_samples)]

            return dict(sample=results, hire_sample=results_hire)

        return results

    def get_hire_tstart(self, num_inference_steps, strength):
        steps = num_inference_steps
        t_enc = int(min(strength, 0.999) * steps)
        return steps, t_enc

    def inpainting_sample(self,
                          samples,
                          mask,
                          prompt,
                          a_prompt,
                          n_prompt,
                          scale,
                          image_resolution,
                          ddim_steps,
                          num_samples=1,
                          hire_strength=0.7,
                          no_control=False,
                          ):

        eta = 0.0
        H, W = image_resolution
        shape = (4, H // 8, W // 8)

        cond_prompt = [prompt + ', ' + a_prompt]
        # [num_samples, 77, 768] for sd15
        cond_c_crossattn = self.model.get_learned_conditioning(cond_prompt *
                                                               num_samples)
        uncond_c_crossattn = self.model.get_learned_conditioning([n_prompt] *
                                                                 num_samples)

        cond = {
            # "c_concat": [control],
            "c_crossattn": [cond_c_crossattn],
        }
        un_cond = {
            # "c_concat": [control],
            "c_crossattn": [uncond_c_crossattn]
        }

        if self.controlnets is not None:
            self.controlnets.set_cond_tensor((H, W))
        # x_samples = self.model.decode_first_stage(samples)
        # x_samples_np = (
        #     einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
        #     127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        # results = [x_samples_np[i] for i in range(num_samples)]

        # do hire / img2img
        steps, t_enc = self.get_hire_tstart(ddim_steps, hire_strength)
        init_latent_dist = self.model.encode_first_stage(samples)
        init_latent = self.model.get_first_stage_encoding(
            init_latent_dist).detach()

        # add mask
        random_init_latent = self.model.get_first_stage_encoding()
        init_latent_masked = init_latent * mask + (1 - mask)
        # init_latent = samples

        # if hire_upscale != 1:
        #     h, w = init_latent.shape[2:]
        #     new_h = int(h * hire_upscale // 8) * 8
        #     new_w = int(w * hire_upscale // 8) * 8
        #     init_latent = F.interpolate(init_latent,
        #                                 size=(new_h, new_w),
        #                                 mode='bicubic',
        #                                 antialias=True)
        # if self.controlnets is not None:
        #     self.controlnets.set_cond_tensor((new_h * 8, new_w * 8))

        self.sampler.make_schedule(ddim_num_steps=steps,
                                   is_hires=True,
                                   t_enc=t_enc)
        x1 = self.sampler.stochastic_encode(
            init_latent,
            torch.tensor([t_enc] * int(init_latent.shape[0])).to(
                init_latent.device),
            noise=torch.randn_like(init_latent))

        samples_hire = self.sampler.decode(x1,
                                           cond,
                                           t_enc,
                                           unconditional_guidance_scale=scale,
                                           unconditional_conditioning=un_cond,
                                           no_control=no_control)
        x_samples_hires = self.model.decode_first_stage(samples_hire)
        x_samples_hires = (
            einops.rearrange(x_samples_hires, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results_hire = [x_samples_hires[i] for i in range(num_samples)]

        return results_hire


class BlendedControlNet(nn.Module):
    """
    [
        dict(
            mode='depth',
            ckpt=xxx,
            weight=0.5,
            layer=[xxx],
            need_process=True),
        ...
    ]
    """

    def __init__(self, control_cfgs, avg_control=False, save_output=False):
        super().__init__()

        if not isinstance(control_cfgs, list):
            control_cfgs = [control_cfgs]

        for cfg in control_cfgs:
            cfg['save_output'] = save_output

        # self.image_resolution = 512
        self.detected_resolution = 512
        self.controlnet_list, self.cond_list = self.load_controlnets(
            control_cfgs)
        self.controlnet_list.cuda()
        self.avg_control = avg_control

        self.cond_resolution = None
        self.cond_ten_list, self.detected_map_list = None, None

    def change_base_model(self, targetmodel, basemodel=None):
        target_state_dict = targetmodel.state_dict()
        target_state_dict = {
            k[len('model.diffusion_model.'):]: v.clone()
            for k, v in target_state_dict.items() if 'diffusion_model' in k
        }

        if basemodel is None:
            base_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
                         'v1-5-pruned-emaonly.safetensors')
            base_state_dict = dict()
            with safe_open(base_ckpt, framework="pt", device="cpu") as file:
                for k in file.keys():
                    base_state_dict[k] = file.get_tensor(k)
        else:
            base_state_dict = basemodel.state_dict()
        base_state_dict = {
            k[len('model.diffusion_model.'):]: v.clone()
            for k, v in base_state_dict.items() if 'diffusion_model' in k
        }

        for controlnet in self.controlnet_list:
            controlnet.change_base_model(target_state_dict, base_state_dict)

    def set_timesteps(self, timesteps, is_hires=False, t_enc=None):
        for controlnet in self.controlnet_list:
            controlnet.set_timesteps(timesteps, is_hires, t_enc)

    def load_controlnets(self, controlnets):
        controlnet_list = nn.ModuleList()
        cond_inp_list = []
        for controlnet in controlnets:
            cfg_ = deepcopy(controlnet)
            mode = cfg_.pop('mode').capitalize()
            cond = cfg_.pop('cond')

            type_ = f'{mode}ControlNet'
            cfg_['type'] = type_
            controlnet_obj = MODELS.build(cfg_)
            controlnet_list.append(controlnet_obj)
            cond_inp_list.append(cond)

        return controlnet_list, cond_inp_list

    def set_cond_tensor(self, resolution):

        if (self.cond_resolution == resolution
                and self.cond_ten_list is not None):
            return

        detected_map_list, cond_tensor_list = [], []
        for cond, controlnet in zip(self.cond_list, self.controlnet_list):

            image = np.array(Image.open(cond))
            cond_tensor, detected_map = controlnet.process(
                image, resolution, self.detected_resolution)

            detected_map_list.append(detected_map)
            cond_tensor_list.append(cond_tensor.cuda())

        self.cond_resolution = resolution
        self.cond_ten_list = cond_tensor_list
        self.detected_map_list = detected_map_list

    def forward(self, x, t, context):

        control_list = []
        control_weight_list = []
        for cond_ten, controlnet in zip(self.cond_ten_list,
                                        self.controlnet_list):
            control = controlnet(cond_ten, x, t, context)
            control_list.append(control)

            # print('<<<<<<<')
            # for c in control:
            #     print(c.mean().item(), c.std().item())
            # print('>>>>>>>')

            weights = []
            for c in control:
                if isinstance(c, torch.Tensor):
                    if (c == 0).all():
                        weights.append(0)
                    else:
                        weights.append(controlnet.weight)
                else:
                    if c == 0:
                        weights.append(0)
                    else:
                        weights.append(controlnet.weight)

            control_weight_list.append(weights)

        # ipdb.set_trace()
        # apply layer cond blend
        num_layers = len(control_list[0])
        assert all([len(control) == num_layers for control in control_list])
        control_list_blended = []
        for idx in range(num_layers):
            control_blended = sum([control[idx] for control in control_list])
            if self.avg_control:
                weight = sum([ws[idx] for ws in control_weight_list])
                if weight != 0:
                    control_blended = control_blended / weight
            control_list_blended.append(control_blended)

        # TODO: support this feat
        # if self.add_const_noise:
        #     # NOTE: add const nosie to the middle layer of unet
        #     control_list_blended[-1] += self.const_noise
        # for c in control_list_blended:
        #     print(c.shape)
        return control_list_blended

    def vis_output(self, save_dir, vis_res):
        for idx, controlnet in enumerate(self.controlnet_list):
            sub_save_dir = osp.join(save_dir, f'{idx}_control')
            os.makedirs(sub_save_dir, exist_ok=True)
            controlnet.vis_output(sub_save_dir, vis_res)


class ControlNetWithPreprocessor(nn.Module):

    def __init__(self,
                 ckpt,
                 weight=1,
                 layers=None,
                 timesteps=None,
                 hires_timesteps=None,
                 need_process=True,
                 save_output=False,
                 cfg='./cfgs/control_model.yml'):
        super().__init__()

        self.model = create_model(cfg).cpu()
        self.load_controlnet(ckpt)

        self.weight = weight
        self.timestep_index = parse_list(timesteps)
        self.hires_time_index = parse_list(hires_timesteps)

        self.layers = parse_list(layers)

        self.need_process = need_process

        self.init_args = dict(ckpt=ckpt, weight=weight, layers=layers)

        # self.output_store = defaultdict(list)
        self.save_output = save_output
        self.output_store = dict()

    def set_timesteps(self, timesteps, is_hires=False, t_enc=None):
        if is_hires:

            if self.hires_time_index is None:
                self.timesteps = [int(t) for t in timesteps]
                return

            if self.hires_time_index == [-1]:
                self.timesteps = [-1]  # NOTE: disable for all timesteps
                return

            new_length = t_enc
            new_indexs = [
                int(t / 20 * new_length) for t in self.hires_time_index
            ]
            sample_timesteps = [int(timesteps[i]) for i in new_indexs]
            self.timesteps = sample_timesteps
        else:

            if self.timestep_index is None:
                self.timesteps = [int(t) for t in timesteps]
                return

            if self.timestep_index == [-1]:
                self.timesteps = [-1]  # NOTE: disable for all timesteps
                return

            sample_timesteps = [int(timesteps[i]) for i in self.timestep_index]
            self.timesteps = sample_timesteps

    def load_controlnet(self, ckpt):
        state_dict = load_state_dict(ckpt)
        # NOTE: hard code here
        controlnet_state_dict = {
            k[len('control_model.'):]: v
            for k, v in state_dict.items() if 'control' in k
        }
        self.model.load_state_dict(controlnet_state_dict)

    def change_base_model(self, target_model, base_model):

        changed_state_dict = {}
        for k, v in self.model.state_dict().items():
            if k in base_model:
                base_model_weight = base_model[k]
                target_model_weight = target_model[k]
                delta_weight = target_model_weight - base_model_weight
                new_control_weight = v.cpu() + delta_weight.cpu()
                changed_state_dict[k] = new_control_weight
                # print(f'Convert key: {k}')
            else:
                changed_state_dict[k] = v

        self.model.load_state_dict(changed_state_dict)

        print(f'Change BaseModel for {self.__class__.__name__}')

    def resize(self, control, detected_map, img_resolution):
        """Resize the control to fit the target resolution
        """
        h, w = img_resolution
        transform = Compose([
            Resize(h if h < w else w, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size=(h, w))
        ])

        control = transform(control)
        detected_map = transform(detected_map)
        return control, detected_map

    @torch.inference_mode()
    def forward(self, cond, x, timestep, context):

        if timestep[0] not in self.timesteps:
            print(f'skip: {timestep[0]}')
            return [0 for _ in range(13)]

        cond_txt = torch.cat(context['c_crossattn'], 1)
        # get control for each layers
        control_list = self.model(x=x,
                                  hint=cond,
                                  timesteps=timestep,
                                  context=cond_txt)
        control_list = [control * self.weight for control in control_list]

        if self.layers is not None:
            for idx in self.layers:
                control_list[idx] = 0

        if self.save_output:
            if self.layers is None:
                layers = [i for i in range(len(control_list))]
            else:
                layers = self.layers
            for idx in layers:
                prefix = f't{timestep[0]}_l{idx}'
                self.output_store[prefix] = control_list[idx].detach().cpu()

        return control_list

    def vis_output(self, save_dir, vis_res=None):
        if not self.save_output:
            return

        vis_terms = self.output_store.keys()
        layers = [term.split('_')[-1] for term in vis_terms]
        layers = list(set(layers))

        mode = 'avg'
        cond_t, uncond_t = defaultdict(list), defaultdict(list)
        # TODO: support t-avg
        for name in self.output_store:
            output = self.output_store[name]
            if isinstance(output, int):
                continue
            b = output.shape[0]
            # NOTE: we assume bz == 2 (cond + uncond) * 1
            cond, uncond = output.split(b // 2)

            if mode == 'avg':
                cond_t[name.split('_')[-1]].append(cond)
                uncond_t[name.split('_')[-1]].append(uncond)
            else:
                cond_vis = self.vis_single_feature(cond.clone(), vis_res)
                uncond_vis = self.vis_single_feature(uncond.clone(), vis_res)

                cond_vis.save(osp.join(save_dir, f'cond_{name}.png'))
                uncond_vis.save(osp.join(save_dir, f'uncond_{name}.png'))

        if mode == 'avg':
            for name, feats in cond_t.items():
                feats_sum = torch.cat(feats).sum(0) / len(feats)
                feats_vis = self.vis_single_feature(feats_sum.clone(), vis_res)
                feats_vis.save(osp.join(save_dir, f'cond_avg_{name}.png'))
                feats_vis = self.vis_single_feature(feats_sum.clone(), vis_res,
                                                    'bicubic')
                feats_vis.save(
                    osp.join(save_dir, f'cond_avg_{name}_smooth.png'))
            for name, feats in uncond_t.items():
                feats_sum = torch.cat(feats).sum(0) / len(feats)
                feats_vis = self.vis_single_feature(feats_sum.clone(), vis_res)
                feats_vis.save(osp.join(save_dir, f'uncond_avg_{name}.png'))
                feats_vis = self.vis_single_feature(feats_sum.clone(), vis_res,
                                                    'bicubic')
                feats_vis.save(
                    osp.join(save_dir, f'uncond_avg_{name}_smooth.png'))

    @staticmethod
    def vis_single_feature(feat, vis_res, vis_mode='nearest'):
        """feat: [C, H, W]"""
        # TODO: support static anaylis
        feat = F.interpolate(feat[None, ...], size=vis_res, mode=vis_mode)
        feat = feat[0].permute(1, 2, 0).mean(-1)
        feat = feat / feat.max() * 255.
        feat = feat.cpu().numpy().astype(np.uint8)
        feat = Image.fromarray(feat)
        return feat


@MODELS.register_module()
class PoseControlNet(ControlNetWithPreprocessor):

    def __init__(self, ckpt=None, *args, **kwargs):

        if ckpt is None:
            ckpt = './models/control_sd15_openpose.pth'
        super().__init__(ckpt=ckpt, *args, **kwargs)

    @torch.inference_mode()
    def process(self,
                input_image,
                image_resolution,
                detect_resolution,
                num_samples=1):

        input_image = HWC3(input_image)
        if not self.need_process:
            detected_map = input_image
        else:
            detected_map, _ = apply_openpose(
                resize_image(input_image, detect_resolution))

        detected_map = HWC3(detected_map)

        cond = torch.from_numpy(detected_map.copy()).cuda().float() / 255.
        cond = torch.stack([cond for _ in range(num_samples)])
        cond = einops.rearrange(cond, 'b h w c -> b c h w')
        detected_map = einops.rearrange(torch.from_numpy(detected_map),
                                        'h w c -> c h w')
        cond, detected_map = self.resize(cond, detected_map, image_resolution)
        detected_map = einops.rearrange(
            detected_map, 'c h w -> h w c').numpy().astype(np.uint8)
        return cond, detected_map


@MODELS.register_module()
class DepthControlNet(ControlNetWithPreprocessor):

    def __init__(self, ckpt=None, *args, **kwargs):
        if ckpt is None:
            ckpt = './models/control_sd15_depth.pth'
        super().__init__(ckpt, *args, **kwargs)

    @torch.inference_mode()
    def process(self,
                input_image,
                image_resolution,
                detect_resolution,
                num_samples=1):

        input_image = HWC3(input_image)
        if not self.need_process:
            detected_map = input_image
        else:
            detected_map, _ = apply_midas(
                resize_image(input_image, detect_resolution))

        detected_map = HWC3(detected_map)

        cond = torch.from_numpy(detected_map.copy()).cuda().float() / 255.
        cond = torch.stack([cond for _ in range(num_samples)])
        cond = einops.rearrange(cond, 'b h w c -> b c h w')
        detected_map = einops.rearrange(torch.from_numpy(detected_map),
                                        'h w c -> c h w')
        cond, detected_map = self.resize(cond, detected_map, image_resolution)
        # NOTE: just for debug
        # scale = 0.1
        # cond, detected_map = cond * scale, detected_map * scale
        # cond = 1 - cond
        # detected_map = 255 - detected_map

        detected_map = einops.rearrange(
            detected_map, 'c h w -> h w c').numpy().astype(np.uint8)
        return cond, detected_map


@MODELS.register_module()
class EmptyControlNet(ControlNetWithPreprocessor):

    def __init__(self, ckpt=None, *args, **kwargs):
        if ckpt is None:
            ckpt = './models/control_sd15_scribble.pth'
        super().__init__(ckpt, *args, **kwargs)

    @torch.inference_mode()
    def process(self,
                input_image,
                image_resolution,
                detect_resolution,
                num_samples=1):

        input_image = HWC3(input_image)
        if not self.need_process:
            detected_map = input_image
        else:
            detected_map, _ = apply_midas(
                resize_image(input_image, detect_resolution))

        detected_map = HWC3(detected_map)

        cond = torch.from_numpy(detected_map.copy()).cuda().float() / 255.
        cond = torch.stack([cond for _ in range(num_samples)])
        cond = einops.rearrange(cond, 'b h w c -> b c h w')
        detected_map = einops.rearrange(torch.from_numpy(detected_map),
                                        'h w c -> c h w')
        cond, detected_map = self.resize(cond, detected_map, image_resolution)

        detected_map = einops.rearrange(
            detected_map, 'c h w -> h w c').numpy().astype(np.uint8)

        # NOTE: return an empty cond tensor
        return torch.zeros_like(cond), detected_map
