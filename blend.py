from PIL import Image
import hashlib
from copy import deepcopy
import os
import os.path as osp
import shutil

from datetime import datetime
import cv2
import einops
import numpy as np
from mmengine import Config
from argparse import ArgumentParser
from blend_controlnet import BlendLoop
from mmengine.runner import set_random_seed
from hooks import vis_attention


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('cfgs')
    parser.add_argument('--name', default=None, help='Name of the experiment.')
    parser.add_argument('--work-dir', default='work_dir')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfgs)
    seeds = cfg.get('seed_matrix', None)
    if not isinstance(seeds, list):
        seeds = [seeds]

    base_model_cfgs = cfg.get('base_model_cfgs', './cfgs/sd-v-1-5.yml')
    base_model_ckpt = cfg.get('base_model_ckpt',
                              './models/v1-5-pruned-emaonly.safetensors')

    prompt = cfg['prompt']
    a_prompt, n_prompt = cfg['a_prompt'], cfg['n_prompt']
    control_cfgs = cfg.get('control', None)
    no_control = control_cfgs is None

    change_base_model = cfg.get('change_base_model', True)
    avg_control = cfg.get('avg_control', False)
    save_attn = cfg.get('save_attn', False)
    save_control_output = cfg.get('save_control_output', False)

    blendloop = BlendLoop(base_model_cfgs, base_model_ckpt, control_cfgs,
                          change_base_model, avg_control, save_attn,
                          save_control_output)

    # make log dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    if args.name is None:
        name = cfg.filename.split('/')[-1].split('.')[0]
    else:
        name = args.name
    output_root = osp.join(args.work_dir, f'{name}_{timestamp}')
    os.makedirs(output_root, exist_ok=True)
    shutil.copy2(args.cfgs, osp.join(output_root, cfg.filename.split('/')[-1]))

    change_base_model = cfg.get('change_base_model', True)
    num_samples = cfg.get('num_samples', 1)
    ddim_steps = cfg.get('ddim_steps', 20)
    scale = cfg.get('cfg_scale', 7.5)
    do_hire = cfg.get('do_hire', True)
    hire_strength = cfg.get('hire_strength', 0.7)
    hire_upscale = cfg.get('hire_upscale', 1.0)
    resolution = cfg.get('resolution', (512, 512))
    noise_inject_scale = cfg.get('noise_injection_scale', 0.0)

    random_sample = cfg.get('random_sample', False)

    for seed in seeds:
        if seed is not None:
            set_random_seed(seed)

        # if random_sample:
        #     random_outputs = blendloop.random_sample(resolution, num_samples)

        #     for idx, out in enumerate(random_outputs):
        #         save_name = f'KL_{idx}.png' if seed is None \
        #             else f'KL_seed{seed}_{idx}.png'
        #         save_path = os.path.join(output_root, save_name)
        #         Image.fromarray(out).save(save_path)
        # continue  # NOTE: just for quick run

        outputs = blendloop.sample(prompt,
                                   a_prompt,
                                   n_prompt,
                                   scale=scale,
                                   image_resolution=resolution,
                                   ddim_steps=ddim_steps,
                                   num_samples=num_samples,
                                   do_hire=do_hire,
                                   hire_upscale=hire_upscale,
                                   hire_strength=hire_strength,
                                   no_control=no_control,
                                   noise_inject_scale=noise_inject_scale)

        if isinstance(outputs, dict):
            samples, samples_hire = outputs['sample'], outputs['hire_sample']
            for idx, (out, out_hire) in enumerate(zip(samples, samples_hire)):
                save_name = f'{idx}.png' if seed is None \
                    else f'seed{seed}_{idx}.png'
                save_name_hire = f'{idx}.png' if seed is None \
                    else f'seed{seed}_{idx}_hire.png'
                save_path = os.path.join(output_root, save_name)
                save_path_hire = os.path.join(output_root, save_name_hire)
                Image.fromarray(out).save(save_path)
                Image.fromarray(out_hire).save(save_path_hire)

        else:
            for idx, out in enumerate(outputs):
                save_name = f'{idx}.png' if seed is None \
                    else f'seed{seed}_{idx}.png'
                save_path = os.path.join(output_root, save_name)
                Image.fromarray(out).save(save_path)

        # save attention map
        tokenizer = blendloop.model.cond_stage_model.tokenizer
        if blendloop.attn_store is not None:
            vis_attention(prompt,
                          a_prompt,
                          n_prompt,
                          blendloop.attn_store,
                          resolution,
                          tokenizer,
                          save_dir=output_root,
                          seed=seed)
        if blendloop.control_attn_stores is not None:
            for idx, attn_store in enumerate(blendloop.control_attn_stores):
                sub_path = os.path.join(output_root, f'{idx}_attn')
                os.makedirs(sub_path, exist_ok=True)
                vis_attention(prompt,
                              a_prompt,
                              n_prompt,
                              attn_store,
                              resolution,
                              tokenizer,
                              save_dir=sub_path,
                              seed=seed)

        # save control output
        if save_control_output:
            vis_res = (resolution[0] // 4, resolution[1] // 4)
            blendloop.controlnets.vis_output(output_root, vis_res=vis_res)

    if blendloop.controlnets is not None:
        for idx, detect_map in enumerate(
                blendloop.controlnets.detected_map_list):
            Image.fromarray(detect_map).save(
                os.path.join(output_root, f'{idx}_detect_map.png'))


if __name__ == '__main__':
    main()
