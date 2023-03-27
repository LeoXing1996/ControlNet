base_model_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
                   'Counterfeit-V2.5.safetensors')
# base_model_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
#                    'anything-v3-fp32-pruned.safetensors')
seed_matrix = [42, 233, 1919, 114514, 2300112, 29500]

resolution = (448, 448)
num_samples = 1
cfg_scale = 10
hire_strength = 0.7
hire_upscale = 1.8

prompt = 'girl, head shot'
a_prompt = 'best quality, extremely detailed'
n_prompt = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')
random_sample = True
