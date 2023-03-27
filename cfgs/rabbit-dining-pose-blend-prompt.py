base_model_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
                   'Counterfeit-V2.5.safetensors')
# seed_matrix = [42, 233, 1919, 114514, 2300112, 29500]
seed_matrix = [233, 1919, 114514, 2300112, 29500]

# resolution = (768, 448)
resolution = (448, 768)
num_samples = 1
cfg_scale = 7.5
hire_strength = 0.7
hire_upscale = 1.8

# NOTE: remove *outdoor*
# prompt = ('((masterpiece,best quality)), girl, animal ears, rabbit, '
#           'barefoot, dress, rabbit ears, short sleeves, '
#           'looking at viewer, short hair, smile, white hair, '
#           'puffy sleeves, puffy short sleeves, bangs, '
#           'full body, animal, white dress, brown eyes, '
#           'depth of field')
prompt = ('girl, indoor, dining room, girl in front of the table')
a_prompt = 'best quality, extremely detailed'
n_prompt = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')

avg_control = True
control = [
    dict(
        mode='depth',
        cond='./data/depth/dining.jpg',
        timesteps=[1, ..., 17],
        hires_timesteps=[1, ..., 15],
        weight=0.5,
    ),
    dict(
        mode='pose',
        cond='./data/pose_image/single-ske.png',
        # timesteps=[10, ..., 19],
        # hires_timesteps=[1, ..., 17],
        weight=1,
        need_process=False)
]
