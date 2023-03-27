base_model_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
                   'Counterfeit-V2.5.safetensors')
seed_matrix = [42, 233, 1919, 114514, 2300112, 29500]

# resolution = (768, 448)
# resolution = (1344, 768)
resolution = (768, 1344)
num_samples = 1
cfg_scale = 7.5
hire_strength = 0.7
hire_upscale = 1.8

do_hire = False

# NOTE: remove *outdoor*
prompt = ('((masterpiece,best quality)), girl, animal ears, rabbit, '
          'barefoot, dress, rabbit ears, short sleeves, '
          'looking at viewer, grass, short hair, smile, white hair, '
          'puffy sleeves, puffy short sleeves, bangs, on ground, '
          'full body, animal, white dress, sunlight, brown eyes, '
          'dappled sunlight, day, depth of field')
# prompt = ('rabbit ears, rabbit, animal ears, animal')
a_prompt = 'best quality, extremely detailed'
n_prompt = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')

avg_control = False
control = [
    dict(
        mode='depth',
        cond='./data/depth/dining.jpg',
        timesteps=[18, 19],
        hires_timesteps=[10, ..., 15],
        weight=0.75,
    ),
    dict(mode='pose', cond='./data/pose_image/MJ.png', weight=1)
]
