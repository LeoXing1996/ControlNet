
base_model_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
                   'Counterfeit-V2.5.safetensors')
seed_matrix = [42, 233, 1919, 114514, 2300112, 29500]

resolution = (768, 448)
num_samples = 1
cfg_scale = 10
hire_strength = 0.7
hire_upscale = 1.8

prompt = ('((masterpiece,best quality)),1girl, solo, animal ears, rabbit, '
          'barefoot, knees up, dress, sitting, rabbit ears, short sleeves, '
          'looking at viewer, grass, short hair, smile, white hair, '
          'puffy sleeves, outdoors, puffy short sleeves, bangs, on ground, '
          'full body, animal, white dress, sunlight, brown eyes, '
          'dappled sunlight, day, depth of field')
a_prompt = 'best quality, extremely detailed'
n_prompt = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')


# control = [
#     dict(mode='pose',
#          layers=None,
#          timesteps=[1, ..., 19],
#          cond='./data/pose_image/MJ.png',
#          weight=0),
#     # dict(mode='pose', layer=None, timesteps=None, cond='', weight=1)
# ]
