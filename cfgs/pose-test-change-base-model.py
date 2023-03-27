base_model_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
                   'Counterfeit-V2.5_pruned.safetensors')
change_base_model = True

seed_matrix = [42, 233, 1919, 114514, 2300112, 29500]

resolution = (512, 512)
num_samples = 1
ddim_steps = 20
# cfg_scale = 10

# prompt = 'single girl, black hair, short hair, glasses, sport wearing'
prompt = ('girl, barefoot, knees up, dress, rabbit ears, short sleeves, '
          'looking at viewer, grass, short hair, smile, white hair, '
          'puffy sleeves, outdoors, puffy short sleeves, bangs, on ground, '
          'full body, animal, white dress, sunlight, brown eyes, '
          'dappled sunlight, day, depth of field')
a_prompt = 'best quality, extremely detailed'
n_prompt = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')

control = [
    # dict(mode='pose',
    #      layers=None,
    #      timesteps=None,
    #      cond='./data/pose_image/MJ.png',
    #      weight=1),
    dict(mode='pose',
         layers=None,
         timesteps=None,
         cond='./data/pose_image/multi-ske.png',
         need_process=False,
         weight=2)
]
