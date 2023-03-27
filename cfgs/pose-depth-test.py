
base_model_ckpt = '/nvme/xingzhening/diffusion-ckpts/Counterfeit-V2.5.safetensors'
seed_matrix = [42, 233, 1919, 114514, 2300112, 29500]

resolution = (512, 512)
num_samples = 4

prompt = 'single girl, black hair, short hair, glasses, sport wearing'
a_prompt = 'best quality, extremely detailed'
n_prompt = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')


control = [
    dict(mode='pose',
         layers=None,
         timesteps=[1, ..., 19],
         cond='./data/pose_image/MJ.png',
         weight=1),
    # dict(mode='pose', layer=None, timesteps=None, cond='', weight=1)
]
