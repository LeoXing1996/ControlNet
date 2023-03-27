base_model_ckpt = ('/nvme/xingzhening/diffusion-ckpts/'
                   'Counterfeit-V2.5.safetensors')
seed_matrix = [42]
ddim_steps = 20

resolution = (448, 768)
num_samples = 1
cfg_scale = 10
hire_strength = 0.7
hire_upscale = 1.8
do_hire = False
prompt = 'A painting of a girl eating a burger, burger, seagull'
# prompt = ('girl, animal ears, rabbit, '
#           'barefoot, dress, rabbit ears, short sleeves, '
#           'looking at viewer, grass, short hair, smile, white hair, '
#           'puffy sleeves, outdoors, puffy short sleeves, bangs, on ground, '
#           'full body, animal, white dress, sunlight, brown eyes, '
#           'dappled sunlight, day, depth of field')
a_prompt = 'best quality, extremely detailed'
n_prompt = ('longbody, lowres, bad anatomy, bad hands, missing fingers, '
            'extra digit, fewer digits, cropped, worst quality, low quality')

save_control_output = True
save_attn = True
control = [
    dict(
        mode='pose',
        cond='./data/pose_image/single-ske.png',
        weight=1,
        need_process=False)
]
