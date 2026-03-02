'''
Follow the colab version script, try to debug this pkg on windows...
'''



#%%

from huggingface_hub import login
from PIL import Image
from diffusers import DiffusionPipeline

token = '此处输入huggingface的api'
login(token=token)


style = 'a pencil sketch of'
obj1 = 'Albert  Einstein'
obj2 = 'Marilyn Monroe'

prompt_1 = style+obj1
prompt_2 = style+obj2

# #%%
# ! pip install -q \
#   diffusers\
#   transformers\
#   safetensors \
#   sentencepiece \
#   accelerate\
#   bitsandbytes \
#   einops \
#   mediapy \
#   accelerate

# !pip install -q git+https://github.com/dangeng/visual_anagrams.git

# #%% version checking 
# import transformers
# import diffusers
# import accelerate

# print(f"Transformers version: {transformers.__version__}")
# print(f"Diffusers version: {diffusers.__version__}")

#%%
import gc
import mediapy as mp
from PIL import Image
import torch
from diffusers import DiffusionPipeline

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata

device = 'cuda'

def im_to_np(im):
  im = (im / 2 + 0.5).clamp(0, 1)
  im = im.detach().cpu().permute(1, 2, 0).numpy()
  im = (im * 255).round().astype("uint8")
  return im


# Garbage collection function to free memory
def flush():
    gc.collect()
    torch.cuda.empty_cache()

#%%
from transformers import T5EncoderModel
def Run_Anagram(prompt_1,prompt_2,pre_fix = '10001-',animate=True):
    text_encoder = T5EncoderModel.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        subfolder="text_encoder",
        device_map="auto",
        variant="fp16",
        torch_dtype=torch.float16,
    )

    pipe = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",safety_checker=None,
        text_encoder=text_encoder,  # pass the previously instantiated text encoder
        unet=None                   # do not use a UNet here, as it uses too much memory
    )
    pipe = pipe.to(device)

    #

    ###############################
    ### Feel free to change me: ###
    ###############################
    # prompt_1 = 'painting of a snowy mountain village'


    # Embed prompts using the T5 model
    prompts = [prompt_1, prompt_2]
    prompt_embeds = [pipe.encode_prompt(prompt) for prompt in prompts]
    prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
    prompt_embeds = torch.cat(prompt_embeds)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds


    ######################################
    ### Optionally embed more prompts, ###
    ### as we will delete the T5       ###
    ### encoder in the next block      ###
    ######################################

    #more_prompts = ['another prompt', 'another prompt']
    #more_prompt_embeds = [pipe.encode_prompt(prompt) for prompt in more_prompts]
    #more_prompt_embeds, _ = zip(*more_prompt_embeds)
    #more_prompt_embeds = torch.cat(more_prompt_embeds)

    # Delete the Text Encoder
    del text_encoder
    del pipe
    flush()
    flush()   # For some reason we need to do this twice


    #


    # Load DeepFloyd IF stage I
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-L-v1.0",
        text_encoder=None,safety_checker=None,
        variant="fp16",
        torch_dtype=torch.float16,
    )
    stage_1.enable_model_cpu_offload()
    stage_1.to(device)

    # Load DeepFloyd IF stage II
    stage_2 = DiffusionPipeline.from_pretrained(
                    "DeepFloyd/IF-II-L-v1.0",
                    text_encoder=None,safety_checker=None,
                    variant="fp16",
                    torch_dtype=torch.float16,
                )
    stage_2.enable_model_cpu_offload()
    stage_2.to(device)

    # Load DeepFloyd IF stage III
    # (which is just Stable Diffusion 4x Upscaler)
    stage_3 = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler",
                    torch_dtype=torch.float16
                )
    stage_3.enable_model_cpu_offload()
    stage_3 = stage_3.to(device)

    #
    # UNCOMMENT ONE OF THESE

    # views = get_views(['identity', 'rotate_180'])
    # views = get_views(['identity', 'rotate_cw'])
    # views = get_views(['identity', 'rotate_ccw'])
    # views = get_views(['identity', 'flip'])
    #views = get_views(['identity', 'negate'])
    # views = get_views(['identity', 'skew'])
    # views = get_views(['identity', 'patch_permute'],view_args=[None,4])
    # views = get_views(['identity', 'pixel_permute'])
    # views = get_views(['identity', 'inner_circle'])
    # views = get_views(['identity', 'square_hinge'])
    views = get_views(['identity', 'jigsaw'])

    #
    image_64 = sample_stage_1(stage_1,
                            prompt_embeds,      # Replace with different prompts
                            negative_prompt_embeds,
                            views,
                            num_inference_steps=30,
                            guidance_scale=10.0,
                            reduction='mean',
                            generator=None)

    # Show result
    # mp.show_images([im_to_np(view.view(image_64[0])) for view in views])
    #
    image_256 = sample_stage_2(stage_2,
                            image_64,
                            prompt_embeds,      # Replace with different prompts
                            negative_prompt_embeds,
                            views,
                            num_inference_steps=30,
                            guidance_scale=10.0,
                            reduction='mean',
                            noise_level=50,
                            generator=None)

    # Show result
    # mp.show_images([im_to_np(view.view(image_256[0])) for view in views])
    #
    image_1024 = stage_3(
                    prompt=prompts[0],  # Note this is a string, and not an embedding
                    image=image_256,
                    noise_level=0,
                    output_type='pt',
                    generator=None).images
    image_1024 = image_1024 * 2 - 1

    # Limit display size, otherwise it's too large for most screens
    # mp.show_images([im_to_np(view.view(image_1024[0])) for view in views], width=400)
    # save 1024 img1 and img2.
    # pre_fix = '10001-'
    img1 = im_to_np(views[0].view(image_1024[0]))
    img2 = im_to_np(views[1].view(image_1024[0]))
    img1 = Image.fromarray(img1, 'RGB')
    img2 = Image.fromarray(img2, 'RGB')
    img1.save(pre_fix+'1.png')
    img2.save(pre_fix+'2.png')

    #
    from visual_anagrams.animate import animate_two_view
    import torchvision.transforms.functional as TF

    ##############################
    # UNCOMMENT FOR DESIRED SIZE #
    ##############################
    #image = image_64
    #image = image_256
    image = image_1024

    # Get size
    im_size = image.shape[-1]
    frame_size = int(im_size * 1.5)

    # Make save path
    save_video_path = f'./{pre_fix}3.mp4'

    # Convert to PIL
    pil_image = TF.to_pil_image(image[0] / 2. + 0.5)

    # Make the animation
    if animate:
        animate_two_view(
                    pil_image,
                    views[1], # Use the non-identity view to transform
                    prompt_1,
                    prompt_2,
                    save_video_path=save_video_path,
                    hold_duration=120,
                    text_fade_duration=10,
                    transition_duration=45,
                    im_size=im_size,
                    frame_size=frame_size,
                )

    # Display the video (using max width of 600 so will fit on most screens)
    # mp.show_video(mp.read_video(save_video_path), fps=30, width=min(600, frame_size))
    del stage_1
    del stage_2
    del stage_3
    flush()
    flush()


#%% run parts
if __name__ == '__main__':

    N_repeat = 15
    # styles = ['a pencil sketch of ','an oil painting of ','a lithograph of ','a painting of ','a watercolor painting of ']
    styles = ['an oil painting of ','a lithograph of ','a painting of ','a watercolor painting of ']
    # almost face-face
    obj_pair1 = [['a cat','a dog'],['a monkey','Albert Einstein'],['a rabbit','a duck'],['a monkey','a sheep'],['an old man','a bear']]
    # face-ani
    obj_pair2 = [['a butterfly','a dog'],['a fish','a duck'],['a snake','an eagle'],['a deer','an insect'],['a cow','a penguin']]
    # ani-inani
    obj_pair3 = [['a monkey','a kitchenware'],['a horse','a chair'],['a deer','a truck'],['a monkey','a fruit bowl'],['a human face','a houseplant']]
    # inani-inani
    obj_pair4 = [['a train','a house'],['a kettle','a fruit bowl'],['a shoe','a light bulb'],['a jar','a chair'],['a piece of cake','a shirt']]

    counter = 231

    for j,c_style in enumerate(styles):
        for k,c_pair in enumerate(obj_pair4):
            for i in range(N_repeat):
                print(f'Processing Graph {counter}')
                obj1 = c_pair[0]
                obj2 = c_pair[1]
                prompt_1 = c_style+obj1
                prompt_2 = c_style+obj2
                pre_fix = str(10000+counter)+'-'+str(i)+'-'
                Run_Anagram(prompt_1,prompt_2,pre_fix=pre_fix)
            counter += 3
    #%% single adjust
    styles = ['a pencil sketch of ','an oil painting of ','a lithograph of ','a painting of ','a watercolor painting of ']
    obj_pair1 = [['a cat','a dog'],['a monkey','Albert Einstein'],['a rabbit','a duck'],['a monkey','a sheep'],['an old man','a bear']]
    # face-ani
    obj_pair2 = [['a butterfly','a dog'],['a fish','a duck'],['a snake','an eagle'],['a deer','an insect'],['a cow','a penguin']]
    # ani-inani
    obj_pair3 = [['a monkey','a kitchenware'],['a horse','a chair'],['a deer','a truck'],['a monkey','a fruit bowl'],['a man','a houseplant']]
    # inani-inani
    obj_pair4 = [['a train','a house'],['a kettle','a fruit bowl'],['a shoe','a light bulb'],['a jar','a chair'],['a piece of cake','a shirt']]
    for i in range(15):
        c_style = styles[4]
        graph_id = 4
        p1 = f'{c_style}{obj_pair4[graph_id][0]}'
        p2 = f'{c_style}{obj_pair4[graph_id][1]}'
        Run_Anagram(p1,p2,pre_fix=f'10298-{i}-',animate = True)


#%%


