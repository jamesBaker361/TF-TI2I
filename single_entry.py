import numpy as np
import torch
import os
import re
from tqdm import tqdm
import PIL
from PIL import Image
from diffusers.utils import make_image_grid
import sys

from src.customized_pipe import TI2I_StableDiffusion3Pipeline
from src.attn_processor import TI2I_JointAttnProcessor2_0_multi

# pipe = Customized_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# pipe = Customized_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
#                                                             torch_dtype=torch.float16,
#                                                             device_map="balanced")
pipe = TI2I_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                            torch_dtype=torch.float16,
                                                            device_map="balanced")


test_type_2_assignments = {
    "objimg_acttxt": ["obj","act"],
    "objimg_bgtxt": ["obj","bg"],
    "objimg_textxt": ["obj","tex"],
    "objtxt_actimg": ["obj","act"],
    "objtxt_bgimg": ["obj","bg"],
    "objtxt_teximg": ["obj","tex"],
}

test_type_2_image_assigment = {
    "objimg_acttxt": ["obj"],
    "objimg_bgtxt": ["obj"],
    "objimg_textxt": ["obj"],
    "objtxt_actimg": ["act"],
    "objtxt_bgimg": ["bg"],
    "objtxt_teximg": ["tex"],
}

def convert_to_raimg_prompt(source_prompt):
    ref_txt_dict={}
    obj_match = re.findall(r"<obj>(.*?)</obj>", source_prompt)
    if obj_match:
        ref_txt_dict["obj"]=obj_match

    act_match = re.findall(r"<act>(.*?)</act>", source_prompt)
    if act_match:
        ref_txt_dict["act"]=act_match


    tex_match = re.findall(r"<tex>(.*?)</tex>", source_prompt)
    if tex_match:
        ref_txt_dict["tex"]=tex_match
        


    bg_match = re.findall(r"<bg>(.*?)</bg>", source_prompt)
    if bg_match:
        ref_txt_dict["bg"]=bg_match


    image_assignment = test_type_2_image_assigment[test_type]
    # sub_prompts=[]
    # for word in source_prompt.split():         
    #     word = word.replace("<obj>", "</obj>").replace("<bg>", "</bg>").replace("<tex>", "</tex>").replace("<act>", "</act>")
    #     emu_prompt.append(word)
    return source_prompt.replace("<obj>","").replace("</obj>","").replace("<bg>","").replace("</bg>","").replace("<tex>","").replace("</tex>","").replace("<act>","").replace("</act>",""),ref_txt_dict