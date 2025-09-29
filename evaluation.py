import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
from datasets import load_dataset,Dataset
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from style_rl.pipeline_stable_diffusion_3_instruct_pix2pix import StableDiffusion3InstructPix2PixPipeline
from style_rl.prompt_list import real_test_prompt_list
from style_rl.img_helpers import concat_images_horizontally
import wandb
from style_rl.eval_helpers import DinoMetric
from transformers import AutoProcessor, CLIPModel
#import ImageReward as RM
from src.customized_pipe import TI2I_StableDiffusion3Pipeline
from src.attn_processor import TI2I_JointAttnProcessor2_0_multi


parser=argparse.ArgumentParser()



parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/mtg")
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--project_name",type=str,default="baseline")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--background",action="store_true")
parser.add_argument("--object",type=str, default="person")
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/rectifid")



def main(args):
    output_dict={
        "image":[],
        "augmented_image":[],
        "text_score":[],
        "image_score":[],
        "dino_score":[],
        "prompt":[]
    }
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device
    #ir_model=RM.load("ImageReward-v1.0")
    pipe = TI2I_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                            torch_dtype=torch_dtype,
                                                            device_map="balanced")

    layer={"layer_count" :0}
    operator="concat"
    attn_processors = []
    def iter_net(net):
        #global layer_count
        for child in net.children():
            if "Attention" in child.__class__.__name__:
                child.processor = TI2I_JointAttnProcessor2_0_multi(contextual_replace=True, operator=operator,
                wta_control_signal={"on":False},
                ref_control_signal={"on":False})
                attn_processors.append(child.processor)
                layer["layer_count"] += 1
            iter_net(child)

    if args.background:
        def iter_net(net):
            #global layer_count
            for child in net.children():
                if "Attention" in child.__class__.__name__:
                    child.processor = TI2I_JointAttnProcessor2_0_multi(layer=layer["layer_count"], contextual_replace=True,
                                                                    wta_control_signal={"on":True,
                                                                                        "hyper_parameter":{
                                                                                                        "wta_weight":[1, 1, 1,1],
                                                                                                        "cross2ref":True,
                                                                                                        "wta_shift":[0, 0, 0,0],
                                                                                                        "wta_cross":False
                                                                                                        },
                                                                                        "debug":False},
                                                                    ref_control_signal={"on":True,
                                                                                        "ref_idxs":[0,1,2,3], 
                                                                                        "control_type":"main_context",
                                                                                        "debug":False,
                                                                                        "control_layers":[i for i in range(25,40)],
                                                                                        "hyper_parameter":{}},
                                                                                        )
                    attn_processors.append(child.processor)
                    layer["layer_count"] += 1
                iter_net(child)

    iter_net(pipe.transformer)
        
        
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    

    dino_metric=DinoMetric(accelerator.device)

    

    

    data=load_dataset(args.src_dataset, split="train")
    #ir_model=RM.load("ImageReward-v1.0")

    background_data=load_dataset("jlbaker361/real_test_prompt_list",split="train")
    background_dict={row["prompt"]:row["image"] for row in background_data}

    text_score_list=[]
    image_score_list=[]
    image_score_background_list=[]
    #ir_score_list=[]
    dino_score_list=[]

    if args.background:
        from quad_entry import convert_to_raimg_prompt
    else:
        from single_entry import convert_to_raimg_prompt

    for k,row in enumerate(data):
        if k==args.limit:
            break
        
        object=args.object
        if "object" in row:
            object=row["object"]
        prompt=real_test_prompt_list[k%len(real_test_prompt_list)]
        background_image=background_dict[prompt]
        image=row["image"]
        main_prompt=object+" "+prompt
        if args.background:
            sub_prompt=[object, prompt]
            image=[image, background_image]
        else:
            sub_prompt=[main_prompt]
            image=[image]


        augmented_image = pipe.img2img_multi(
            images=image,
            prompt_a=main_prompt,
            prompt_b=sub_prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=3.5,
            strength=1,
            height=args.size,
            width=args.size,
        )

        print(type(augmented_image))
        print(len(augmented_image))
        print(type(augmented_image[0]),type(augmented_image[1]))

        concat=concat_images_horizontally([row["image"]]+augmented_image)

        accelerator.log({
            f"image_{k}":wandb.Image(concat)
        })
        with torch.no_grad():
            inputs = processor(
                    text=[prompt], images=[image,augmented_image,background_image], return_tensors="pt", padding=True
            )

            outputs = clip_model(**inputs)
            image_embeds=outputs.image_embeds.detach().cpu()
            text_embeds=outputs.text_embeds.detach().cpu()
            logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
        #accelerator.print("logits",logits_per_text.size())

        image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]

        [_,text_score,__]=logits_per_text
        [_,image_score,image_score_background]=image_similarities
        #ir_score=ir_model.score(prompt,augmented_image)
        dino_score=dino_metric.get_scores(image, [augmented_image])

        text_score_list.append(text_score.detach().cpu().numpy())
        image_score_list.append(image_score)
        image_score_background_list.append(image_score_background)
       # ir_score_list.append(ir_score)
        dino_score_list.append(dino_score)

        output_dict["augmented_image"].append(augmented_image)
        output_dict["image"].append(image)
        output_dict["dino_score"].append(dino_score)
        output_dict["image_score"].append(image_score)
        output_dict["text_score"].append(text_score)
        output_dict["prompt"].append(prompt)

       
    accelerator.log({
        "text_score_list":np.mean(text_score_list),
        "image_score_list":np.mean(image_score_list),
        "image_score_background_list":np.mean(image_score_background_list),
       # "ir_score_list":np.mean(ir_score_list),
        "dino_score_list":np.mean(dino_score_list)
    })

    Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)



if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")