import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import random

from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *
from models.classifier import Masker

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from generative_model.VQGAN_v2 import VQGAN_v2, load_vqgan_model
import generative_model.clip as clip

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
    parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
    parser.add_argument("--workeri", default=None, type=int, help="Which Domain")
    parser.add_argument("--config", default=None, help="Experiment configs")
    args = parser.parse_args()
    config_file = "config." + args.config.replace("/", ".")
    print(f"\nLoading experiment {args.config}\n")
    config = __import__(config_file, fromlist=[""]).config

    return args, config


class InterventionHandler:
    def __init__(self, args, config, device):
        self.args = args
        self.config = config
        self.device = device
        self.global_step = 0

        # dataloaders
        # self.intervene_loader = get_intervention_dataloader(args=self.args, config=self.config)

        # generative model
        # clip_model = "ViT-B/32"
        # self.perceptor, self.preprocess = clip.load(clip_model, jit=False)
        # self.perceptor.eval().requires_grad_(False).to(self.device)
        # self.model = load_vqgan_model('/datasets/jianhaoy/checkpoints/vqgan_imagenet_f16_16384.yaml', '/datasets/jianhaoy/checkpoints/vqgan_imagenet_f16_16384.ckpt')
        # self.model.to(self.device)

    def officehome_vqgan_intervene(self):
        cur_domain = self.args.source[self.args.workeri] 
        aug_dict = {'Art':['an art image of'],'Clipart':['a clipart image of'],'Product':['an product image of '],'Real_World':['a real world image of']}
        # aug_dict = {'Art':['a sketch of', 'a painting of', 'an ornamentation of the shape of', 'an artistic image of'],'Clipart':['a clipart image of'],'Product':['an image without background of '],'Real_World':['a realistic photo of']}
        # from_dict = {'Art':'an artistic image of','Clipart':'a clipart image of','Product':'a no background image of','Real_World':'a realistic photo of'}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            # Mark
            # if c < 3200:
            #     continue
            toks = line.split(' ')
            img_path = toks[0]
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            img = Image.open(img_path).convert('RGB')
            remain_domains = [i for i in self.args.source if i != cur_domain]

            aug_path_dict = {}
            for to_domain in remain_domains:
                # if to_domain in ['Clipart','Real_World']:
                #     continue
                template = random.choice(aug_dict[to_domain])
                to_text = f'{template} the {class_label}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    img_res= VQGAN_v2(img_path=img_path,model=self.model,perceptor=self.perceptor,device=self.device,texts=to_text,max_iterations=100,image_size=[224,224])[-1]
                    img_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/oh_{cur_domain}_min_vqgan_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f) 
     
    def pacs_intervene(self):
        cur_domain = self.args.source[self.args.workeri] 
        # aug_dict = {'art_painting':['an art painting of'],'sketch':['a sketch of'],'cartoon':['a cartoon of'],'photo':['a photo of']}
        aug_dict = {'art_painting': ['an oil painting of', 'a painting of', 'a fresco of', 'a colourful painting of', 'an abstract painting of', 'a naturalistic painting of', 'a stylised painting of', 'a watercolor painting of', 'an impressionist painting of', 'a cubist painting of', 'an expressionist painting of','an artistic painting of'], 'sketch':['an ink pen sketch of', 'a charcoal sketch of', 'a black and white sketch', 'a pencil sketch of', 'a rough sketch of', 'a kid sketch of', 'a notebook sketch of','a simple quick sketch of'], 'photo': ['a photo of', 'a picture of', 'a polaroid photo of', 'a black and white photo of', 'a colourful photo of', 'a realistic photo of'], 'cartoon': ['an anime drawing of', 'a cartoon of', 'a colorful cartoon of', 'a black and white cartoon of', 'a simple cartoon of', 'a disney cartoon of', 'a kid cartoon style of']}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            # if c < 1000:
                # continue

            toks = line.split(' ')
            img_path = toks[0]
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            remain_domains = [i for i in self.args.source if i != cur_domain]
            aug_path_dict = {}

            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                to_text = f'{template} the {class_label}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}')
                # print(to_text,out_path)
                if os.path.exists(out_path):
                    if os.path.exists(os.path.join(out_path,'14.jpg')):
                        aug_path_dict.update({to_domain:out_path})
                        continue
                else:
                    os.mkdir(out_path)
                try:
                    img_res= VQGAN_v2(img_path=img_path,model=self.model,perceptor=self.perceptor,device=self.device,texts=to_text,max_iterations=300,image_size=[224,224])
                    for i in range(len(img_res)):
                        save_path = os.path.join(out_path,f'{i}.jpg')
                        img_res[i].save(save_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')

            link_dict.update({img_path:aug_path_dict})

        print(c)
        outfile = f'./link/{cur_domain}_engi_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f) 

    def pacs_sd_intervene(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = self.args.source[self.args.workeri] 
        # aug_dict = {'art_painting':['an art painting of'],'sketch':['a sketch of'],'cartoon':['a cartoon of'],'photo':['a photo of']}
        aug_dict = {'art_painting': ['an oil painting of', 'a painting of', 'a fresco of', 'a colourful painting of', 'an abstract painting of', 'a naturalistic painting of', 'a stylised painting of', 'a watercolor painting of', 'an impressionist painting of', 'a cubist painting of', 'an expressionist painting of','an artistic painting of'], 'sketch':['an ink pen sketch of', 'a charcoal sketch of', 'a black and white sketch', 'a pencil sketch of', 'a rough sketch of', 'a kid sketch of', 'a notebook sketch of','a simple quick sketch of'], 'photo': ['a photo of', 'a picture of', 'a polaroid photo of', 'a black and white photo of', 'a colourful photo of', 'a realistic photo of'], 'cartoon': ['an anime drawing of', 'a cartoon of', 'a colorful cartoon of', 'a black and white cartoon of', 'a simple cartoon of', 'a disney cartoon of', 'a kid cartoon style of']}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            print(f'Current Image:{c}/{len(lines)}')
            c += 1
            # if c < 2000:
            #     continue

            toks = line.split(' ')
            img_path = toks[0]
            init_image = Image.open(img_path).resize((512, 512))
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            remain_domains = [i for i in self.args.source if i != cur_domain]
            aug_path_dict = {}

            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                to_text = f'{template} the {class_label}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                # print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=80, 
                        strength=0.9, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')

            link_dict.update({img_path:aug_path_dict})

        # print(c)
        outfile = f'./link/{cur_domain}_sd_engi_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)

    def pacs_sd_adaptation_intervene(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = self.args.source[self.args.workeri] 
        aug_dict = {'art_painting':['art painting'],'sketch':['black ink sketch'],'cartoon':['cartoon'],'photo':['photo']}
        # aug_dict = {'art_painting': ['an oil painting of', 'a painting of', 'a fresco of', 'a colourful painting of', 'an abstract painting of', 'a naturalistic painting of', 'a stylised painting of', 'a watercolor painting of', 'an impressionist painting of', 'a cubist painting of', 'an expressionist painting of','an artistic painting of'], 'sketch':['an ink pen sketch of', 'a charcoal sketch of', 'a black and white sketch', 'a pencil sketch of', 'a rough sketch of', 'a kid sketch of', 'a notebook sketch of','a simple quick sketch of'], 'photo': ['a photo of', 'a picture of', 'a polaroid photo of', 'a black and white photo of', 'a colourful photo of', 'a realistic photo of'], 'cartoon': ['an anime drawing of', 'a cartoon of', 'a colorful cartoon of', 'a black and white cartoon of', 'a simple cartoon of', 'a disney cartoon of', 'a kid cartoon style of']}
        file_name = f'{cur_domain}_test.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            print(f'Current Image:{c}/{len(lines)}')
            c += 1
            # if c < 2000:
            #     continue

            toks = line.split(' ')
            img_path = toks[0]
            init_image = Image.open(img_path).resize((512, 512))
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            remain_domains = [i for i in self.args.source if i != cur_domain]
            aug_path_dict = {}

            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                to_text = f'{template} image'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                # print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=25, 
                        strength=0.75, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')

            link_dict.update({img_path:aug_path_dict})

        # print(c)
        outfile = f'./link/{cur_domain}_sd_dumb_adaptation_noclass_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)

    def pacs_sd_multi_samples_intervene(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 8

        cur_domain = self.args.source[self.args.workeri] 
        aug_dict = {'art_painting':['an art painting of'],'sketch':['a sketch of'],'cartoon':['a cartoon of'],'photo':['a photo of']}
        # aug_dict = {'art_painting': ['an oil painting of', 'a painting of', 'a fresco of', 'a colourful painting of', 'an abstract painting of', 'a naturalistic painting of', 'a stylised painting of', 'a watercolor painting of', 'an impressionist painting of', 'a cubist painting of', 'an expressionist painting of','an artistic painting of'], 'sketch':['an ink pen sketch of', 'a charcoal sketch of', 'a black and white sketch', 'a pencil sketch of', 'a rough sketch of', 'a kid sketch of', 'a notebook sketch of','a simple quick sketch of'], 'photo': ['a photo of', 'a picture of', 'a polaroid photo of', 'a black and white photo of', 'a colourful photo of', 'a realistic photo of'], 'cartoon': ['an anime drawing of', 'a cartoon of', 'a colorful cartoon of', 'a black and white cartoon of', 'a simple cartoon of', 'a disney cartoon of', 'a kid cartoon style of']}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            print(f'Current Image:{c}/{len(lines)}')
            c += 1
            # if c < 750:
            #     continue

            toks = line.split(' ')
            img_path = toks[0]
            init_image = Image.open(img_path).resize((512, 512))
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            remain_domains = [i for i in self.args.source if i != cur_domain]
            aug_path_dict = {}

            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                to_text = f'{template} the {class_label}'
                
                for part in [2]:
                    out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}_p{part}')
                    # print(to_text,out_path)
                    if os.path.exists(out_path):
                        aug_path_dict.update({to_domain:out_path})
                        continue
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    # try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=80, 
                        strength=0.9, guidance_scale=7.5).images
                    for i in range(len(image_res)):
                        out_s_path = f'{out_path}/{i}.jpg'
                        image_res[i].save(out_s_path)
                        aug_path_dict.update({to_domain:out_s_path})
                # except:
                #     aug_path_dict.update({to_domain:img_path})
                #     print(img_path,'Here Error?')

            link_dict.update({img_path:aug_path_dict})

        # print(c)
        outfile = f'./link/{cur_domain}_sd_multis_link_2.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)
    
    def pacs_text_inversion(self):
        class_lis = ["dog","elephant","giraffe","guitar","horse","house","person"]

        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = self.args.source[self.args.workeri] 
        aug_dict = {'art_painting': ['an oil painting in the style of ', 'a painting in the style of ', 'a fresco in the style of ', 'a colourful painting in the style of ', 'a stylised painting in the style of ', 'an artistic painting in the style of '], 
                    'sketch':['an ink pen sketch in the style of ', 'a charcoal sketch in the style of ', 'a black and white sketch in the style of ', 'a pencil sketch in the style of ', 'a rough sketch in the style of ', 'a kid sketch in the style of '], 
                    'photo': ['a photo in the style of ', 'a picture in the style of ', 'a polaroid photo in the style of ', 'a colourful photo in the style of ', 'a realistic photo in the style of '], 
                    'cartoon': ['an cartoon drawing in the style of ', 'a cartoon in the style of ', 'a colorful cartoon in the style of ', 'a simple cartoon in the style of ', 'a disney cartoon in the style of ', 'a kid cartoon style in the style of ']}
        link_dict = {}
        c = 0
        for img_class in class_lis:
            
            label_path = os.path.join('/homes/55/jianhaoy/projects/EKI/test_VQGAN/text_inversion_all_labels/pacs',f'{cur_domain}_{img_class}_train.txt')
            f = open(label_path,'r')
            lines = f.readlines()
            for line in tqdm(lines):
                c += 1
                # if c < 2000:
                #     continue
                toks = line.split(' ')
                img_path = toks[0]
                init_image = Image.open(img_path).resize((512, 512))
                class_label = img_path.split('/')[-2].replace('_', ' ').lower()
                remain_domains = [i for i in self.args.source if i != cur_domain]
                aug_path_dict = {}
                for to_domain in remain_domains:
                    placeholder_token = f"<{to_domain}-{img_class}-style>"
                    template = random.choice(aug_dict[to_domain])
                    to_text = f'{template}{placeholder_token}'
                    out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                    print(to_text,out_path)
                    if os.path.exists(out_path):
                        aug_path_dict.update({to_domain:out_path})
                        continue
                    try:
                        image_res = pipe(
                            prompt=[to_text]*num_samples, init_image=init_image,
                            num_inference_steps=80, 
                            strength=0.9, guidance_scale=7.5).images[0]
                        image_res.save(out_path)
                        aug_path_dict.update({to_domain:out_path})
                    except:
                        aug_path_dict.update({to_domain:img_path})
                        print(img_path,'Here Error?')

                link_dict.update({img_path:aug_path_dict})

        # print(c)
        outfile = f'./link/{cur_domain}_sd_textinv_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)

    def officehome_intervene_sd(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = self.args.source[self.args.workeri] 
        aug_dict = {'Art':['an art image of'],'Clipart':['a clipart image of'],'Product':['an product image of '],'Real_World':['a real world image of']}
        # aug_dict = {'Art':['a sketch of', 'a painting of', 'an artistic image of'],'Clipart':['a clipart image of'],'Product':['an image without background of '],'Real_World':['a realistic photo of']}
        # Hand-crafted
        # aug_dict = {'Art':['a sketch of ', 'a painting of ', 'an artistic image of ','an oil painting ', 'a painting ', 'a fresco ', 'a colourful painting of ', 'a stylised painting of ', 'an artistic painting of '],\
        #             'Clipart':['a clipart image of ','an cartoon drawing in the style of ', 'a cartoon in the style of ', 'a colorful cartoon in the style of ', 'a simple cartoon '],\
        #             'Product':["an image on a white background of ", "an image on a grey background of ", "an image on a black background of ", "an product showcase picture of "],\
        #             'Real_World':['a realistic photo of ', 'a photo in the style of ', 'a picture in the style of ', 'a polaroid photo of ', 'a colourful photo of ']}
        # from_dict = {'Art':'an artistic image of','Clipart':'a clipart image of','Product':'a no background image of','Real_World':'a realistic photo of'}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):

            c += 1
            if c < 3900:
                continue
            toks = line.split(' ')
            img_path = toks[0]
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            remain_domains = [i for i in self.args.source if i != cur_domain]
            init_image = Image.open(img_path).resize((512, 512))
            aug_path_dict = {}
            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                to_text = f'{template} the {class_label}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=80, 
                        strength=0.9, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/officehome_SD_dumb_link/{cur_domain}_handcrafted_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f) 
     
    def imagenet9_intervene_sd(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = 'original'
        aug_lis = [" in a parking lot",
                        # " on a sidewalk",
                        " on a tree root",
                        " on the branch of a tree",
                        " in an aquarium",
                        " in front of a reef",
                        " on the grass",
                        " on a sofa",
                        " in the sky",
                        " in front of a cloud",
                        " in a forest",
                        " on a rock",
                        " in front of a red-brick wall",
                        " in a living room",
                        " in a school class",
                        " in a garden",
                        " on the street",
                        " in a river",
                        " in a wetland",
                        " held by a person",
                        " on the top of a mountain",
                        " in a nest",
                        " in the desert",
                        " on a meadow",
                        " on the beach",
                        " in the ocean",
                        " in a plastic container",
                        " in a box",
                        " at a restaurant",
                        " on a house roof",
                        " in front of a chair",
                        " on the floor",
                        " in the lake",
                        " in the woods",
                        " in a snowy landscape",
                        " in a rain puddle",
                        " on a table",
                        " in front of a window",
                        " in a store",
                        " in a blurred backround"]
        #  aug_dict = {'rand_background':[' in a beautiful landscape', ' in the forest', ' in the woods' ' on the grass', ' in the meadow', ' under a tree', ' in the garden', ' on the mountain', ' in the sky', ' in the desert', ' in a wetland',
# ' by the river', ' on the lake', ' in the water', ' on the beach', ' in a water background', ' in an ocean background', ' on the street', ' by the road', ' held by a person', ' next to a person', ' on the table', ' on the floor', ' on the sofa', ' on the chair', ' on the roof', ' in the house', ' in the living room', ' in the building', ' in the bedroom']}
        aug_dict = {'rand_background':aug_lis,'rand_background1':aug_lis,'rand_background2':aug_lis,'rand_background3':aug_lis,'rand_background4':aug_lis,'rand_background5':aug_lis,'rand_background6':aug_lis,'rand_background7':aug_lis,'rand_background8':aug_lis}
        class_label_dict = {'carnivore':['lion','lion','wolf','wolf','tiger','tiger','panther','panther','hyenas','hyenas','polar bear','bear','mustelid','pandas','marmot'],
                            'bird':['bird','bird','bird','chicken','rooster','eagle','owl','parrot','geese'],
                            'reptile':['reptile','turtle','sea turtle','lizard','chameleon','gecko','snake','snake','snake'],
                            'dog':['dog'],
                            'primate':['primate'],
                            'fish':['fish'],
                            'insect':['insect'],
                            'instrument':['musical instrument','guitar','violin','cello','chimes','accordion','bell','bell','trombone','trumpet','tambour','drum','marimba','sand hammer','clarinet','oboe','ocarina','pipe organ','saxophone'],
                            'vehicle':['wheeled vehicle','wheeled vehicle','vehicle','vehicle','vehicle','tank','bulldozer','train']}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            # if c < 2300:
            #     continue
            toks = line.split(' ')
            # print(toks)
            img_path = toks[0]
            class_label_r = img_path.split('/')[-2].replace('_', ' ').lower().split(' ')[-1]
        
            # remain_domains = ['rand_background','rand_background1','rand_background2','rand_background3','rand_background4','rand_background5','rand_background6']
            remain_domains = ['rand_background','rand_background1','rand_background2','rand_background3','rand_background4','rand_background5']
            init_image = Image.open(img_path).resize((512, 512))
            aug_path_dict = {}
            for to_domain in remain_domains:
                class_label = random.choice(class_label_dict[class_label_r])
                print(class_label)
                template = random.choice(aug_dict[to_domain])
                to_text = f'a realistic photo of the {class_label}{template}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label_r}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=50, 
                        strength=0.9, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/imagenet9_SD_template_link/{cur_domain}_sd_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)     

    def imagenet9_intervene_sd_inpaint(self):
        # SD hyper-parameters
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = 'original'
        aug_lis = [" in a parking lot",
                        # " on a sidewalk",
                        " on a tree root",
                        " on the branch of a tree",
                        " in an aquarium",
                        " in front of a reef",
                        " on the grass",
                        " on a sofa",
                        " in the sky",
                        " in front of a cloud",
                        " in a forest",
                        " on a rock",
                        " in front of a red-brick wall",
                        " in a living room",
                        " in a school class",
                        " in a garden",
                        " on the street",
                        " in a river",
                        " in a wetland",
                        " held by a person",
                        " on the top of a mountain",
                        " in a nest",
                        " in the desert",
                        " on a meadow",
                        " on the beach",
                        " in the ocean",
                        " in a plastic container",
                        " in a box",
                        " at a restaurant",
                        " on a house roof",
                        " in front of a chair",
                        " on the floor",
                        " in the lake",
                        " in the woods",
                        " in a snowy landscape",
                        " in a rain puddle",
                        " on a table",
                        " in front of a window",
                        " in a store",
                        " in a blurred backround"]
        #  aug_dict = {'rand_background':[' in a beautiful landscape', ' in the forest', ' in the woods' ' on the grass', ' in the meadow', ' under a tree', ' in the garden', ' on the mountain', ' in the sky', ' in the desert', ' in a wetland',
# ' by the river', ' on the lake', ' in the water', ' on the beach', ' in a water background', ' in an ocean background', ' on the street', ' by the road', ' held by a person', ' next to a person', ' on the table', ' on the floor', ' on the sofa', ' on the chair', ' on the roof', ' in the house', ' in the living room', ' in the building', ' in the bedroom']}
        aug_dict = {'rand_background':aug_lis,'rand_background1':aug_lis,'rand_background2':aug_lis,'rand_background3':aug_lis,'rand_background4':aug_lis,'rand_background5':aug_lis,'rand_background6':aug_lis,'rand_background7':aug_lis,'rand_background8':aug_lis}
        class_label_dict = {'carnivore':['lion','lion','wolf','wolf','tiger','tiger','panther','panther','hyenas','hyenas','polar bear','bear','mustelid','pandas','marmot'],
                            'bird':['bird','bird','bird','chicken','rooster','eagle','owl','parrot','geese'],
                            'reptile':['reptile','turtle','sea turtle','lizard','chameleon','gecko','snake','snake','snake'],
                            'dog':['dog'],
                            'primate':['primate'],
                            'fish':['fish'],
                            'insect':['insect'],
                            'instrument':['musical instrument','guitar','violin','cello','chimes','accordion','bell','bell','trombone','trumpet','tambour','drum','marimba','sand hammer','clarinet','oboe','ocarina','pipe organ','saxophone'],
                            'vehicle':['wheeled vehicle','wheeled vehicle','vehicle','vehicle','vehicle','tank','bulldozer','train']}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        tot_len = len(lines)
        c = 0
        link_dict = {}
        mask_dir = '/datasets/jianhaoy/ImageNet-9/mask'
        for line in tqdm(lines):
            print(f'Current Image {c}/{tot_len}')
            c += 1
            if c > 2025:
                continue
            toks = line.split(' ')
            # print(toks)
            img_path = toks[0]
            class_label_r = img_path.split('/')[-2].replace('_', ' ').lower().split(' ')[-1]
            img_name = img_path.split('/')[-1]

            mask_image_path = os.path.join(mask_dir,img_name)
            # print(mask_image_path)
            remain_domains = ['rand_background','rand_background1','rand_background2','rand_background3','rand_background4','rand_background5']
            init_image = Image.open(img_path).resize((512, 512))
            mask_image = Image.open(mask_image_path).resize((512,512)).convert('RGB')
            aug_path_dict = {}
            for to_domain in remain_domains:
                class_label = random.choice(class_label_dict[class_label_r])
                template = random.choice(aug_dict[to_domain])
                to_text = f'a realistic photo of the {class_label}{template}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label_r}_{c}.jpg')
                print(class_label,to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(prompt=to_text, init_image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.9).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/imagenet9_SD_template_link/{cur_domain}_sd_inpaint_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f) 

    def imagenet9_intervene_vqgan(self):

        cur_domain = 'original'
        aug_lis =         aug_lis = [" in a parking lot",
                        # " on a sidewalk",
                        " on a tree root",
                        " on the branch of a tree",
                        " in an aquarium",
                        " in front of a reef",
                        " on the grass",
                        " on a sofa",
                        " in the sky",
                        " in front of a cloud",
                        " in a forest",
                        " on a rock",
                        " in front of a red-brick wall",
                        " in a living room",
                        " in a school class",
                        " in a garden",
                        " on the street",
                        " in a river",
                        " in a wetland",
                        " held by a person",
                        " on the top of a mountain",
                        " in a nest",
                        " in the desert",
                        " on a meadow",
                        " on the beach",
                        " in the ocean",
                        " in a plastic container",
                        " in a box",
                        " at a restaurant",
                        " on a house roof",
                        " in front of a chair",
                        " on the floor",
                        " in the lake",
                        " in the woods",
                        " in a snowy landscape",
                        " in a rain puddle",
                        " on a table",
                        " in front of a window",
                        " in a store",
                        " in a blurred backround"]
        aug_dict = {'rand_background':aug_lis,'rand_background1':aug_lis,'rand_background2':aug_lis,'rand_background3':aug_lis,'rand_background4':aug_lis,'rand_background5':aug_lis,'rand_background6':aug_lis,'rand_background7':aug_lis,'rand_background8':aug_lis}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            # if c < 1500:
            #     continue
            toks = line.split(' ')
            # print(toks)
            img_path = toks[0]
            class_label = img_path.split('/')[-2].replace('_', ' ').lower().split(' ')[-1]
            if class_label == 'carnivore':
                class_label = 'carnivore animal'
            remain_domains = ['rand_background','rand_background1','rand_background2','rand_background3']
            init_image = Image.open(img_path).resize((512, 512))
            aug_path_dict = {}
            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                to_text = f'a realistic photo of the {class_label}{template}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                # try:
                img_res= VQGAN_v2(img_path=img_path,model=self.model,perceptor=self.perceptor,device=self.device,texts=to_text,max_iterations=100,image_size=[224,224])[-1]
                img_res.save(out_path)
                aug_path_dict.update({to_domain:out_path})
                # except:
                #     aug_path_dict.update({to_domain:img_path})
                #     print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/imagenet9_SD_template_link/{cur_domain}_vqgan_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)     

    def celeba_sd_intervene(self):
       # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = 'original'
        aug_lis = {"female":["a gray hair","a brown hair","a black hair"],"male":["a blonde hair"]}
        aug_dict = {'rand_background':aug_lis,'rand_background1':aug_lis,'rand_background2':aug_lis,'rand_background3':aug_lis,'rand_background4':aug_lis,'rand_background5':aug_lis,'rand_background6':aug_lis,'rand_background7':aug_lis,'rand_background8':aug_lis}
        class_label_dict = {'1':['female'],
                            '0':['male']}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            if c < 2000:
                continue
            toks = line.split(' ')
            img_path = toks[0]
            class_label_r = toks[1]
        
            # remain_domains = ['rand_background','rand_background1','rand_background2','rand_background3','rand_background4','rand_background5','rand_background6']
            remain_domains = ['rand_background','rand_background1','rand_background2']
            init_image = Image.open(img_path).resize((512, 512))
            aug_path_dict = {}
            for to_domain in remain_domains:
                class_label = random.choice(class_label_dict[class_label_r])
                # print(class_label)
                template = random.choice(aug_dict[to_domain][class_label])
                to_text = f'a realistic photo of {class_label} with a {template}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label_r}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=50, 
                        strength=0.9, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/CelebA/{cur_domain}_sd_reversed_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)     


    def celeba_vqgan_intervene(self):


        cur_domain = 'original'
        aug_lis = ["a male","a female"]
        aug_dict = {'male1':["a male"],'male2':["a male"],'female1':["a female"],'female2':["a female"]}
        class_label_dict = {'1':['blonde hair'],
                            '0':['black hair','brown hair','gray hair']}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            # if c < 1800:
            #     continue
            toks = line.split(' ')
            img_path = toks[0]
            class_label_r = toks[1]

            if from_domain == 'male':
                remain_domains = ['female1','female2']
            else:
                remain_domains = ['male1','male2']
            aug_path_dict = {}
            for to_domain in remain_domains:
                class_label = random.choice(class_label_dict[class_label_r])
                # print(class_label)
                template = random.choice(aug_dict[to_domain])
                to_text = f'a realistic photo of {template} with a {class_label}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label_r}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    img_res= VQGAN_v2(img_path=img_path,model=self.model,perceptor=self.perceptor,device=self.device,texts=to_text,max_iterations=100,image_size=[224,224])[-1]
                    img_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/CelebA/{cur_domain}_vqgan_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)     
    
    def celeba_sd_control_intervene(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1
        
        cur_domain = 'original'
        aug_lis = ["a male","a female"]
        aug_dict = {'male1':["a male"],'male2':["a male"],'female1':["a female"],'female2':["a female"]}
        class_label_dict = {'1':['blonde hair'],
                            '0':['black hair','brown hair','gray hair']}
        domain_label_dict = {'1':'female','0':'male'}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            # if c < 2500:
            #     continue
            toks = line.split(' ')
            img_path = toks[0]
            class_label_r = toks[1]
            init_image = Image.open(img_path).resize((512, 512))

            from_domain = domain_label_dict[class_label_r]
            print(from_domain)
            
            if from_domain == 'male':
                remain_domains = ['female1','female2']
            else:
                remain_domains = ['male1','male2']
            aug_path_dict = {}
            for to_domain in remain_domains:
                class_label = random.choice(class_label_dict[class_label_r])
                # print(class_label)
                template = random.choice(aug_dict[to_domain])
                to_text = f'a realistic photo of {template} with a {class_label}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label_r}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=50, 
                        strength=0.8, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/CelebA/{cur_domain}_control_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)     

    def pacs_autoprompt_intervene(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = self.args.source[self.args.workeri] 
        # aug_dict = {'art_painting':['an art painting of'],'sketch':['a sketch of'],'cartoon':['a cartoon of'],'photo':['a photo of']}
        # aug_dict = {'art_painting': ['an oil painting of', 'a painting of', 'a fresco of', 'a colourful painting of', 'an abstract painting of', 'a naturalistic painting of', 'a stylised painting of', 'a watercolor painting of', 'an impressionist painting of', 'a cubist painting of', 'an expressionist painting of','an artistic painting of'], 'sketch':['an ink pen sketch of', 'a charcoal sketch of', 'a black and white sketch', 'a pencil sketch of', 'a rough sketch of', 'a kid sketch of', 'a notebook sketch of','a simple quick sketch of'], 'photo': ['a photo of', 'a picture of', 'a polaroid photo of', 'a black and white photo of', 'a colourful photo of', 'a realistic photo of'], 'cartoon': ['an anime drawing of', 'a cartoon of', 'a colorful cartoon of', 'a black and white cartoon of', 'a simple cartoon of', 'a disney cartoon of', 'a kid cartoon style of']}
        aug_dict = {'art_painting': {'dog': ['art of a dog painted in acrylic paint on a white board', 'this is art from the painting of my dog', 'art _doodles_the dog has a heart for painting', 'art... painting on the dog', 'art of dogs and painter in a pink velvet canvas with an acrylic and wooden background', 'a cat in art with painting and an old dog', 'the dog and the art of painting', 'art with a dog and paintings'], 'elephant': ['art i need to make more art with the painting on the elephant.', 'art _the elephant in the sky', 'art and painting of an elephant', 'this is a painting of a young elephant at a small village in western art', 'art of a cute elephant and it was a painting.', 'art of the elephant and the king of the palace and the castle _painting', 'art of a big elephant in a red leather painting', 'a drawing of art... an elephant is shown painting'], 'giraffe': ['art by a giraffe', 'these are a collection of collages of giraffes and other animal art done', 'a little girl grabbing a brush and some art for an ostrich and ', 'art of a giraffe with green flowers and blue skies and paints.', 'art _of giraffes on the mountain', 'art of a giraffe in its womb _paintings', 'art piece a lion and a giraffe at home using animal painted canvas', 'artwork of a giraffe on the wall in a forest with a flower'], 'guitar': ['a man makes art of painting a guitar', 'art of painting on a guitar', 'art_of_painting on guitar', 'artist sharing an art with the people using the guitar in painting', 'art of a girl painting a guitar', 'art and painting on a guitar', 'art & drawing of the guitar with a watercolor painting', 'art with a big guitar with a picture of his painting'], 'horse': ['art... some horses are made of acrylic painted frames.', 'art of painting the horses', 'art with a horse as art in painting', 'art... in the form of a horse with painted panels', 'art of art by a horse', 'a horse with painting is considered to be art', 'art and drawing from a horse_paintings.', '_art of horses and painting'], 'house': ['a house a portrait of a little girl in his art and painting', 'house painted with an art', 'a portrait of house with artwork and a painting', 'a house with a large mural of art and artwork.', 'abstract art _painting at house', 'art_in-the-house_painting', 'art for house of artists with painting', 'a series of paintings in art at the house'], 'person': ['an artist shows some original art by a person... and then a _pai', 'person expressing his love for the art of paintings', 'some person has a hard time getting enough artwork done by other people.', 'painting of a person for art', 'art for people from art of creating a collage...', 'how to pick the right art for your body... this is how i like the idea', 'a person of art and painting', 'art - an image of a person in a glass painting and the windows']}, 'photo': {'dog': ['photo of a white dog in the backyard', 'The photo above shows a dog sneezing around on a couch.', 'a photo of a dog.', 'this photo shows a dog', 'this is a photo of a dog', 'photo of a young happy dog.', 'This is a photo of a dog looking at camera.', 'this is a photo of a puppy in the dog'], 'elephant': ['a picture of an elephant', 'A photo of an elephant in the meadows.', 'aerial photo of elephants on a hillside', 'these are photos of elephants from another species', 'The picture above shows two elephants.', 'A photo of elephants on a grasslands.', 'A photo of elephants at a treetop.', 'this is a photo of an elephant for the show.'], 'giraffe': ['Two men have a photo of a giraffe in the park.', 'A photo of two giraffes eating grass.', 'A photo of a giraffe grazing on some grass.', 'A photo of a giraffe resting between two trees', 'A photo of a giraffe trying to make out.', 'the photo of a giraffe in the wild', 'A photo of a giraffe in its enclosure on the grassy ground.', 'a photo of a giraffe in the wild'], 'guitar': ['A photo of someone playing guitar and playing in their garage.', 'a photo of a woman with a blue guitar.', 'A photo of a young boy playing a guitar on a guitar.', 'photos of a man playing guitar', 'photo of guitar and bass.', 'photo of a female guitarist playing his guitar', 'photo of young girl playing guitar', 'photo of a kid playing a guitar'], 'horse': ['photos of horses grazing in the countryside', 'photograph of a black and white horse', 'A photo of a horse and his mane.', 'an animal with a horse in a grassy area and a quick photo', 'a photo of two horses grazing at the hay bale', 'photo of a horse', 'a photo of a wild horse.', 'a photo of a horse with his tail pulled over and peeping'], 'house': ['the photos on this page show this house as a restaurant.', 'some photos of house in the city', 'A photo of a house at the beach.', 'photo of a decorated and stained cottage house', 'vintage photo of old house', 'a photo of a house.', 'this is a photo of houses at a neighborhood on the island', 'aerial photo of a house'], 'person': ['person taking a photo inside a hotel', 'candid photo of person and her husband', 'photo of one of the persons', 'person appears in the photos above.', 'a photo of two people together at a coffee shop', 'an image of a person he says he has seen before.', 'person poses with photo', 'person taking a photo of the bridge']}, 'cartoon': {'dog': ['cartoon of dog in the forest', 'dog in a cartoon', 'cartoon of the dog', 'a cartoon of a dog with a wand and buttons', 'this cartoon with the big dogs is based on a movie', 'a cartoon of a man with a dog', 'cartoon of a dog with the black and white', 'this cartoon is about a black dog'], 'elephant': ['cartoon of an elephant and a woman', 'cartoon of a white elephant and his mother', 'cartoon of an elephant in the city', 'cartoon of elephant.', 'cartoon of a lone elephant', 'cartoon of a man who takes care of the elephants', 'cartoon about elephants.', 'cute cute elephant with the elephant in cartoon'], 'giraffe': ['cartoon of a giraffe with many tails', 'A cartoon showing a giraffe with a large red nose.', 'cartoon of a giraffe in a pen', 'cartoon of a large giraffe in a zoo', 'cartoon of giraffes around the trees', 'cartoon of a giraffe with two adults', 'cartoon of three giraffes sitting beside each other', 'cartoon of a giraffe in the forest'], 'guitar': ['a cartoon of a woman playing guitar at an apartment', 'cartoon of the blue and white guitar playing in a park', 'cartoon about a guitar', 'cartoon of a man with a guitar... but it’s still guitar', 'cartoon on a guitar', 'cartoon of men playing the guitar', 'cartoon on guitar', 'cartoon of boy playing guitar at the beach'], 'horse': ['the cartoon of the horses', 'cartoon of a white horse', 'cartoon of horse and his cow', 'cartoon of a horse.', 'cartoon of a horse.', 'cartoon of horses on the beach', 'cartoon of a horse', 'cartoon of young boy and girl playing around with a horse'], 'house': ['cartoon of house of worship', 'cartoon of a house.', 'cartoon of the house', 'cartoon showing a house in summer', 'a cartoon depicting people destroying a house', 'cartoon of a house to go from tv programmer to teacher', 'cartoon of an old house in the mountains', 'cartoon of a beautiful house on green'], 'person': ['cartoon about a woman waking up in the morning and a smiling person', 'cartoon of one of the first person to come across a painting', 'a cartoon about the person and his family', 'person in the cartoon of man with a cup of coffee', 'cartoon of a person', 'cartoon of a man and a man and a woman', 'cartoons showing different person as a person', 'cartoon featuring human figure with a young person']}, 'sketch': {'dog': ['a sketch showing how to get a dog on the floor', 'sketch of a dog on a green yard', 'sketch of a dog in the snow', 'sketch of a dog', 'a sketch of a dog at home', 'sketch of two dogs', 'a sketch of a dog.', 'sketch of a dog.'], 'elephant': ['A sketch of a elephant he is holding.', 'A sketch of two elephants posing next to each other.', 'sketch with a red elephants on it', 'sketch of elephant on the white and grey', 'sketch of an elephant', 'sketch of elephant in the wild', 'drawing of an elephant in the forest', 'A sketch shows all of the animals with the elephants on them.'], 'giraffe': ['A sketch of a small giraffe near a lake.', 'A sketch of a giraffe in a zoo.', 'a sketch of a giraffe at the zoo', "A sketch of a giraffe and it's mate.", 'a sketch of a young giraffe', 'sketch of a giraffe on a muddy ground.', 'A sketch of a giraffe grazing down a tree.', 'sketches of giraffes near large tree and tree stump'], 'guitar': ['sketch of a large guitar', 'sketch of a guitar', 'A sketch of a man playing a guitar.', 'sketches of small guitars and tv in a dark room', 'sketch of a guitar', 'Sketch of a woman playing a guitar', 'sketch of a guitar', 'sketch of a guitar.'], 'horse': ['sketch of a young girl standing on a horse', 'sketch of a mule deer on the horse', 'the sketch of an old horse', 'a sketch of horses in an open barn', 'a sketch of a horse on a farm', 'sketch of a horse in the mountains', 'sketch of a horse and the carriage.', 'sketch of a horse on the horizon'], 'house': ['sketch of a house', 'sketch of a large house', 'sketch of house from a book of the year', 'sketch for the house in a town', 'sketch of a house', 'this sketch demonstrates how to do this on your new house', 'sketch of a house from behind', 'sketch of the house'], 'person': ['person sketched on the beach', 'this sketch shows a portrait of a single person.', 'sketches of famous people among the creatures', 'people sketching a portrait of a female', 'A sketch of a person in a park.', 'sketch of person living in a dark room', 'famous people sketching the shattered roof on his bed', 'sketch of an ordinary person']}}
        # Conservative
        # aug_dict = {'art_painting': {'dog': ['art_painting of a dog', 'art _painting of a dog', 'art_painting of a dog.', 'art_paintings of a dog', 'art and painting of a dog', 'art _paintings of a dog', 'art_paintings of a dog.', 'art _painting of a dog.'], 'elephant': ['art _painting of an elephant', 'art _paintings of elephants', 'art _paintings of elephants.', 'art _painting of an elephant.', 'art_painting of an elephant', 'art _paintings of an elephant', 'art in the form of a painting of an elephant', 'art _paintings of elephants in the forest'], 'giraffe': ['art _painting of a giraffe', 'art_painting of a giraffe', 'art and painting of a giraffe', 'art _paintings of a giraffe', 'art_painting of a giraffe in a zoo', 'art_paintings of a giraffe', 'art_painting of a giraffe at a zoo', 'art _painting of a giraffe at the zoo'], 'guitar': ['art _painting on a guitar', 'art_painting on a guitar', 'art_of_painting on a guitar', 'art _paintings on a guitar', 'art _painting on a guitar.', 'art_paintings on a guitar', 'art_painting on a guitar.', 'art_of_painting on a guitar.'], 'horse': ['art_painting of a horse', 'art _painting of a horse', 'art _paintings on a horse', 'art_paintings on a horse', 'art_paintings of a horse', 'art_painting of a horse.', 'art _painting on a horse', 'art _paintings of a horse'], 'house': ['art _paintings in the house', 'art _paintings in a house', 'art_paintings in a house', 'art _paintings on the walls of a house', 'art_paintings in the house', 'art _paintings on a house', 'art _paintings of a house', 'art _paintings in the house.'], 'person': ['a person is a fan of art_paintings.', 'a person is a fan of art _paintings.', 'a person is a fan of art_paintings', 'portrait of a person with art _painting', 'a portrait of a person with art _paintings.', 'a person is a fan of art _paintings', 'portrait of a person with art _paintings', 'a portrait of a person with art _painting']}, 'photo': {'dog': ['a photo of a dog', 'this is a photo of a dog.', 'a photo of a dog.', 'this is a photo of a dog', 'This is a photo of a dog.', 'this is a photo of a black and white dog.', 'this is a close up photo of a dog.', 'this is a photo of a cute dog.'], 'elephant': ['A photo of an elephant in a zoo.', 'A photo of two elephants in a zoo.', 'This is a photo of an elephant.', 'This is a photo of an elephant in a zoo.', 'a photo of an elephant', 'A photo of elephants in a zoo.', 'A photo of a giraffe and an elephant.', 'A photo of a giraffe and an elephant in a zoo'], 'giraffe': ['A photo of a giraffe in a zoo.', 'A photo of a giraffe at a zoo.', 'A photo of a giraffe on a grassy field.', 'a photo of a giraffe in a zoo', 'A photo of two giraffes in a zoo.', 'a photo of a giraffe at a zoo', 'A photo of a giraffe in a zoo enclosure.', 'a photo of a giraffe at the zoo'], 'guitar': ['A photo of a man playing a guitar.', 'This is a photo of a man playing a guitar.', 'a photo of a man playing a guitar', 'A photo of a man playing guitar.', 'This is a photo of a man playing guitar.', 'A photo of a man playing his guitar.', 'This is a photo of a man playing his guitar.', 'A photo of a young man playing a guitar.'], 'horse': ['a photo of a horse', 'a photo of a horse.', 'this is a photo of a horse.', 'this is a photo of a horse', 'a photo of a white horse', 'a photo of a black and white horse', 'this is a photo of a black and white horse.', 'a photo of a giraffe and a horse'], 'house': ['this is a photo of a house.', 'a photo of a house', 'a photo of a house.', 'this is a photo of the house.', 'this is a photo of a house', 'this is a close up photo of a house.', 'this is a photo of a house in a city.', 'this is a photo of a house in the countryside.'], 'person': ['a photo of a person', 'this is a photo of a person.', 'a photo of a person.', 'this is a photo of a person', 'This is a photo of a person.', 'this is a photo of a single person.', 'a photo of a man and a woman', 'person posing for a photo']}, 'cartoon': {'dog': ['cartoon of a dog', 'cartoon of a dog.', 'a cartoon of a dog.', 'a cartoon of a dog', 'cartoon of a cute dog.', 'cartoon of a black and white dog.', 'cartoon of a cute dog', 'cartoon of a dog in a zoo'], 'elephant': ['cartoon of an elephant.', 'cartoon of an elephant in a zoo', 'cartoon of an elephant', 'cartoon of a giraffe and an elephant', 'cartoon of a giraffe and elephants', 'cartoon of elephants in a zoo', 'cartoon of a baby elephant in a zoo', 'cartoon of a giraffe and elephants in a zoo'], 'giraffe': ['cartoon of a giraffe in a zoo', 'cartoon of a giraffe at the zoo', 'a cartoon of a giraffe in a zoo', 'cartoon of a giraffe at a zoo', 'cartoon of a giraffe in the zoo', 'A cartoon of a giraffe in a zoo.', 'cartoon of a giraffe on a grassy field', 'A cartoon of a giraffe in a zoo enclosure.'], 'guitar': ['cartoon of a man playing a guitar', 'cartoon of a man playing guitar', 'cartoon of a boy playing a guitar', 'cartoon of a man playing a guitar.', 'cartoon of a young man playing a guitar', 'cartoon of a girl playing a guitar', 'cartoon of a man playing guitar.', 'cartoon of a young boy playing a guitar'], 'horse': ['cartoon of a horse', 'cartoon of a horse.', 'cartoon of a bald eagle on a horse', 'cartoon of a black and white horse', 'cartoon of a bald eagle grazing on a horse', 'cartoon of a giraffe and a horse', 'cartoon of a white horse', 'cartoon of a black and white horse.'], 'house': ['cartoon of a house.', 'cartoon of a house', 'cartoon of a house in a city', 'cartoon of the house.', 'cartoon of a man in a house', 'a cartoon of a house.', 'cartoon of a man in a house.', 'cartoon of a house in a city.'], 'person': ['cartoon of a person.', 'cartoon of a person', 'a cartoon of a person.', 'a cartoon of a person', 'person in a cartoon', 'cartoon of a person with a teddy bear.', 'cartoon of a single person.', 'a cartoon about a person.']}, 'sketch': {'dog': ['sketch of a dog', 'sketch of a dog.', 'a sketch of a dog.', 'a sketch of a dog', 'sketch of a cute dog', 'sketch of a cute dog.', 'sketch of a dog in a zoo', 'sketch of a dog in a park'], 'elephant': ['sketch of an elephant', 'sketch of an elephant.', 'sketch of an elephant in a zoo', 'A sketch of an elephant in a zoo.', 'sketch of a giraffe and elephant', 'a sketch of an elephant', 'sketch of a giraffe and elephants', 'a sketch of an elephant.'], 'giraffe': ['sketch of a giraffe in a zoo', 'sketch of a giraffe at the zoo', 'A sketch of a giraffe in a zoo.', 'sketch of a giraffe at a zoo', 'a sketch of a giraffe in a zoo', 'A sketch of a giraffe on a grassy field.', 'A sketch of a giraffe in a zoo enclosure.', 'a sketch of a giraffe at a zoo'], 'guitar': ['sketch of a guitar', 'sketch of a guitar.', 'A sketch of a man playing a guitar.', 'sketch of a man playing a guitar', 'sketch of a man playing guitar', 'sketch of a man playing a guitar.', 'Sketch of a man playing a guitar.', 'A sketch of a guitar.'], 'horse': ['sketch of a horse', 'sketch of a horse.', 'a sketch of a horse', 'a sketch of a horse.', 'sketch of a white horse', 'sketch of a bald eagle on a horse', 'sketch of a bald eagle grazing on a horse', 'sketch of a bald eagle riding on a horse'], 'house': ['sketch of a house', 'a sketch of a house', 'a sketch of a house.', 'sketch of a house.', 'sketch of the house', 'a sketch of the house.', 'sketch of the house.', 'sketch of a house in the countryside'], 'person': ['a sketch of a person.', 'a sketch of a person', 'sketch of a person.', 'this is a sketch of a person.', 'a sketch of a young person.', 'sketch of a person', 'portrait of a person in a sketch', 'a sketch of a young person']}}
        # Diverse
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            print(f'Current Image:{c}/{len(lines)}')
            c += 1
            # if c < 2000:
            #     continue

            toks = line.split(' ')
            img_path = toks[0]
            init_image = Image.open(img_path).resize((512, 512))
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            remain_domains = [i for i in self.args.source if i != cur_domain]
            aug_path_dict = {}

            for to_domain in remain_domains:
                template = aug_dict[to_domain]
                # to_text = f'{template} the {class_label}'
                to_text=random.choice(template[class_label])
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                print(class_label,to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=80, 
                        strength=0.9, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')

            link_dict.update({img_path:aug_path_dict})

        # print(c)
        outfile = f'./link/{cur_domain}_sd_autoprompt_conservative_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)

    def texture_intervene_vqgan(self):
        cur_domain = 'original'
        aug_lis =   ['coarse','clock','furry','furry','furry','furry','furry','furry','glass','metallic','metal','metal luster','metallic','metal luster','metallic','metal luster','metal luster','metal luster','metal luster','stained', 'striped', 'banded', 'marbled', 'paisley', 'pleated', 'fibrous', 'zigzagged', 'sprinkled', 'pitted', 'swirly', 'chequered', 'bubbly', 'cobwebbed', 'smeared', 'crystalline', 'lined', 'flecked', 'stratified', 'meshed', 'crosshatched', 'perforated', 'wrinkled', 'dotted', 'lacelike', 'woven', 'knitted', 'frilly', 'porous', 'grid', 'potholed', 'veined']
        aug_dict = {'rand_1':aug_lis,'rand_2':aug_lis,'rand_3':aug_lis,'rand_4':aug_lis}
        index_to_name = {0: 'airplane', 1: 'bear', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'dog', 7: 'cat', 8: 'car', 9: 'clock', 10: 'chair', 11: 'elephant', 12: 'keyboard', 13: 'knife', 14: 'oven', 15: 'truck'}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            # if c < 3200:
            #     continue
            toks = line.split(' ')
            # print(toks)
            img_path,index = toks[0],toks[1]
            class_label = index_to_name[int(index)]
            remain_domains = ['rand_1','rand_2','rand_3','rand_4']
            # init_image = Image.open(img_path).resize((512, 512))
            aug_path_dict = {}
            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                to_text = f'a {class_label} in a {template} background, {class_label}'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                # try:
                img_res= VQGAN_v2(img_path=img_path,model=self.model,perceptor=self.perceptor,device=self.device,texts=to_text,max_iterations=50,image_size=[224,224])[-1]
                img_res.save(out_path)
                aug_path_dict.update({to_domain:out_path})
                # except:
                #     aug_path_dict.update({to_domain:img_path})
                #     print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/Texture/{cur_domain}_vqgan_link_v4.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)     

    def texture_intervene_sd(self):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4',
        use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = 'original'
        # aug_lis =   ['coarse','furry','furry','furry','furry','furry','glass','metallic','metal','metal luster','metal luster','stained', 'striped', 'banded', 'marbled', 'polka-dotted', 'paisley', 'pleated', 'spiralled', 'fibrous', 'zigzagged', 'matted', 'braided', 'sprinkled', 'pitted', 'swirly', 'chequered', 'bubbly', 'cobwebbed', 'smeared', 'crystalline', 'studded', 'honeycombed', 'interlaced', 'lined', 'scaly', 'gauzy', 'flecked', 'stratified', 'meshed', 'crosshatched', 'perforated', 'freckled', 'wrinkled', 'bumpy', 'dotted', 'waffled', 'cracked', 'lacelike', 'blotchy', 'woven', 'knitted', 'frilly', 'porous', 'grid', 'grooved', 'potholed', 'veined']
        # aug_lis = ['black pen sketch','quickdraw sketch','grainy','surreal art','oil painting','fresco', 'naturalistic painting', 'stylised painting', 'watercolor painting', 'impressionist painting', 'cubist painting', 'expressionist painting','artistic painting']
        aug_lis = ['pointillism','rubin statue', 'rusty statue','ceramic','vaporwave','stained glass','wood statue','metal statue','bronze statue','iron statue','marble statue','stone statue','mosaic','furry','corel draw','simple sketch','stroke drawing', 'black ink painting','silhouette painting','black pen sketch','quickdraw sketch','grainy','surreal art','oil painting','fresco', 'naturalistic painting', 'stylised painting', 'watercolor painting', 'impressionist painting', 'cubist painting', 'expressionist painting','artistic painting']
        aug_dict = {'rand_1':aug_lis,'rand_2':aug_lis,'rand_3':aug_lis,'rand_4':aug_lis}
        index_to_name = {0: 'airplane', 1: 'bear', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'dog', 7: 'cat', 8: 'car', 9: 'clock', 10: 'chair', 11: 'elephant', 12: 'keyboard', 13: 'knife', 14: 'oven', 15: 'truck'}
        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            c += 1
            if c < 6400:
                continue
            toks = line.split(' ')
            # print(toks)
            img_path,index = toks[0],toks[1]
            class_label = index_to_name[int(index)]
            remain_domains = ['rand_1','rand_2','rand_3','rand_4']
            # remain_domains = ['rand_3']
            init_image = Image.open(img_path).resize((512, 512))
            aug_path_dict = {}
            for to_domain in remain_domains:
                template = random.choice(aug_dict[to_domain])
                # to_text = f'a photo of a {class_label}, {template} image'
                to_text = f'a {class_label}, {template} image'
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                print(to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=50, 
                        strength=0.9, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')
            link_dict.update({img_path:aug_path_dict})
            
        print(c)
        outfile = f'/homes/55/jianhaoy/projects/EKI/link/Texture/{cur_domain}_sd_v1_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)     

    def officehome_autoprompt_intervene(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            'CompVis/stable-diffusion-v1-4',
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = self.args.source[self.args.workeri] 
        # Conservative
        # aug_dict = json.load(open('/homes/55/jianhaoy/projects/EKI/link/officehome/autoprompt_conservative.json'))
        # moderate
        aug_dict = json.load(open('/homes/55/jianhaoy/projects/EKI/link/officehome/autoprompt_moderate.json'))

        file_name = f'{cur_domain}_train.txt'
        label_path = os.path.join(self.args.input_dir,file_name)
        f = open(label_path,'r')
        lines = f.readlines()
        c = 0
        link_dict = {}
        for line in tqdm(lines):
            print(f'Current Image:{c}/{len(lines)}')
            c += 1
            # if c < 2000:
            #     continue

            toks = line.split(' ')
            img_path = toks[0]
            init_image = Image.open(img_path).resize((512, 512))
            clo = img_path.split('/')[-2].replace('_', ' ')
            if clo == 'Ruler':
                clo = 'ruler'
            elif clo == 'ToothBrush':
                clo = 'Toothbrush'
            print(img_path,clo)
            class_label = img_path.split('/')[-2].replace('_', ' ').lower()
            remain_domains = [i for i in self.args.source if i != cur_domain]
            aug_path_dict = {}

            for to_domain in remain_domains:
                template = aug_dict[to_domain]
                # to_text = f'{template} the {class_label}'
                to_text=random.choice(template[clo])
                
                out_path = os.path.join(self.args.output_dir,f'{cur_domain}_to_{to_domain}_{class_label}_{c}.jpg')
                print(class_label,to_text,out_path)
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=80, 
                        strength=0.9, guidance_scale=7.5).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')

            link_dict.update({img_path:aug_path_dict})

        # print(c)
        outfile = f'./link/{cur_domain}_sd_autoprompt_moderate_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)
    
    def pacs_get_pair(self):
        for workeri in [0,1,2,3]:
            cur_domain = self.args.source[workeri] 
            # aug_dict = {'art_painting':['an art painting of'],'sketch':['a sketch of'],'cartoon':['a cartoon of'],'photo':['a photo of']}
            # aug_dict = {'art_painting': ['an oil painting of', 'a painting of', 'a fresco of', 'a colourful painting of', 'an abstract painting of', 'a naturalistic painting of', 'a stylised painting of', 'a watercolor painting of', 'an impressionist painting of', 'a cubist painting of', 'an expressionist painting of','an artistic painting of'], 'sketch':['an ink pen sketch of', 'a charcoal sketch of', 'a black and white sketch', 'a pencil sketch of', 'a rough sketch of', 'a kid sketch of', 'a notebook sketch of','a simple quick sketch of'], 'photo': ['a photo of', 'a picture of', 'a polaroid photo of', 'a black and white photo of', 'a colourful photo of', 'a realistic photo of'], 'cartoon': ['an anime drawing of', 'a cartoon of', 'a colorful cartoon of', 'a black and white cartoon of', 'a simple cartoon of', 'a disney cartoon of', 'a kid cartoon style of']}
            aug_dict = json.load(open('/homes/55/jianhaoy/projects/EKI/link/pacs/autoprompt_moderate.json'))

            file_name = f'{cur_domain}_train.txt'
            label_path = os.path.join(self.args.input_dir,file_name)
            f = open(label_path,'r')
            lines = f.readlines()
            c = 0
            link_dict = {}
            for line in tqdm(lines):
                print(f'Current Image:{c}/{len(lines)}')
                c += 1

                toks = line.split(' ')
                img_path = toks[0]
                init_image = Image.open(img_path).resize((512, 512))
                class_label = img_path.split('/')[-2].replace('_', ' ').lower()
                remain_domains = [i for i in self.args.source if i != cur_domain]
                aug_path_dict = {}

                for to_domain in remain_domains:
                    template = random.choice(aug_dict[to_domain][class_label])
                    to_text = f'{template} the {class_label}'
                    aug_path_dict.update({to_domain:to_text})

                link_dict.update({img_path:aug_path_dict})

        
        outfile = f'./link/handcrafted_pair.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)
        print(len(link_dict))

def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handler = InterventionHandler(args, config, device)
    # handler.pacs_sd_intervene()
    # handler.officehome_intervene_sd()
    # handler.officehome_vqgan_intervene()
    # handler.imagenet9_intervene_sd()
    # handler.imagenet9_intervene_vqgan()
    # handler.pacs_text_inversion()
    # handler.pacs_sd_multi_samples_intervene()
    # handler.imagenet9_intervene_sd_inpaint()
    # handler.celeba_sd_intervene()
    # handler.celeba_vqgan_intervene()
    # handler.celeba_sd_control_intervene()
    # handler.pacs_autoprompt_intervene()
    # handler.texture_intervene_vqgan()
    # handler.texture_intervene_sd()
    # handler.officehome_autoprompt_intervene()
    # handler.pacs_sd_adaptation_intervene()
    handler.pacs_get_pair()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()