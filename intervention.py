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

from data import *
from utils.Logger import Logger
from utils.tools import *

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


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

    def pacs_sd_intervene(self):
        # SD hyper-parameters
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            use_auth_token=True
        ).to('cuda')
        num_samples = 1

        cur_domain = self.args.source[self.args.workeri] 
        # Minimal Prompt
        aug_dict = {'art_painting':['an art painting of'],'sketch':['a sketch of'],'cartoon':['a cartoon of'],'photo':['a photo of']}
        # Hand-crafted Prompt
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
                if os.path.exists(out_path):
                    aug_path_dict.update({to_domain:out_path})
                    continue
                try:
                    image_res = pipe(
                        prompt=[to_text]*num_samples, init_image=init_image,
                        num_inference_steps=50, 
                        strength=0.75, guidance_scale=2.0).images[0]
                    image_res.save(out_path)
                    aug_path_dict.update({to_domain:out_path})
                except:
                    aug_path_dict.update({to_domain:img_path})
                    print(img_path,'Here Error?')

            link_dict.update({img_path:aug_path_dict})

        # print(c)
        outfile = f'./link/{cur_domain}_test_link.json'
        with open(outfile,'w') as f:
            json.dump(link_dict,f)

def main():
    args, config = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    handler = InterventionHandler(args, config, device)
    handler.pacs_sd_intervene()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
