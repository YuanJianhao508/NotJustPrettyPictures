import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")
parser.add_argument("--root", default=None, type=str)
parser.add_argument("--workeri","-wi", default=None, type=int)

args = parser.parse_args()

###############################################################################

source = ['original','in','flip','random']


input_dir = '/datasets/jianhaoy/celebA/label_files'
input_dir = '/datasets/jianhaoy/celebA/new_label'
# output_dir = '/datasets/jianhaoy/CelebA_VQGAN/dumb'
# output_dir = '/datasets/jianhaoy/CelebA_SD/dumb'
output_dir = '/datasets/jianhaoy/CelebA_SD/new'

config = "Officehome/ResNet18"

workeri = args.workeri

##############################################################################

for i in range(args.times):
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python intervention.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--workeri {workeri} '
              f'--config {config}',)

