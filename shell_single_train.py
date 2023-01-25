import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain", "-d", default="sketch", help="Pick One Domain As Train")
parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")
parser.add_argument("--algo","-a",default='OURS',type=str,help="Choose Algorithm")
parser.add_argument("--test_domain", "-td", default="photo", type=str, help="Test on a Single Domain")
parser.add_argument("--ifcons","-ic",default="ceclip",type=str,help="Which loss to use")
parser.add_argument("--link_path",'-lp',default='/homes/55/jianhaoy/projects/CIRL/pacs_new_link.json',type=str,help='Link Dict Path')


args = parser.parse_args()

###############################################################################
# In single case target is train domain
source = ["photo", "cartoon", "art_painting", "sketch"]
target = args.domain
source.remove(target)

algo = args.algo

input_dir = '/datasets/jianhaoy/PACSC/updated_label_files'
output_dir = f'/datasets/jianhaoy/PACS_TEST/{algo}'


config = "PACS/ResNet18_step"

domain_name = target
test_domain = args.test_domain
path = os.path.join(output_dir, config.replace("/", "_"), domain_name)
ifcons = args.ifcons

# link_dict = '/homes/55/jianhaoy/projects/CIRL/new_link.json'
link_dict = args.link_path

##############################################################################

for i in range(args.times):
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python ./algos/aug_single_{algo}.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--link_dict {link_dict} '
              f'--test_domain {test_domain} '
              f'--ifcons {ifcons} ' 
              f'--config {config}',)

