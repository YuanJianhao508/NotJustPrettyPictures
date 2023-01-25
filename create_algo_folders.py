import os

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:             
		os.makedirs(path)           


# base_dir = '/homes/55/jianhaoy/projects/EKI/results_ImageNet9/test1'
# algo_lis = ['OURS','ERM','AugMix','RandAugment','CutOut','MixUp','CutMix','RSC','MEADA','ACVC','PixMix']

# base_dirs = [f'/homes/55/jianhaoy/projects/EKI/results_CelebA/E30_PRET_CE{i}' for i in [1,2,3,4,5]]
# base_dirs = [f'/homes/55/jianhaoy/projects/EKI/results_Texture/ForTest']
# algo_lis = ['OURS','ERM','AugMix','RandAugment','CutOut','MixUp','CutMix','RSC','MEADA','ACVC','PixMix','L2D','SN','SRN']
# algo_lis = ['OURS']
# base_dirs = [f'/homes/55/jianhaoy/projects/EKI/results_Officehome/Minimal/{i}' for i in [1,2,3,4,5]]

# for base_dir in base_dirs:
# 	for i in algo_lis:
# 		f_name = os.path.join(base_dir,i)
# 		mkdir(f_name)    

# base_dir = '/homes/55/jianhaoy/projects/EKI/results/Optim'
# algo_lis = ['adam','adamw','radam','rmsprop','sgd']

# base_dir = '/homes/55/jianhaoy/projects/EKI/results/multi_iter/engi'
# algo_lis = ['OURS_20','OURS_40','OURS_60','OURS_80','OURS_100','OURS_120','OURS_140','OURS_160','OURS_180','OURS_200','OURS_220','OURS_240','OURS_260','OURS_280','OURS_300']

# base_dir = '/homes/55/jianhaoy/projects/EKI/results/multi_iter/measure'
# algo_lis = ['VGG-gram','ViT-cosim']
# source_lis = ['art_painting','photo','sketch','cartoon']

# base_dir = '/homes/55/jianhaoy/projects/EKI/results/OURS_SD'

base_dirs = [f'/homes/55/jianhaoy/projects/EKI/results/Crossval/{i}' for i in [1,2,3,4,5]]
algo_lis = ['OURS','ERM','AugMix','RandAugment','CutOut','MixUp','CutMix','RSC','MEADA','ACVC','PixMix','L2D','SN','SRN']
# algo_lis = ['OURS']

for base_dir in base_dirs:
	for i in algo_lis:
		f_name = os.path.join(base_dir,i)
		print(f_name)
		mkdir(f_name)    

# base_dir = '/homes/55/jianhaoy/projects/EKI/results/Multi_results/'
# algo_lis = [f'{i}/{j}' for i in ["Exclude","Include"] for j in ["photo", "cartoon", "art_painting", "sketch"]]
# # source_lis = ['Art','Clipart','Product','Real_World']
# print(algo_lis)

# for i in algo_lis:
# 	f_dir = os.path.join(base_dir,i)
# 	mkdir(f_dir)
	# for j in source_lis:
	# 	f_name = os.path.join(f_dir,j)
	# 	mkdir(f_name)

# base_dir = '/homes/55/jianhaoy/projects/EKI/results/Multi_results/CIRL_2/Plain2'
# algo_lis = ["photo", "cartoon", "art_painting", "sketch"]
# for i in algo_lis:
# 	f_name = os.path.join(base_dir,i)
# 	# print(f_name)
# 	mkdir(f_name)    