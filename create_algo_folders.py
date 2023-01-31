import os

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:             
		os.makedirs(path)           

base_dirs = [f'./results/Trail{i}' for i in [1]]
algo_lis = ['OURS','ERM']

for base_dir in base_dirs:
	for i in algo_lis:
		f_name = os.path.join(base_dir,i)
		print(f_name)
		mkdir(f_name)    
