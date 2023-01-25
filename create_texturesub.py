import os
import difflib
import sys 
# sys.path.append("..") 
import generative_model.clip as clip
import torch
from torch import nn
import numpy as np



clip_model = "ViT-B/32"
device ="cuda:0"
perceptor, preprocess = clip.load(clip_model, jit=False)
perceptor.eval().requires_grad_(False).to(device)
cosim = nn.CosineSimilarity(dim=1, eps=1e-6)

map_file = open('/homes/55/jianhaoy/projects/EKI/script/imagnet_map.txt','r')
wanted = ['airplane','bear','bicycle','bird','boat','bottle','dog','cat','car','clock','chair','elephant','keyboard','knife','oven','truck']

lines = map_file.readlines()
map_dic = {}
name_lis = []
for line in lines:
    toks = line.split(' ')
    index, name = toks[0],toks[-1].strip('\n')
    # print(name)
    map_dic.update({index:name})
# print(map_dic)

reverse_dic = {}
for key,val in map_dic.items():
    reverse_dic.update({val:key})
# print(reverse_dic)

# from_dir = '/datasets/jianhaoy/Texture/train'

# g = os.walk(from_dir)  


# dir_lis_tmp = []
# for path,dir_list,file_list in g:  
#     # print(dir_list,file_list)
#     dir_lis_tmp = file_list
#     break

# name_lis = []
# filtered_dir_list = []
# for index in dir_lis_tmp:
#     index = index.strip('.tar')
#     if 'ILS' in index:
#         continue
#     # try:
#     name = map_dic[index].strip('\n')
#     name_lis.append(name)

# # print(name_lis)

# want_map = {}
# word_set = list(set(name_lis))
# # print(word_set)
# set_emb = []
# for word in word_set:
#     text = clip.tokenize(word).to(device)
#     with torch.no_grad():
#         text_embed = perceptor.encode_text(text)
#     set_emb.append(text_embed)
# # print(set_emb)
# want_emb = []
# for word in wanted:
#     text = clip.tokenize(word).to(device)
#     with torch.no_grad():
#         text_embed = perceptor.encode_text(text)
#     want_emb.append(text_embed)

# mapped = {}
# for idx in range(len(want_emb)):
#     target = want_emb[idx]
#     res = []
#     for candidate in set_emb:
#         out = cosim(target,candidate)
#         res.append(out.cpu().numpy())
#     res = np.array(res).reshape(-1)
#     # print(res)
#     p = list(res)
#     topk = 8
#     index_list = sorted(range(len(p)), key=lambda i: p[i])[-topk:] # sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:2]
#     pro_list = np.array(p)[index_list]
#     topk = []
#     for i in index_list:
#         topk.append(word_set[i])
#     # print("name:",topk)
#     mapped.update({wanted[idx]:topk})

# print(mapped)

cleaned_mapped = {'airplane': ['warplane', 'airliner', 'plane'], 
'bear': ['American_black_bear', 'brown_bear'], 
'bicycle': ['bicycle-built-for-two', 'mountain_bike'], 
'bird': ['bulbul','coucal','jacamar','toucan'], 
'boat': ['dock', 'canoe', 'lifeboat', 'speedboat'], 
'bottle': ['beer_bottle', 'wine_bottle'], 
'dog': ['pug','dingo','Siberian_husky','African_hunting_dog'], 
'cat': ['Egyptian_cat', 'Madagascar_cat', 'Persian_cat'], 
'car': ['sports_car', 'cab'], 
'clock': ['wall_clock', 'analog_clock'], 
'chair': ['barber_chair', 'rocking_chair'], 
'elephant': ['African_elephant', 'Indian_elephant'], 
'keyboard': ['computer_keyboard'], 
'knife': ['cleaver'], 
'oven': ['microwave', 'stove'], 
'truck': ['tow_truck', 'trailer_truck']}

tar_dic = {}
for key,val in cleaned_mapped.items():
    cont = []
    for name in val:
        cont.append(reverse_dic[name])
    tar_dic.update({key:cont})
# print(tar_dic)

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:             
		os.makedirs(path)       

#### get command 
base_dir = './train'
out_dir = '/datasets/jianhaoy/Texture/texture'

# for key,val in tar_dic.items():
#     for name in val:
#         file_name = f'{name}.tar'
#         tar_path = os.path.join(base_dir,file_name)
#         out_path = os.path.join(out_dir,name)
#         out_v_path = os.path.join('./texture',name)
#         mkdir(out_path)
#         command = f'tar -xvf {tar_path} -C {out_v_path}'
#         print(command)


name_to_index = {}
for i in range(16):
    name_to_index.update({wanted[i]:i})
print(name_to_index)

index_to_name = {}
for key,val in name_to_index.items():
    index_to_name.update({val:key})
print(index_to_name)


# Sample
label_dir = '/datasets/jianhaoy/Texture/dataset/reduced2'
line_dict = {}
for key,val in tar_dic.items():
    line_lis = []
    lim = int(1200/len(val))
    # print(lim)
    for name in val:
        c = 0
        out_path = os.path.join(out_dir,name)
        for path,dir_list,file_list in os.walk(out_path):
            for img_name in file_list:  
                img_path = os.path.join(out_path,img_name)
                if c < lim:
                    line_lis.append(f'{img_path} {name_to_index[key]}')
                    c += 1
                else:
                    continue
    line_dict.update({key:line_lis})

out_paths = []
for split in ['train','val','test']:
    out_file = f'original_{split}.txt'
    out_path = os.path.join(label_dir,out_file)
    if os.path.exists(out_path):
        os.remove(out_path)
    # print(out_path)
    out_paths.append(out_path)

[train_out,val_out,test_out] = out_paths
# lim_train,lim_val,lim_test = 300,500,600
lim_train,lim_val,lim_test = 200,400,500

with open(train_out, "a", encoding="utf-8") as f1:
    with open(val_out, "a", encoding="utf-8") as f2:
        with open(test_out, "a", encoding="utf-8") as f3:
            for key,val in line_dict.items():
                c = 0
                for line in val:
                    if c < lim_train:
                        f1.write(f'{line}\n')
                        c+=1
                    elif c < lim_val:
                        f2.write(f'{line}\n')
                        c+=1
                    elif c < lim_test:
                        f3.write(f'{line}\n')
                        c+=1
                    else:
                        break


test_dirs = ["/datasets/jianhaoy/Texture/test/style-transfer-preprocessed-512","/datasets/jianhaoy/Texture/edges","/datasets/jianhaoy/Texture/filled-silhouettes"]
out_files = ['random_test.txt','edge_test.txt','silo_test.txt']
for idx in range(3):
    test_dir = test_dirs[idx]
    out_file = out_files[idx]
    out_path = os.path.join(label_dir,out_file)
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path, "a", encoding="utf-8") as f4:
        for target in wanted:
            folder = os.path.join(test_dir,target)
            for path,dir_list,file_list in os.walk(folder):
                # print(path,dir_list,file_list)
                toks = path.split('/')[-1]
                index = name_to_index[toks]
                for img_name in file_list:        
                    img_path = os.path.join(path,img_name)
                    line = f'{img_path} {index}\n'
                    # print(line)
                    f4.write(line)




























# mapped = {'airplane','bear','bicycle','bird','boat','bottle','dog','cat','car','clock','chair','elephant','keyboard','knife','oven','truck'}

# for class_index in filtered_dir_list:
#     if '_' in class_index:
#         continue 
#     class_name = map_dic[class_index]
#     print(class_name)
    
#     one_class_dir = os.path.join(from_dir,class_index)
#     for path,dir_list,file_list in os.walk(one_class_dir):
#         for img_name in file_list:  
#             img_path = os.path.join(one_class_dir,img_name)
#             print(img_path)
#     break