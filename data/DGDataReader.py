from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
from data.data_utils import *
import random
import cv2
import os
import torch
import torch.nn.functional as F
import random
import json
import numpy as np
from imagecorruptions import corrupt, get_corruption_names
import glob

from data.RandAugment import RandAugment
import data.pixmix_utils as pixmix_utils
from styleaug import StyleAugmentor

class DGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):
        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        if self.transformer is not None:
            img = self.transformer(img)
        label = self.labels[index]
        return img, label

class TestAdaptationDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):
        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        link_dict = '/homes/55/jianhaoy/projects/EKI/link/pacs/pacs_dumb_adaptation_noclass_link.json'
        self.link_dict = json.load(open(link_dict))
        self.target = self.args.target

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img_aug_name = self.link_dict[img_name][self.target]

        img = Image.open(img_aug_name).convert('RGB')
        if self.transformer is not None:
            img = self.transformer(img)
        label = self.labels[index]
        return img, label

def get_testadap_dataset(args, path, train=False, image_size=224, crop=False, jitter=0, config=None):
    names, labels = dataset_info(path)
    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    img_transform = get_img_transform(train, image_size, crop, jitter)
    return TestAdaptationDataset(args, names, labels, img_transform)


class FourierDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None, from_domain=None, alpha=1.0):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.from_domain = from_domain
        self.alpha = alpha
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)

        img_s, label_s, domain_s = self.sample_image(domain)
        img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img_s2o, img_o2s = self.post_transform(img_s2o), self.post_transform(img_o2s)
        img = [img_o, img_s, img_s2o, img_o2s]
        label = [label, label_s, label, label_s]
        domain = [domain, domain_s, domain, domain_s]
        return img, label, domain

    def sample_image(self, domain):
        if self.from_domain == 'all':
            domain_idx = random.randint(0, len(self.names)-1)
        elif self.from_domain == 'inter':
            domains = list(range(len(self.names)))
            domains.remove(domain)
            domain_idx = random.sample(domains, 1)[0]
        elif self.from_domain == 'intra':
            domain_idx = domain
        else:
            raise ValueError("Not implemented")
        img_idx = random.randint(0, len(self.names[domain_idx])-1)
        img_name_sampled = self.names[domain_idx][img_idx]
        img_name_sampled = os.path.join(self.args.input_dir, img_name_sampled)
        img_sampled = Image.open(img_name_sampled).convert('RGB')
        label_sampled = self.labels[domain_idx][img_idx]
        return self.transformer(img_sampled), label_sampled, domain_idx



def get_dataset(args, path, train=False, image_size=224, crop=False, jitter=0, config=None):
    names, labels = dataset_info(path)
    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    img_transform = get_img_transform(train, image_size, crop, jitter)
    return DGDataset(args, names, labels, img_transform)


def get_fourier_dataset(args, path, image_size=224, crop=False, jitter=0, from_domain='all', alpha=1.0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
        from_domain = config["from_domain"]
        alpha = config["alpha"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return FourierDGDataset(args, names, labels, img_transform, from_domain, alpha)


### Customize for SemANTIC Augmentation
class SemAugDGDataset(Dataset):
    def __init__(self, args, names, labels, target=None, transformer=None, from_domain=None, link_dict=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.from_domain = from_domain
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

        self.link_dict = json.load(open(link_dict))
        self.target = target

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)

        img_s = self.sample_aug_image(img_name)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img = [img_o, img_s]
        label = [label, label]
        domain = [domain, domain]
        return img, label, domain

    def sample_aug_image(self, img_name):
        # print(self.from_domain == 'all')
        
        if self.from_domain == 'all':
            img_name_sampled = random.choice([*self.link_dict[img_name].values()])
        elif self.from_domain == 'all_m':
            img_name_sampled = random.choice(random.choice([*self.link_dict[img_name].values()]))
        elif self.from_domain == 'targeted':
            aug_ref = [self.link_dict[img_name][i] for i in self.link_dict[img_name].keys() if i != self.args.target]
            img_name_sampled = random.choice(aug_ref)
        elif self.from_domain == 'teston':
            aug_ref = [self.link_dict[img_name][i] for i in self.link_dict[img_name].keys() if i != self.args.teston]
            img_name_sampled = random.choice(aug_ref)

        else:
            raise ValueError("Not implemented")
        # print(img_name_sampled)
        img_sampled = Image.open(img_name_sampled).convert('RGB')
        return self.transformer(img_sampled)

def get_semaug_dataset(args, path, image_size=224, crop=False, jitter=0, from_domain='all', alpha=1.0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
        # from_domain = config["from_domain"]
        from_domain = from_domain

    print("Sample Strategy:",from_domain)
    try:    
        target = args.test_domain
    except:
        target = None

    link_dict = args.link_dict
    img_transform = get_pre_transform(image_size, crop, jitter)
    return SemAugDGDataset(args, names, labels, target, img_transform, from_domain, link_dict)

# For FID use
class GenDGAugOnlyDataset(Dataset):
    def __init__(self, args, names, labels, target=None, transformer=None, from_domain=None, link_dict=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.from_domain = from_domain
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

        self.link_dict = json.load(open(link_dict))
        self.target = target

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img_s = self.sample_aug_image(img_name)
        img_s = self.post_transform(img_s)
        img = [img_s]
        label = [label]
        domain = [domain]
        return img, label, domain

    def sample_aug_image(self, img_name):
        
        if self.from_domain == 'all':
            img_name_sampled = self.link_dict[img_name][self.target]
        else:
            raise ValueError("Not implemented")

        img_sampled = Image.open(img_name_sampled).convert('RGB')
        return self.transformer(img_sampled)    


def get_augonly_dataset(args, path, target_test, image_size=224, crop=False, jitter=0, from_domain='all', alpha=1.0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = False
        jitter = 0
        from_domain = config["from_domain"]

    print("Sample Strategy:",from_domain)


    link_dict = args.link_dict
    img_transform = get_pre_transform(image_size, crop, jitter)
    return GenDGAugOnlyDataset(args, names, labels, target_test, img_transform, from_domain, link_dict)

class NormalDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()

        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)


    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        # print(img_name,label)
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img = self.transformer(img)
        img = self.post_transform(img)
        img = [img]
        label = [label]
        domain = [domain]
        return img, label, domain


def get_normal_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return NormalDGDataset(args, names, labels, img_transform)

#AugMix
class AugMixDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

        self.aug_prob_coeff= 1.
        self.mixture_width = 3
        self.mixture_depth = -1
        self.aug_severity = 1

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB').resize((224,224),Image.ANTIALIAS)

        img_s1 = self.augmix(img)
        img_s2 = self.augmix(img)
        img_o = self.transformer(img)

        img_o, img_s1, img_s2 = self.post_transform(img_o), self.post_transform(img_s1),self.post_transform(img_s2)
        img = [img_s1, img_s2]
        label = [label, label]
        domain = [domain, domain]
        return img, label, domain

    def augmix(self,x):
        aug_list = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y
        ]

        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = np.zeros_like(np.array(x)).astype("float32")
        for i in range(self.mixture_width):
            x_ = x.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                x_ = op(x_, self.aug_severity)

            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * np.array(x_).astype("float32")

        mix = Image.fromarray(mix.astype("uint8"))
        x_ = Image.blend(x, mix, m)

        return self.transformer(x_)

def get_augmix_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return AugMixDGDataset(args, names, labels, img_transform)

# MixUp
class MixUpDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)

        img_s, label_s, domain_s = self.sample_image()
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img = [img_o, img_s]
        label = [label, label_s]
        domain = [domain, domain_s]
        return img, label, domain

    def sample_image(self):
        domain_idx = random.randint(0, len(self.names)-1)
        img_idx = random.randint(0, len(self.names[domain_idx])-1)
        img_name_sampled = self.names[domain_idx][img_idx]
        img_name_sampled = os.path.join(self.args.input_dir, img_name_sampled)
        img_sampled = Image.open(img_name_sampled).convert('RGB')
        label_sampled = self.labels[domain_idx][img_idx]
        return self.transformer(img_sampled), label_sampled, domain_idx

def get_mixup_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return MixUpDGDataset(args, names, labels, img_transform)

# StyleAug
class StyleAugmentation:
    def __init__(self):
        self.augmentor = StyleAugmentor()

    def __call__(self, img):
        return self.augmentor(img)

class StyleAugDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = self.transformer(Image.open(img_name).convert('RGB'))
        img_s = self.transformer(Image.open(img_name).convert('RGB'))
        # img = Image.open(img_name).convert('RGB').resize((224,224),Image.ANTIALIAS)
        img_o = self.post_transform(img)
        img_s = self.post_transform(img_s)
        img = [img_o,img_s]
        label = [label,label]
        domain = [domain,domain]
        return img, label, domain

def get_styleaug_post_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        StyleAugmentation(),
        transforms.Normalize(mean, std)
    ])
    return img_transform

def get_styleaug_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return StyleAugDGDataset(args, names, labels, img_transform)


# RandAug
class RandAugDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.post_transform.transforms.insert(0, RandAugment(1, 2))
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        # img = self.transformer(Image.open(img_name).convert('RGB'))
        img = Image.open(img_name).convert('RGB').resize((224,224),Image.ANTIALIAS)
        img_o = self.post_transform(img)
        img_s = self.post_transform(img)
        img = [img_o,img_s]
        label = [label,label]
        domain = [domain,domain]
        return img, label, domain

def get_randaug_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return RandAugDGDataset(args, names, labels, img_transform)

# RandAug
class RandAugDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.post_transform.transforms.insert(0, RandAugment(1, 2))
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        # img = self.transformer(Image.open(img_name).convert('RGB'))
        img = Image.open(img_name).convert('RGB').resize((224,224),Image.ANTIALIAS)
        img_o = self.post_transform(img)
        img_s = self.post_transform(img)
        img = [img_o,img_s]
        label = [label,label]
        domain = [domain,domain]
        return img, label, domain

def get_stylemix_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return RandAugDGDataset(args, names, labels, img_transform)

# CutOut
class CutOutDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)
        img_s = self.cutout(img_o)
        img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
        img = [img_o, img_s]
        label = [label, label]
        domain = [domain, domain]
        return img, label, domain

    def get_random_eraser(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0.0, v_h=1.0):
        """
        This CutOut implementation is taken from:
            - https://github.com/yu4u/cutout-random-erasing
        ...and modified for info loss experiments
        # Arguments:
            :param p: (float) the probability that random erasing is performed
            :param s_l: (float) minimum proportion of erased area against input image
            :param s_h: (float) maximum proportion of erased area against input image
            :param r_1: (float) minimum aspect ratio of erased area
            :param r_2: (float) maximum aspect ratio of erased area
            :param v_l: (float) minimum value for erased area
            :param v_h: (float) maximum value for erased area
            :param fill: (str) fill-in mode for the cropped area
        :return: (np.array) augmented image
        """
        def eraser(orig_img):
            input_img = np.copy(orig_img)
            if input_img.ndim == 3:
                img_h, img_w, img_c = input_img.shape
            elif input_img.ndim == 2:
                img_h, img_w = input_img.shape

            p_1 = np.random.rand()

            if p_1 > p:
                return input_img

            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            input_img[top:top + h, left:left + w] = 0

            return input_img

        return eraser

    def cutout(self, x):

        eraser = self.get_random_eraser()
        x_ = eraser(x)

        return x_

def get_cutout_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return CutOutDGDataset(args, names, labels, img_transform)

#ACVC
class ACVCDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)
        img_s = self.corruption(img_o)
        img_o = self.post_transform(img_o)
        img_s = self.post_transform(img_s)
        img = [img_o,img_s]
        label = [label,label]
        domain = [domain,domain]
        return img, label, domain

    def corruption(self, x):

        x_ = np.copy(x)
        x_ = self.acvc(x_)

        return x_
    
    def acvc(self, x):
        i = np.random.randint(0, 22)
        corruption_func = {0: "fog",
                           1: "snow",
                           2: "frost",
                           3: "spatter",
                           4: "zoom_blur",
                           5: "defocus_blur",
                           6: "glass_blur",
                           7: "gaussian_blur",
                           8: "motion_blur",
                           9: "speckle_noise",
                           10: "shot_noise",
                           11: "impulse_noise",
                           12: "gaussian_noise",
                           13: "jpeg_compression",
                           14: "pixelate",
                           15: "elastic_transform",
                           16: "brightness",
                           17: "saturate",
                           18: "contrast",
                           19: "high_pass_filter",
                           20: "constant_amplitude",
                           21: "phase_scaling"}
        return self.apply_corruption(x, corruption_func[i])

    def apply_corruption(self, x, corruption_name):
        severity = np.random.randint(1, 6)

        custom_corruptions = {"high_pass_filter": self.high_pass_filter,
                              "constant_amplitude": self.constant_amplitude,
                              "phase_scaling": self.phase_scaling}

        if corruption_name in get_corruption_names('all'):
            x = corrupt(x, corruption_name=corruption_name, severity=severity)
            x = Image.fromarray(x)

        elif corruption_name in custom_corruptions:
            x = custom_corruptions[corruption_name](x, severity=severity)

        else:
            assert True, "%s is not a supported corruption!" % corruption_name

        return x

    def draw_cicle(self, shape, diamiter):
        """
        Input:
        shape    : tuple (height, width)
        diameter : scalar
        Output:
        np.array of shape  that says True within a circle with diamiter =  around center
        """
        assert len(shape) == 2
        TF = np.zeros(shape, dtype="bool")
        center = np.array(TF.shape) / 2.0

        for iy in range(shape[0]):
            for ix in range(shape[1]):
                TF[iy, ix] = (iy - center[0]) ** 2 + (ix - center[1]) ** 2 < diamiter ** 2
        return TF

    def filter_circle(self, TFcircle, fft_img_channel):
        temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
        temp[TFcircle] = fft_img_channel[TFcircle]
        return temp

    def inv_FFT_all_channel(self, fft_img):
        img_reco = []
        for ichannel in range(fft_img.shape[2]):
            img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:, :, ichannel])))
        img_reco = np.array(img_reco)
        img_reco = np.transpose(img_reco, (1, 2, 0))
        return img_reco

    def high_pass_filter(self, x, severity):
        x = x.astype("float32") / 255.
        c = [.01, .02, .03, .04, .05][severity - 1]

        d = int(c * x.shape[0])
        TFcircle = self.draw_cicle(shape=x.shape[:2], diamiter=d)
        TFcircle = ~TFcircle

        fft_img = np.zeros_like(x, dtype=complex)
        for ichannel in range(fft_img.shape[2]):
            fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(x[:, :, ichannel]))

        # For each channel, pass filter
        fft_img_filtered = []
        for ichannel in range(fft_img.shape[2]):
            fft_img_channel = fft_img[:, :, ichannel]
            temp = self.filter_circle(TFcircle, fft_img_channel)
            fft_img_filtered.append(temp)
        fft_img_filtered = np.array(fft_img_filtered)
        fft_img_filtered = np.transpose(fft_img_filtered, (1, 2, 0))
        x = np.clip(np.abs(self.inv_FFT_all_channel(fft_img_filtered)), a_min=0, a_max=1)

        x = Image.fromarray((x * 255.).astype("uint8"))
        return x

    def constant_amplitude(self, x, severity):
        """
        A visual corruption based on amplitude information of a Fourier-transformed image
        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.05, .1, .15, .2, .25][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Amplitude replacement
        beta = 1.0 - c
        x_abs = np.ones_like(x_abs) * max(0, beta)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = Image.fromarray((x * 255.).astype("uint8"))
        return x

    def phase_scaling(self, x, severity):
        """
        A visual corruption based on phase information of a Fourier-transformed image
        Adopted from: https://github.com/MediaBrain-SJTU/FACT
        """
        x = x.astype("float32") / 255.
        c = [.1, .2, .3, .4, .5][severity - 1]

        # FFT
        x_fft = np.fft.fft2(x, axes=(0, 1))
        x_abs, x_pha = np.fft.fftshift(np.abs(x_fft), axes=(0, 1)), np.angle(x_fft)

        # Phase scaling
        alpha = 1.0 - c
        x_pha = x_pha * max(0, alpha)

        # Inverse FFT
        x_abs = np.fft.ifftshift(x_abs, axes=(0, 1))
        x = x_abs * (np.e ** (1j * x_pha))
        x = np.real(np.fft.ifft2(x, axes=(0, 1)))

        x = Image.fromarray((x * 255.).astype("uint8"))
        return x

def get_acvc_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return ACVCDGDataset(args, names, labels, img_transform)

# PixMix
class PixMixDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.tensorize = get_tensorize()
        self.normalize = get_normalize()
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

        self.mixing_name_list = self.get_mixing_name_list('/datasets/jianhaoy/fractals_and_fvis/fractals/images')

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]

        mixing_pic_name = random.choice(self.mixing_name_list)
        mixing_pic = Image.open(mixing_pic_name).convert('RGB').resize((224,224),Image.ANTIALIAS)
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB').resize((224,224),Image.ANTIALIAS)

        img_s = self.pixmix(img,mixing_pic)
        img_o, img_s = self.post_transform(img), img_s
        img = [img_o, img_s]
        label = [label, label]
        domain = [domain, domain]
        return img, label, domain

    def pixmix(self, orig, mixing_pic):
  
        mixings = pixmix_utils.mixings
        if np.random.random() < 0.5:
            mixed = self.tensorize(self.augment_input(orig))
        else:
            mixed = self.tensorize(orig)
        
        for _ in range(np.random.randint(4 + 1)):
            
            if np.random.random() < 0.5:
                aug_image_copy = self.tensorize(self.augment_input(orig))
            else:
                aug_image_copy = self.tensorize(mixing_pic)

            mixed_op = np.random.choice(mixings)
            mixed = mixed_op(mixed, aug_image_copy, 4)
            mixed = torch.clip(mixed, 0, 1)

        return self.normalize(mixed)

    def augment_input(self,image):
        aug_list = pixmix_utils.augmentations_all
        op = np.random.choice(aug_list)
        return op(image.copy(), 1)

    def get_mixing_name_list(self,root):
        paths = glob.glob(os.path.join(root, '*.jpg'))
        return paths

def get_pixmix_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return PixMixDGDataset(args, names, labels, img_transform)

# Full Ours
class FullOursDGDataset(ACVCDGDataset):
    def __init__(self, args, names, labels, transformer=None, link_dict=None):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

        self.link_dict = json.load(open(link_dict))

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]
        img_name = os.path.join(self.args.input_dir, img_name)
        img = Image.open(img_name).convert('RGB')
        img_o = self.transformer(img)
        img_s = self.sem_aug(img_name)
        img_o = self.corruption(img_o)
        img_s = self.corruption(img_s)
        img_o = self.post_transform(img_o)
        img_s = self.post_transform(img_s)
        img = [img_o,img_s]
        label = [label,label]
        domain = [domain,domain]
        return img, label, domain

    def sem_aug(self, img_name):
        img_name_sampled = random.choice([*self.link_dict[img_name].values()])
        img_sampled = Image.open(img_name_sampled).convert('RGB')
        return self.transformer(img_sampled)

def get_fullours_dataset(args, path, image_size=224, crop=False, jitter=0, config=None):
    assert isinstance(path, list)
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    link_dict = args.link_dict
    img_transform = get_pre_transform(image_size, crop, jitter)
    return FullOursDGDataset(args, names, labels, img_transform,link_dict)