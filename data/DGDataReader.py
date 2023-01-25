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
