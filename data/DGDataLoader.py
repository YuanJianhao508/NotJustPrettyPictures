import os
from torch.utils.data import DataLoader

from data.DGDataReader import *
from data.ConcatDataset import ConcatDataset
from utils.tools import *

default_input_dir = 'path/to/datalists/'

digits_datset = ["mnist", "mnist_m", "svhn", "syn"]
pacs_dataset = ["art_painting", "cartoon", "photo", "sketch"]
officehome_dataset = ['Art', 'Clipart', 'Product', 'Real_World']
waterbird_dataset = ['land','water']
domainnet_dataset = ["clipart","infograph","painting","quickdraw","real","sketch"]
imagenet9_dataset = ['original','mixed_rand','mixed_same']
celebA_dataset = ['original','in','flip','random']
texture_dataset = ['original','in','random','edge','silo']
nico_dataset = ["autumn","dim","grass","outdoor","rock","water"]
available_datasets = pacs_dataset + officehome_dataset + digits_datset + waterbird_dataset + domainnet_dataset + imagenet9_dataset + celebA_dataset + texture_dataset + nico_dataset


def get_datalists_folder(args=None):
    datalists_folder = default_input_dir
    if args is not None:
        if args.input_dir is not None:
            datalists_folder = args.input_dir
    return datalists_folder


def get_train_dataloader(source_list=None, batch_size=64, image_size=224, crop=False, jitter=0, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        train_dataset = get_dataset(args=args,
                                    path=path,
                                    train=True,
                                    image_size=image_size,
                                    crop=crop,
                                    jitter=jitter,
                                    config=data_config)
        datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

def get_train_double_dataloader(source_list=None, batch_size=64, image_size=224, crop=False, jitter=0, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        train_dataset = get_dataset(path=path,
                                    train=True,
                                    image_size=image_size,
                                    crop=crop,
                                    jitter=jitter,
                                    config=data_config)
        datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader



def get_fourier_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        from_domain='all',
        alpha=1.0,
        config=None
):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)

    paths = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        paths.append(path)
    print("Train:",paths)
    dataset = get_fourier_dataset(args=args,
                                  path=paths,
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  from_domain=from_domain,
                                  alpha=alpha,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

def get_copy_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        from_domain='all',
        alpha=1.0,
        config=None
):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)

    paths = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        paths.append(path)
    dataset = get_copy_dataset(args=args,
                                  path=paths,
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  from_domain=from_domain,
                                  alpha=alpha,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader



def get_val_dataloader(source_list=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_val.txt' % dname)
        val_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        datasets.append(val_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader


def get_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_test.txt' % target)
    test_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
    dataset = ConcatDataset([test_dataset])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader



###Added
def get_semaug_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        from_domain='all',
        alpha=1.0,
        config=None
):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)

    paths = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        paths.append(path)
    dataset = get_semaug_dataset(args=args,
                                  path=paths,
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  from_domain=from_domain,
                                  alpha=alpha,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

def get_intervention_dataloader(source_list=None, batch_size=1, image_size=224, crop=False, jitter=0, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    print(source_list)
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_train.txt' % dname)
        train_dataset = get_dataset(args=args,
                                    path=path,
                                    train=True,
                                    image_size=image_size,
                                    crop=crop,
                                    jitter=jitter,
                                    config=data_config)
        datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False, drop_last=True)
    return loader

def get_wb_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_test.txt' % dname)
        val_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        datasets.append(val_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader


### For single domain generalization
def get_single_semaug_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        from_domain='all',
        alpha=1.0,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_semaug_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  from_domain=from_domain,
                                  alpha=alpha,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

def get_single_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_normal_dataset(args=args,
                            path=[path],
                            image_size=image_size,
                            crop=crop,
                            jitter=jitter,
                            config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

def get_erm_dataset(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_normal_dataset(args=args,
                            path=[path],
                            image_size=image_size,
                            crop=crop,
                            jitter=jitter,
                            config=data_config)

def get_single_val_dataloader(source_list=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_val.txt' % target)
    val_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
    dataset = ConcatDataset([val_dataset])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader

def get_adaptation_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_test.txt' % dname)
        val_dataset = get_testadap_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        datasets.append(val_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader

def get_single_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    datasets = []
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_test.txt' % dname)
        val_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        datasets.append(val_dataset)
    dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return loader

def get_separate_adap_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
        # print(source_list)
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    loaders = {}
    for dname in source_list:
        # print(dname)
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_test.txt' % dname)
        val_dataset = get_testadap_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        dataset = ConcatDataset([val_dataset])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        loaders.update({f'adap_{dname}':loader})
    return loaders

def get_separate_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    loaders = {}
    for dname in source_list:
        # print(dname)
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_test.txt' % dname)
        val_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        dataset = ConcatDataset([val_dataset])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        loaders.update({dname:loader})
    return loaders

def get_separate_all_test_loader(target=None, batch_size=64, image_size=224, args=None, config=None):
    if args is not None:
        source_list = args.source
        source_list.append(args.target)
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None
    assert isinstance(source_list, list)
    loaders = {}
    for dname in source_list:
        datalists_folder = get_datalists_folder(args)
        path = os.path.join(datalists_folder, '%s_test.txt' % dname)
        val_dataset = get_dataset(args=args, path=path, train=False, image_size=image_size, config=data_config)
        dataset = ConcatDataset([val_dataset])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        loaders.update({dname:loader})
    return loaders

# For FID use
def get_augonly_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        from_domain='all',
        alpha=1.0,
        config=None
):
    if args is not None:
        target = args.target
        source_list = args.source
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    loaders = {}
    for dname in source_list:
        dataset = get_augonly_dataset(args=args,
                                    path=[path],
                                    target_test=dname,
                                    image_size=image_size,
                                    crop=crop,
                                    jitter=jitter,
                                    from_domain=from_domain,
                                    alpha=alpha,
                                    config=data_config)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
        loaders.update({dname:loader})
    return loaders

#AugMix
def get_single_augmix_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_augmix_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

#Mixup
def get_single_mixup_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_mixup_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

# StyleAug
def get_single_styleaug_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_styleaug_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

# RandAug
def get_single_randaug_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_randaug_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

# CutOut
def get_single_cutout_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_cutout_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

# ACVC
def get_single_acvc_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_acvc_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

#PixMix
def get_single_pixmix_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_pixmix_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

#FullOurs
def get_single_fullours_train_dataloader(
        source_list=None,
        batch_size=64,
        image_size=224,
        crop=False,
        jitter=0,
        args=None,
        config=None
):
    if args is not None:
        target = args.target
    if config is not None:
        batch_size = config["batch_size"]
        data_config = config["data_opt"]
    else:
        data_config = None

    data_folder = get_datalists_folder(args)
    path = os.path.join(data_folder, '%s_train.txt' % target)
    dataset = get_fullours_dataset(args=args,
                                  path=[path],
                                  image_size=image_size,
                                  crop=crop,
                                  jitter=jitter,
                                  config=data_config)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)
    return loader

if __name__ == "__main__":
    batch_size=16
    source = ["art_painting", "cartoon", "photo"]
    # loader = get_fourier_train_dataloader(source, batch_size, image_size=224, from_domain='all', alpha=1.0)
    loader = get_intervention_dataloader(source, batch_size, image_size=224)

    it = iter(loader)
    batch = next(it)
    images = torch.cat(batch[0], dim=0)
    # images = batch[0][0]
    save_image_from_tensor_batch(images, batch_size, path='batch.jpg', device='cpu')