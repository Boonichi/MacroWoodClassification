import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

IMAGE_FOLDER_MEAN = (0.4895, 0.4201, 0.3983)
IMAGE_FOLDER_STD = (0.0336, 0.0276, 0.0243)

def meanstd(dl):
    batch,sum_,sqr_=0,0,0
    for x,y in tqdm(dl):
        sum_+=torch.mean(x,axis=[0,2,3])
        sqr_+=torch.mean(x**2,axis=[0,2,3])
        batch+=1
    mean= sum_/batch
    std= (sqr_/batch)-mean**2
    print(mean,std)

def build_dataset(args, is_train):
    transform = build_transform(args, is_train)
    #transform = transforms.Compose([
    #    transforms.Resize((args.input_size,args.input_size)),
    #    transforms.ToTensor(),
    #])
    
    if args.data_set == "image_folder":
        root = args.data_dir if is_train else args.eval_data_dir
        dataset = datasets.ImageFolder(root, transform=transform)
        print("Size of dataset: ",len(dataset))
        nb_classes = args.nb_classes
        #print(meanstd(DataLoader(dataset)))
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    
    return dataset, nb_classes

def build_transform(args, is_train):
    resize_im = args.input_size > 32
    if args.data_mean_std == "ImageNet":
        mean = IMAGENET_DEFAULT_MEAN
        std =  IMAGENET_DEFAULT_STD
    elif args.data_mean_std == "ImageFolder":
        mean = IMAGE_FOLDER_MEAN
        std = IMAGE_FOLDER_STD
    elif args.data_mean_std == "Original":
        mean = [0,0,0]
        std = [1,1,1]
    
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            #color_jitter=args.color_jitter,
            #auto_augment=args.aa,
            hflip=args.hflip,
            vflip=args.vflip,
            #interpolation=args.train_interpolation,
            #re_prob=args.reprob,
            #re_mode=args.remode,
            #re_count=args.recount,
            mean=mean,
            std=std,
        )

        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform


    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)