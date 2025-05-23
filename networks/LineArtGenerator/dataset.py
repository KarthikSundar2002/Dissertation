import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import numpy as np
import os
import random

from collections import OrderedDict
# import util.util as util
from PIL import Image
import cv2

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dire, stop=10000):
    images = []
    count = 0
    assert os.path.isdir(dire), '%s is not a valid directory' % dire
    for root, _, fnames in sorted(os.walk(dire)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                count += 1
            if count >= stop:
                return images
    return images

def get_params(size):
    w, h = size
    new_h = h
    new_w = w
   
    new_h = new_w = 256

    x = random.randint(0, np.maximum(0, new_w - 256))
    y = random.randint(0, np.maximum(0, new_h - 256))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(params=None, grayscale=False, method=Image.BICUBIC, convert=True, norm=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    osize = [256, 256]
    transform_list.append(transforms.Resize(osize, method))
    

    
    if params is None:
        transform_list.append(transforms.RandomCrop(256))
    else:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 256)))

    # if not opt.no_flip:
    #     if params is None:
    #         transform_list.append(transforms.RandomHorizontalFlip())
    #     elif params['flip']:
    #         transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if not grayscale:
            if norm:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    color = (255, 255, 255)
    if img.mode == 'L':
        color = (255)
    elif img.mode == 'RGBA':
        color = (255, 255, 255, 255)

    if (ow > tw and oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    elif ow > tw:
        ww = img.crop((x1, 0, x1 + tw, oh))
        return add_margin(ww, size, 0, (th-oh)//2, color)
    elif oh > th:
        hh = img.crop((0, y1, ow, y1 + th))
        return add_margin(hh, size, (tw-ow)//2, 0, color)
    return img

def add_margin(pil_img, newsize, left, top, color=(255, 255, 255)):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (newsize, newsize), color)
    result.paste(pil_img, (left, top))
    return result

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

class UnpairedDepthDataset(data.Dataset):
    def __init__(self, root, root2, transforms_r=None, mode='train', midas=False, depthroot=''):

        self.root = root
        self.mode = mode
        self.midas = midas

        all_img = make_dataset(self.root)

        self.depth_maps = 0
        if self.midas:

            depth = []
            print(depthroot)
            if os.path.exists(depthroot):
                depth = make_dataset(depthroot)
            else:
                print('could not find %s'%depthroot)
                import sys
                sys.exit(0)

            newimages = []
            self.depth_maps = []

            for dmap in depth:
                lastname = os.path.basename(dmap)
                trainName1 = os.path.join(self.root, lastname)
                trainName2 = os.path.join(self.root, lastname.split('.')[0] + '.jpg')
                if (os.path.exists(trainName1)):
                    newimages += [trainName1]
                elif (os.path.exists(trainName2)):
                    newimages += [trainName2]
            print('found %d correspondences' % len(newimages))

            self.depth_maps = depth
            all_img = newimages

        self.data = all_img
        self.mode = mode

        self.transform_r = transforms.Compose(transforms_r)
        
        # if mode == 'train':
            
        #     self.img2 = make_dataset(root2)

        #     if len(self.data) > len(self.img2):
        #         howmanyrepeat = (len(self.data) // len(self.img2)) + 1
        #         self.img2 = self.img2 * howmanyrepeat
        #     elif len(self.img2) > len(self.data):
        #         howmanyrepeat = (len(self.img2) // len(self.data)) + 1
        #         self.data = self.data * howmanyrepeat
        #         self.depth_maps = self.depth_maps * howmanyrepeat
            

        #     cutoff = min(len(self.data), len(self.img2))

        #     self.data = self.data[:cutoff] 
        #     self.img2 = self.img2[:cutoff] 

        #     self.min_length =cutoff
        # else:
        #     self.min_length = len(self.data)


    def __getitem__(self, index):

        img_path = self.data[index]

        basename = os.path.basename(img_path)
        base = basename.split('.')[0]

        img_r = Image.open(img_path).convert('RGB')
        transform_params = get_params(img_r.size)
        A_transform = get_transform(transform_params, grayscale=False, norm=False)
        B_transform = get_transform(transform_params, grayscale=True, norm=False)        

        if self.mode != 'train':
            A_transform = self.transform_r

        img_r = A_transform(img_r )

        B_mode = 'L'
        # if self.opt.output_nc == 3:
        #     B_mode = 'RGB'

        img_depth = 0
        if self.midas:
            img_depth = cv2.imread(self.depth_maps[index])
            img_depth = A_transform(Image.fromarray(img_depth.astype(np.uint8)).convert('RGB'))


        img_normals = 0
        label = 0

        input_dict = {'r': img_r, 'depth': img_depth, 'path': img_path, 'index': index, 'name' : base, 'label': label}

        # if self.mode=='train':
        #     cur_path = self.img2[index]
        #     cur_img = B_transform(Image.open(cur_path).convert(B_mode))
        #     input_dict['line'] = cur_img

        return input_dict

    def __len__(self):
        return len(self.data)