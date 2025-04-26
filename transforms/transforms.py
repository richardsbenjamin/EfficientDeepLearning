import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms


def cutout(mask_size, p, cutout_inside = True):
    """https://github.com/hysts/pytorch_cutout
    
    Code slightly adapted for our application. See comments below.
    """
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2] 

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = 0
        return transforms.ToPILImage()(image)

    return _cutout


def hide_and_seek(img):
    """https://github.com/kkanshul/Hide-and-Seek"""
    # get width and height of the image
    img = transforms.ToImage()(img)
    s = img.shape
    wd = s[0]
    ht = s[1]

    # possible grid size, 0 means no hiding
    grid_sizes=[0,4,8,16,32]

    # hiding probability
    hide_prob = 0.5
 
    # randomly choose one grid size
    grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

    # hide the patches
    if(grid_size != 0):
         for x in range(0,wd,grid_size):
             for y in range(0,ht,grid_size):
                 x_end = min(wd, x+grid_size)  
                 y_end = min(ht, y+grid_size)
                 if(random.random() <=  hide_prob):
                       img[x:x_end,y:y_end,:]=0

    return transforms.ToPILImage()(img)

def gaussian_noise(image):
    image = transforms.ToImage()(image)
    image = transforms.ToDtype(torch.float32, scale=True)(image)
    image = transforms.GaussianNoise()(image)
    return transforms.ToPILImage()(image)

