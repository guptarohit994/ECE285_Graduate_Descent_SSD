from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    '''Return tuple containing tensor of batch of images stacked in 0 dimension, list of tensors stacked in 0 dimension containing annotations for given image
    Given tuple of tenser images and lists of annotations.'''
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    '''Returns resized and mean-normalized image'''
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    '''Class to call base_transform function on every image iteratively.'''
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
