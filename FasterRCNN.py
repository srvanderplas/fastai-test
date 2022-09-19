import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


classes = ['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 'triangle', 'exclude',
           'star', 'bowtie', 'line', 'ribbon']

# Perform transfer learning with a pretrained faster RCNN model
base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the pretrained head
num_classes = len(classes)

# get number of input features for the classifier
in_features = base_model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# https://www.kaggle.com/code/pestipeti/pytorch-starter-fasterrcnn-train/notebook
import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET

class VocDataset(Dataset):
    def __init__(self, dataset_path, transform = None):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, 'JPEGImages')
        self.target_path = os.path.join(dataset_path, 'Modified Annotations') # XML path
        assert os.path.exists(self.image_path), 'JPEGImages does not exist'
        assert os.path.exists(self.image_path), 'Annotations does not exist'
        
        self.image_index_list = [s.split('.')[0] for s in os.listdir(self.image_path)]
        self.target_index_list = [s.split('.')[0] for s in os.listdir(self.target_path)]
        # assert self.image_index_list == self.target_index_list, 'image names do not match xml names'
        
        self.length = len(self.image_index_list)
        self.transform = transform
    
    def simple_parse_xml(self, target_name): # could be imporoved by reccursion
        classes = {'logo' : 1, 'polygon' : 2, 'chevron' : 3, 'circle': 4, 'text':5,
                   'quad' : 6, 'other' : 7, 'triangle' : 8, 'exclude' : 9, 'star' : 10, 
                   'bowtie': 11, 'line' : 12, 'ribbon' :13}
        tree = ET.parse(target_name) # target name is the path of target file
        root = tree.getroot()
        labels = [] # list of all labels in current xml file
        boxes = [] # list of coordinates information in current xml file
        for obj in root.findall('object'):
            lab = obj.find('name').text
            labels.append(int(classes[lab]))
            bndbox = obj.find('bndbox')
            box = [float(bndbox.find('xmin').text), float(bndbox.find('ymin').text), 
                  float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)]
            boxes.append(box)
        
        return {'labels' : torch.as_tensor(labels, dtype = torch.int64),
                'boxes' : torch.as_tensor(boxes, dtype = torch.float)}
        
    def __getitem__(self, index):
        image_name = os.path.join(self.image_path, self.image_index_list[index] + '.jpg')
        target_name = os.path.join(self.target_path, self.target_index_list[index] + '.xml')
        
        image = Image.open(image_name).convert('RGB')
        target = self.simple_parse_xml(target_name)
        
        if self.transform:
            image = self.transform(image)
        return {'image' : image, 'target' : target}
      
    def __len__(self):
        return self.length
    
    def collate_fn_(self, batch):
        images = [i['image'] for i in batch]
        targets = [{'labels' : i['target']['labels'], 'boxes':i['target']['boxes']} for i in batch]
        
        return {'images' : images, 'targets' : targets}

transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (255, 255, 255))
        ]
    )
dataset_path = './Modified Data'

ds = VocDataset(dataset_path = dataset_path, transform = transform)
dl = DataLoader(ds, batch_size = 4, shuffle = True, collate_fn = ds.collate_fn_)

classes = ['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 'triangle', 'exclude',
           'star', 'bowtie', 'line', 'ribbon']
# num_classes = len(classes)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=len(classes)+1, pretrained=False, pretrained_backbone = False)

model.train()
for batch in dl:
    pred = model(batch['images'], batch['targets'])
    print(pred)
    break

'''
Pytorch Doc at https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Will adopt Finetuning from a pretrained model for a quick start, will look at different backbone later
'''

