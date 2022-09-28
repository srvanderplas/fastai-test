import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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


import random
import os
    
def data_split(dataset_path, train_validate_percent = 0.9, train_percent = 0.8, test_percent = 0.1):
    target_path = os.path.join(dataset_path, 'Modified Annotations')
    all_xml_file = os.listdir((target_path))
    
    num_xml = len(all_xml_file) # total number of xml files
    num_train_validate = int(num_xml * train_validate_percent)
    num_test = num_xml - num_train_validate
    
    num_train = int(num_train_validate * train_percent)
    num_validate = num_train_validate - num_train
    
    
    train_validate_idx = random.sample(range(num_xml), num_train_validate)
    train_idx = random.sample(train_validate_idx, num_train)
    
    f_train_val = open(dataset_path + '/train_validate.txt', 'w')
    f_train = open(dataset_path + '/train.txt', 'w')
    f_validate = open(dataset_path + '/validate.txt', 'w')
    f_test = open(dataset_path + '/test.txt', 'w')
    
    for i in range(num_xml):
        name = all_xml_file[i].split('.')[0] + '\n'
        if i in train_validate_idx:
            f_train_val.write(name)
            if i in train_idx:
                f_train.write(name)
            else:
                f_validate.write(name)
        else:
              f_test.write(name)
    
    f_train_val.close()
    f_train.close()
    f_validate.close()
    f_test.close()
          
    
    
    
class VocDataset(Dataset):
    def __init__(self, dataset_path, transform = None, mode = 'train'):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, 'JPEGImages')
        self.target_path = os.path.join(dataset_path, 'Modified Annotations') # XML path
        assert os.path.exists(self.image_path), 'JPEGImages does not exist'
        assert os.path.exists(self.image_path), 'Annotations does not exist'
        
        # self.image_index_list = [s.split('.')[0] for s in os.listdir(self.image_path)]
        # self.target_index_list = [s.split('.')[0] for s in os.listdir(self.target_path)]
        # assert self.image_index_list == self.target_index_list, 'image names do not match xml names'
        
        assert mode in ['train', 'validate', 'test', 'train_validate']
        self.index_list_path = os.path.join(self.dataset_path, mode+'.txt')
        with open(self.index_list_path, 'r') as f:
            self.index_list = [l[:-1] for l in f.readlines()]
        
        self.image_index_list = self.index_list
        self.target_index_list = self.index_list

        self.length = len(self.index_list)
        # self.length = len(self.image_index_list)
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
        # print(index)
        image_name = os.path.join(self.image_path, self.image_index_list[index] + '.jpg')
        target_name = os.path.join(self.target_path, self.target_index_list[index] + '.xml')
        # print(image_name)
        
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
      
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (255, 255, 255))
        ]
    )

dataset_path = './Modified Data'
# data_split(dataset_path) # Already completed data splitting

--------
'''
Pytorch Doc at https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Kaggle example https://www.kaggle.com/code/pestipeti/pytorch-starter-fasterrcnn-train
'''

train_ds  = VocDataset(dataset_path = dataset_path, transform = transform, mode = 'train')
valid_ds  = VocDataset(dataset_path = dataset_path, transform = transform, mode = 'validate')

classes = ['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 'triangle', 'exclude',
           'star', 'bowtie', 'line', 'ribbon']

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=len(classes)+1, pretrained=False, pretrained_backbone = False)
           
# replace the classifier with a new one, that has
# num_classes which is user-defined
# num_classes = len(classes) + 1
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

train_dl = DataLoader(train_ds, batch_size = 4, shuffle = True, collate_fn = train_ds.collate_fn_)
valid_dl = DataLoader(valid_ds, batch_size = 4, shuffle = True, collate_fn = valid_ds.collate_fn_)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 2
itr = 1

loss_hist = Averager()

# t_ds = VocDataset(dataset_path = dataset_path, transform = transform, mode = 'train') # for testing
# t_dl = DataLoader(t_ds, batch_size = 4, shuffle = True, collate_fn = ds.collate_fn_) # for testing

for epoch in range(num_epochs):
    for batch in train_dl:
        
        loss_hist.reset()
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        im = batch['images']
        targ = batch['targets']
        images = list(image.to(device) for image in im)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
    
        loss_dict = model(images, targets)
      
        # break
        # print('loss_dict: ', loss_dict)
        # {'loss_classifier': tensor(6.3938, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.1693, grad_fn=<DivBackward0>), 
        # 'loss_objectness': tensor(1.6913, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 
        # 'loss_rpn_box_reg': tensor(0.9039, grad_fn=<DivBackward0>)}

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        print('losses: ', losses)
        # tensor(9.1583, grad_fn=<AddBackward0>)

        print('loss_value: ', loss_value)
        # 9.158288955688477

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0: # 50
            print(f"Iteration #{itr} loss: {loss_value}")

        itr += 1
        
# loss_dict:  {'loss_classifier': tensor(6.3938, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.1693, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.6913, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.9039, grad_fn=<DivBackward0>)}        
# losses:  tensor(9.1583, grad_fn=<AddBackward0>)        
# loss_value:  9.158288955688477        
# Iteration #2 loss: 9.158288955688477        
# loss_dict:  {'loss_classifier': tensor(6.2422, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.0992, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.6525, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.6531, grad_fn=<DivBackward0>)}
# losses:  tensor(8.6470, grad_fn=<AddBackward0>)
# loss_value:  8.646978378295898
# Iteration #3 loss: 8.646978378295898
# loss_dict:  {'loss_classifier': tensor(6.3935, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.2516, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.7266, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.7815, grad_fn=<DivBackward0>)}
# losses:  tensor(9.1533, grad_fn=<AddBackward0>)
# loss_value:  9.153275489807129
# Iteration #4 loss: 9.153275489807129
# loss_dict:  {'loss_classifier': tensor(6.0925, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.2861, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.7179, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.9329, grad_fn=<DivBackward0>)}
# losses:  tensor(9.0295, grad_fn=<AddBackward0>)
# loss_value:  9.029480934143066
# Iteration #5 loss: 9.029480934143066        
# loss_dict:  {'loss_classifier': tensor(5.7527, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.3174, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.8469, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.6728, grad_fn=<DivBackward0>)}
# losses:  tensor(8.5898, grad_fn=<AddBackward0>)
# loss_value:  8.589811325073242
# Iteration #6 loss: 8.589811325073242   
# loss_dict:  {'loss_classifier': tensor(6.3617, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.2610, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.6953, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.9091, grad_fn=<DivBackward0>)}
# losses:  tensor(9.2271, grad_fn=<AddBackward0>)
# loss_value:  9.227115631103516
# Iteration #7 loss: 9.227115631103516        
# loss_dict:  {'loss_classifier': tensor(6.3778, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.2174, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.6300, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.9175, grad_fn=<DivBackward0>)}
# losses:  tensor(9.1427, grad_fn=<AddBackward0>)
# loss_value:  9.142691612243652
# Iteration #8 loss: 9.142691612243652        
# loss_dict:  {'loss_classifier': tensor(6.3670, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.1742, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.6477, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.4726, grad_fn=<DivBackward0>)}
# losses:  tensor(8.6615, grad_fn=<AddBackward0>)
# loss_value:  8.66147232055664
# Iteration #9 loss: 8.66147232055664
# loss_dict:  {'loss_classifier': tensor(6.2625, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.3150, grad_fn=<DivBackward0>), 'loss_objectness': tensor(1.6337, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.7637, grad_fn=<DivBackward0>)}
# losses:  tensor(8.9750, grad_fn=<AddBackward0>)
# loss_value:  8.974950790405273
# Iteration #10 loss: 8.974950790405273

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()
    print(f"Epoch #{epoch} loss: {loss_hist.value}")
------

# # 
# # ds = VocDataset(dataset_path = dataset_path, transform = transform)
# # dl = DataLoader(ds, batch_size = 4, shuffle = True, collate_fn = ds.collate_fn_)
# # 
# # classes = ['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 'triangle', 'exclude',
# #            'star', 'bowtie', 'line', 'ribbon']
# # num_classes = len(classes)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=len(classes)+1, pretrained=False, pretrained_backbone = False)
# 
# model.train()
# for batch in t_dl:  # test the S
#     pred = model(batch['images'], batch['targets'])
#     print(pred)
#     break
# 
# # {'loss_classifier': tensor(6.4603, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.1781, grad_fn=<DivBackward0>),
# # 'loss_objectness': tensor(1.5828, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.8501, 
# # grad_fn=<DivBackward0>)}
# 
# -----


# # import transforms as T
# # 
# # def get_transform(train):
# #     transforms = []
# #     transforms.append(T.PILToTensor())
# #     transforms.append(T.ConvertImageDtype(torch.float))
# #     if train:
# #         transforms.append(T.RandomHorizontalFlip(0.5))
# #     return T.Compose(transforms)
# #   
# # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# # model.to(device)
# # params = [p for p in model.parameters() if p.requires_grad]
# # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# # 
# # # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# # 
# # 
# # for batch in dl:
# #     im = batch['images']
# #     targ = batch['targets']
# #     images = list(image.to(device) for image in im)
# #     targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
# #     
# #     loss_dict = model(images, targets)
# #     break
# # # {'loss_classifier': tensor(2.5329, grad_fn=<NllLossBackward0>),
# # # classifier loss is the loss of prediction of object classes in bounding boxes.
# # 
# # # 'loss_box_reg': tensor(0.0386, grad_fn=<DivBackward0>),
# # # Localisation loss in the ROI head. Measures the loss for box localisation (predicted location vs true location).
# # 
# # # 'loss_objectness': tensor(0.4805, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
# # # This is also when we are extracting the region proposals whether the object is present in the anchorbox or not.
# # 
# # # 'loss_rpn_box_reg': tensor(0.0169, grad_fn=<DivBackward0>)}
# # 
# # num_epochs = 2
# # 
# # itr = 1
# 


