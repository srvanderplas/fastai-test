import xml.etree.ElementTree as ET
import os
import copy
import time
import utils
from PIL import Image
import pandas as pd
import numpy as np
from type_classes import *
# import typo_classes

t0 = time.time()

def get_num_rep(obj):
    '''
    This function gets how many replicated boxes should have
    '''
    names = (objec.find('name').text).split(',')
    names = sum([i.split(' ') for i in names], []) 
    names = [i.strip() for i in names if i.strip()]
    num_rep = len(names) - 1
    return num_rep

def make_rep(num_rep):
    '''This fuction replicates the boxes'''
    while num_rep > 0:
            dup = copy.deepcopy(obj)
            dup.find('name').text = names[rep]
            root.append(dup)
            num_rep -= 1
            if num_rep == 0:
                obj.find('name').text = names[0]
    
def split_bbox(file_path):
    '''This fuction rewrites the xml files that have replicated annotation by
       replicating the boxes with individual class'''
    f = file_path
    tree = ET.parse(f)
    root = tree.getroot()
    for obj in root.findall('object'):
        n_rep = get_num_rep(obj)
        make_rep(n_rep)
    
    t = ET.ElementTree(root)
    with open (file_path, "wb") as file :
        t.write(file)

def find_unique_classes(path):
    unique_class = set([])
    f = path
    tree = ET.parse(f)
    root = tree.getroot()
    for obj in root.findall('object'):
        names = obj.find('name').text
        class_map.add(names)
    return unique_class
        

if __name__ == '__main__':
    paths = get_files('/Users/huamuxin/Documents/fastai-test/Modified data/Modified Annotation', extensions=['.xml']) # fastai
    class_map = set([])
    for path in paths:
        path=str(path)
        # find_unique_classes(path) # to find all classes annotated
        # split_bbox(path) # to split bbox 
    
    
    
    t1 = time.time()
    print('The whole splitting process costs', t1-t0, 's')
    
    # paths = ['/Users/huamuxin/Documents/fastai-test/Modified data/Modified Annotation/ys-by-yohji-yamamoto-regular-sneaker-off-white_product_9109337_color_527.xml']         
    for path in paths:
        path=str(path)
        tree = ET.parse(path)
        root = tree.getroot()
        for obj in root.findall('object'):
            idx = 0
            name = obj.find('name').text
            # for cls in classes:
            #     idx = name in cls
            #     if idx:
            #         count[str(cls[0])] += 1
            # 
            if name == 'elongated':
                print(path)
    print('count')
            
    count
    # {'logo': 1600, 'polygon': 55, 'octagon': 21, 'pentagon': 317, 'hexagon': 1133, 
    # 'chevron': 4479, 'circle': 5474, 'text': 7198, 'quad': 8987, 'other': 3543, 
    # 'triangle': 2784, 'exclude': 800, 'star': 1430, 'bowtie': 1175, 'line': 5214, 
    # 'ribbon': 256}
