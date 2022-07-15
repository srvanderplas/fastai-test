import xml.etree.ElementTree as ET
import os
import copy
import time
import utils
from PIL import Image
import pandas as pd
import numpy as np

t0 = time.time()

def split_bbox(file_path):
    f = file_path
    tree = ET.parse(f)
    root = tree.getroot()
    for obj in root.findall('object'):
        names = (obj.find('name').text).split(',')
        rep = len(names) - 1
        while rep > 0:
            dup = copy.deepcopy(obj)
            dup.find('name').text = names[rep].strip()
            root.append(dup)
            rep -= 1
            if rep == 0:
                obj.find('name').text = names[0]
    
    t = ET.ElementTree(root)
    with open (file_path, "wb") as file :
        t.write(file)
    


paths = get_files('/Users/huamuxin/Documents/fastai-test/Modified data/Modified Annotation', extensions=['.xml']) # fastai
for path in paths:
    path=str(path)
    split_bbox(path)
# path = r'/Users/huamuxin/Documents/fastai-test/Modified Data/Annotations/5-11-tactical-a-t-a-c-8-coyote-coyote_product_8851777_color_293417.xml'


t1 = time.time()
print('The whole splitting process costs', t1-t0, 's')
