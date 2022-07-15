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
    

def all_classes(file_path):
    f = file_path
    tree = ET.parse(f)
    root = tree.getroot()
    for obj in root.findall('object'):
        names = (obj.find('name').text).split(',')
        
paths = get_files('/Users/huamuxin/Documents/fastai-test/Modified data/Modified Annotation', extensions=['.xml']) # fastai
class_map = set([])
for path in paths:
    path=str(path)

# ------ This part finds out unique classes --------------
    f = path
    tree = ET.parse(f)
    root = tree.getroot()
    for obj in root.findall('object'):
        names = obj.find('name').text
        class_map.add(names)
print(class_map)
# Looks more cleaning work than expected
  # 'qud', 'rectangle', 'hex', 'rounded', 'othre', 'hexagon', 'circle_', 'triangle',
  # 'etxt', 'stars', 'qudarilateral', 'cirlce', 'quardilateral', 'quadrilateral', 
  # 'ext', 'cirle', 'quadrilaterl', 'chrevron', 'othere', 'star', 'triangl', 
  # 'elongated', 'chervron', 'toe', 'circles', 'texture_crepe', 'background', 
  # 'iother', 'hexagon chevron', 'hatching', None, 'qiad', 'lines', 'quadilateral',
  # 'quadrilaterals', 'smooth', 'irregular', 'qua', 'ribbon', 'circle text', 'star.',
  # 'circe', 'quadrliateral', 'ciricle', 'circle triangle', 'quad', 'octagon', 'texture',
  # 'cricle', 'lkns', 'pentagons', 'crepe', 'texxt', 'chevrons', 'triangles', 
  # 'texture_smooth', 'bowite', 'blowtie', 'tet', 'oter', 'bowtie', 'circleline',
  # 'pentagon', 'quarilateral', 'heagon', 'region', 'ttriangle', 'start', 'triangels',
  # 'circle', 'qaud', 'heel', 'cheron', 'star quad', 'quadrilateralline', 'chrvron',
  # 'polygon', 'trianlges', 'lie', 'quadrilatteral', 'excludue', 'quadmline',
  # 'quadrilteral', 'trianglee', 'exclud', 'other', 'quadrilaterlal', 'exclude', 
  # 'ribon', 'cicle', 'text', 'logo', 'qyad', 'chevron', 'line', 'bootie', 
  # 'quadcircle', 'curved line'

    # split_bbox(path)

t1 = time.time()
print('The whole splitting process costs', t1-t0, 's')
