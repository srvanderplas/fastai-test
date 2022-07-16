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
        names = sum([i.split(' ') for i in names], []) 
        names = [i.strip() for i in names if i.strip()]
        rep = len(names) - 1
        while rep > 0:
            dup = copy.deepcopy(obj)
            dup.find('name').text = names[rep]
            root.append(dup)
            rep -= 1
            if rep == 0:
                obj.find('name').text = names[0]
    
    t = ET.ElementTree(root)
    with open (file_path, "wb") as file :
        t.write(file)

        
paths = get_files('/Users/huamuxin/Documents/fastai-test/Modified data/Modified Annotation', extensions=['.xml']) # fastai
class_map = set([])
for path in paths:
    path=str(path)

# ------ This part finds out unique classes --------------
    f = path
    n_file += 1
    tree = ET.parse(f)
    root = tree.getroot()
    for obj in root.findall('object'):
        names = obj.find('name').text
        if names == None:
            print(f)
        class_map.add(names)
print(class_map)
# ------ End of finding out unique classes --------------

    # split_bbox(path)

t1 = time.time()
print('The whole splitting process costs', t1-t0, 's')

'qud', 'rectangle', 'hex', 'rounded', 'othre', 'hexagon', 'circle_', 'triangle',
'curved', 'etxt', 'stars', 'qudarilateral', 'cirlce', 'quardilateral', 
'quadrilateral', 'ext', 'cirle', 'quadrilaterl', 'chrevron', 'othere', 'star', 
'triangl', 'elongated', 'chervron', 'toe', 'circles', 'texture_crepe', 
'background', 'iother', 'hatching', 'qiad', 'lines', 'quadilateral', 
'quadrilaterals', 'smooth', 'irregular', 'qua', 'ribbon', 'star.', 'circe', 
'quadrliateral', 'ciricle', 'quad', 'octagon', 'texture', 'cricle', 'lkns', 
'pentagons', 'crepe', 'texxt', 'chevrons', 'triangles', 'texture_smooth', 
'bowite', 'blowtie', 'tet', 'oter', 'bowtie', 'circleline', 'pentagon', 
'quarilateral', 'heagon', 'region', 'ttriangle', 'start', 'star,', 'triangels', 
'circle', 'qaud', 'heel', 'cheron', 'quadrilateralline', 'chrvron', 'polygon', 
'trianlges', 'lie', 'quadrilatteral', 'excludue', 'quadmline', 'quadrilteral', 
'trianglee', 'exclud', 'other', 'quadrilaterlal', 'exclude', 'ribon', 'cicle', 
'text', 'logo', 'qyad', 'chevron', 'line', 'bootie', 'quadcircle'




