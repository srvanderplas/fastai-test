import xml.etree.ElementTree as ET
import glob
import os

no_obj_list = []

paths = get_files('/Original Data/Annotations', extensions=['.xml']) # get the file paths

# file_path = '/Users/huamuxin/Documents/fastai-test/Modified Data/Modified Annotations/kizik-new-york-coffee-gum_product_9170616_color_8575.xml'
# file_path = '/Users/huamuxin/Documents/fastai-test/Modified Data/Modified Annotations/kizik-new-york-castle-grey_product_9170616_color_112673.xml'

dir = '/Users/huamuxin/Documents/fastai-test/Modified Data/Modified Annotations/'
file_paths = glob.glob(dir+'*.xml')

i = 0
for file_path in file_paths:
    # f = path + file_path
    i += 1
    tree = ET.parse(file_path)
    root = tree.getroot()
    if len(root.findall('object')) == 0:
        no_obj_list.append(file_path)
        
no_obj = [i.split('/')[-1] for i in no_obj_list]
no_obj = [i.split('.')[0] + '\n' for i in no_obj]


f_train = open('/Users/huamuxin/Documents/fastai-test/Modified Data/train.txt', 'r')
train_file = f_train.readlines()
f_train.close()

for delete in no_obj:
    for train in train_file:
        if train == delete:
            train_file.remove(train)

f_train2 = open(dataset_path + '/train2.txt', 'w')
for each in train_file:
    f_train2.write(each)
f_train2.close()
