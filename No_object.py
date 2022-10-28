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


class Check():
    def __init__(self, txt_path = 'Modified Data/train_origin.txt', 
                       xml_path = 'Modified Data/Modified Annotations/', 
                       img_path = 'Modified Data/JPEGImages/'):
                         
        f = open(txt_path, 'r')
        names = f.readlines()
        f.close()
        
        self.names = names
        self.txt_path = txt_path
        self.xml_path = xml_path
        self.img_path = img_path
      
        self.xml = glob.glob(xml_path + '*.xml')
        self.img = glob.glob(img_path + '*.jpg')
        
    def if_exist(self):
        xml_names = [i.split('/')[-1] for i in self.xml]
        img_names = [i.split('/')[-1] for i in self.img]
        
        unmatch = []
        for name in self.names:
            name = name.split('\n')[0]
            name_xml = name + '.xml'
            name_img = name + '.jpg'
            
            if (name_xml not in xml_names) or (name_img not in img_names):
                unmatch.append(name)
                
        return unmatch
      
    def if_boxvalid(self):
        not_validbox = []
        for xml_file in self.xml:
            tree = ET.parse(xml_file) # target name is the path of target file
            root = tree.getroot()
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text) 
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                if (xmin < 0) or (xmin < 0) or (xmin < 0) or (xmin < 0) or (xmin >= xmax) or (ymin >= ymax):
                    not_validbox.append(xml_file)
        return not_validbox

check = Check()
check.if_exist()
check.if_boxvalid()

