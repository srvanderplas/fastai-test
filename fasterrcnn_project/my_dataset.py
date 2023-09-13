from PIL import Image
from torch.utils.data import Dataset
from lxml import etree
import json
import torch
import json
import numpy as np

class VocDataset(Dataset):
    '''
    This class defines my dataset for trainig the model. '''
    
    def __init__(self, dataset_path, transform = None, mode = 'train'):
        '''
        Args:
            dataset_path: A string. 
                          A root directory includes:
                              a folder named "JPEG_Images" saving the images in jpeg form; 
                              a folder named "Modified_Annotations" saving voc xml files;
                              at least one of the txt files named "train.txt", "validate.txt" or 
                              "test.txt".
                              Refer to the directory tree of Dataset.

            transform: An instanced tranformation.
            mode: A string. This string can be 'train', 'validate' or 'test'.
        '''
        super().__init__()
        
        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, 'JPEG_Images')
        self.target_path = os.path.join(dataset_path, 'Modified_Annotations') # XML path
        assert os.path.exists(self.image_path), 'JPEGImages does not exist'
        assert os.path.exists(self.image_path), 'Annotations does not exist'
        
      
        self.index_list_path = os.path.join(self.dataset_path, mode+'.txt')
        with open(self.index_list_path, 'r') as f:
            self.index_list = [l[:-1] for l in f.readlines()]
        
        self.image_index_list = self.index_list
        self.target_index_list = self.index_list
        
        # Read the classes as a dictionary
        try:
            classes_json = open('./my_classes.json', 'r')
            self.classes_dict = json.load(classes_json)
        except Exception as e:
            print(e)
            exit(-1)
        
        self.length = len(self.index_list)
        self.transform = transform
        
    def recur_parse_xml_to_dict(self, xml):
        '''
        This function recursively parses an xml files into a dictionary of dictionaries. 
        Refered to recursive_parse_xml_to_dict in tensorflow.
        Args:
            xml: an xml tree object. This object obtained by parsing XML file using lxml.etree
            
        Returns:
            A dictionary. This dictionary holding XML contents
        '''
        if len(xml) == 0: # stop condition
            return {xml.tag: xml.text}
        
        content = {}
        for c in xml:
            c_content = self.recur_parse_xml_to_dict(c)
            if c.tag != 'object':
                content[c.tag] = c_content[c.tag]
            else:
                if c.tag not in content.keys():
                    content[c.tag] = []
                content[c.tag].append(c_content[c.tag])
        return {xml.tag: content}
                
                
    def __getitem__(self, index):
        image_p = os.path.join(self.image_path, self.image_index_list[index] + '.jpg')
        target_p = os.path.join(self.target_path, self.target_index_list[index] + '.xml')
        
        image = Image.open(image_p)
        if image.format != 'JPEG':
            raise ValueError("Image '{}' format is not JPEG".format(self.target_index_list[index]))
        
        with open(target_p) as t:
            t_str = t.read()
        xml = etree.fromstring(t_str)
        xml_content = self.recur_parse_xml_to_dict(xml)['annotation']
        
        
        boxes = []
        labels = []
        iscrowd = [] # overlapping object or not, need to work on xml "difficult" item
        assert 'object' in xml_content, '{} lack of object information.'.format(xml)
        for obj in xml_content["object"]:
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])
            
            if xmax <= xmin or ymax <= ymin:
                print("Warning: There might be coordinates flipped in '{}'  xml".format(self.target_index_list[index]))
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes_dict[obj['name']])
            if 'difficult' in obj:
                iscrowd.append(int(obj['difficult']))
            else:
                iscrowd.append(0)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transform is not None:
            image, target = self.transform(image, target)
            
        return image, target
    
    
    def __len__(self):
        return self.length
