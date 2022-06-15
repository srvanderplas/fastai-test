from PIL import Image
import xml.etree.ElementTree as ET
import utils # get_files
import os
    
def is_bbox(obj):
  
    '''Return 1 if the object is a bounding box
    
    :param obj: an object element in xml file, polygon or bounding box
    '''
    return 1 if obj.findall('type') else 0
  
def find_obj_coordinates(obj):
  
    '''Return two lists containing x coordinates, y coordinates of a bounding box or polygon
    Remove the pt elements to avoid confusing
      
    :param obj: an object element in xml file, polygon or bounding box
    '''
    x_coords, y_coords = [], []
    xml_bbox = obj.findall('polygon')
    for bbox in xml_bbox:
        pts = (obj.find('polygon')).findall('pt')
        for pt in pts:
            x_coords.append(int(pt.find('x').text))
            y_coords.append(int(pt.find('y').text))
        obj.remove(bbox)
    return x_coords, y_coords
  
def write_bbox(xmin, xmax, ymin, ymax, obj):
  
    '''Return a BOUNDING BOX object with written coordinates
    
    :param xmin: minimum x coordinates of a bounding box or polygon
    :param xmax: minimum x coordinates of a bounding box or polygon
    :param ymin: minimum x coordinates of a bounding box or polygon
    :param ymax: minimum x coordinates of a bounding box or polygon
    :param obj: an object element in xml file, polygon or bounding box
    '''
    m = obj
    b1 = ET.SubElement(m, 'bndbox')
    c1 = ET.SubElement(b1, 'xmin')
    c1.text = str(xmin)
    c2 = ET.SubElement(b1, 'ymin')
    c2.text = str(ymin)
    c3 = ET.SubElement(b1, 'xmax')
    c3.text = str(xmax)
    c4 = ET.SubElement(b1, 'ymax')
    c4.text = str(ymax)
    return obj    
      
def polygon_remove(root, obj):

    '''Remove a POLYGON element
    
    :param root: root of an xml file
    :param object: a POLYGON element in xml file 
    '''
    root.remove(obj)

def to_bbox(xcoord_list, ycoord_list, obj):
  
    '''Return a BOUNDING BOX element
  
    :param xcoord_list: list containing x coordinates
    :param xcoord_list: list containing y coordinates
    :param obj: an object element in xml file, polygon or bounding box
    '''
    xmin, xmax, ymin, ymax = min(xcoord_list), max(xcoord_list), min(ycoord_list), max(ycoord_list)
    print('xmin, xmax: ', xmin, xmax, type(xmin))
    write_bbox(xmin, xmax, ymin, ymax, obj)
    return obj
    
    
def convert(file_path):
  
    ''' Return a voc annotation that can be parsed by VOCBBoxParser
    
    :param file_path : file location of our xml file
    '''
    
    f = file_path
    img_path = f.split('.')[0] + '.jpg'
    tree = ET.parse(f)
    root = tree.getroot()
    
    # Add size element 
    # Read image size x and y from origin image file and write in
    img = Image.open(img_path)
    m1 = ET.Element('size')
    root.append(m1)
    
    b1 = ET.SubElement(m1, 'width') 
    b1.text = str(img.width)
    b2 = ET.SubElement(m1, 'height')
    b2.text = str(img.height)
    b3 = ET.SubElement(m1, 'depth') # 
    b3.text = "3"
    
    # Add bndbox element and remove <pt>
    for obj in root.iter('object'):
        (xcoords_list, ycoords_list) = find_obj_coordinates(obj)
        print(xcoords_list, ycoords_list)
        print('--------')
        to_bbox(xcoords_list, ycoords_list, obj)
    
    
    # xes, ys = [], []
    # for obj in root.iter('object'):
    #     if len(obj) != 0:
    #         xml_bbox = obj.findall('polygon')
    #         for bbox in xml_bbox:
    #             pts = (obj.find('polygon')).findall('pt')
    #             for pt in pts:
    #                 xes.append(pt.find('x').text)
    #                 ys.append(pt.find('y').text)
    #     #         xml_bbox = obj.findall('polygon')
    # #         for each in xml_bbox:
    #             obj.remove(bbox)
    #     box_num = int(len(ys)/4)
    #     # x coordinates lists
    #     xmins = [xes[i*4] for i in range(box_num)]
    #     xmaxs = [xes[i*4+1] for i in range(box_num)]
    #     
    #     # y coordinates lists
    #     ymins = [ys[i*4] for i in range(box_num)]
    #     ymaxs = [ys[i*4+2] for i in range(box_num)]
    #     
    #     # compare to get xmin, xmax, ymin, ymax, and write 
    #     for i in range(box_num):
    #         xmin, xmax = min(xmins[i], xmaxs[i]), max(xmins[i], xmaxs[i])
    #         ymin, ymax = min(ymins[i], ymaxs[i]), max(ymins[i], ymaxs[i])
    #         
    #         m = obj
    #         b1 = ET.SubElement(m, 'bndbox')
    #         c1 = ET.SubElement(b1, 'xmin')
    #         c1.text = str(xmins[i])
    #         c2 = ET.SubElement(b1, 'ymin')
    #         c2.text = str(ymins[i])
    #         c3 = ET.SubElement(b1, 'xmax')
    #         c3.text = str(xmaxs[i])
    #         c4 = ET.SubElement(b1, 'ymax')
    #         c4.text = str(ymaxs[i])
    
    t = ET.ElementTree(root)
    # file_path = '02' + file_path
    with open (file_path, "wb") as files :
        t.write(files)
    files.close()


paths = get_files('/Users/huamuxin/Documents/fastai-test/VOCParser_test/testing', extensions=['.xml']) # get the file paths
for path in paths:
    path=str(path)
    convert(path)


