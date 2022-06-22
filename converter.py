from PIL import Image
import xml.etree.ElementTree as ET
import utils # get_files
import os
import shapely
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
    
def is_bbox(obj):
  
    '''Return 1 if the object is a bounding box
    
    :param obj: an object element in xml file, polygon or bounding box
    '''
    return 1 if obj.findall('type') else 0
  
  
def is_drop(obj):
    
    '''Return 1 if the obj is a segment element
    
    :param obj: an object element in xml file, polygon or bounding box
    '''
    return 1 if obj.findall('segm') else 0
    


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
    write_bbox(xmin, xmax, ymin, ymax, obj)
    return obj
  
    
def area(polygon_xcoord_list, polygon_ycoord_list):
    
    ''' Return the area of a polygon
    
    :param polygon_xcoord_list: list containing x coordinates of a polygon
    :param polygon_ycoord_list: list containing y coordinates of a polygon
    '''
    
    polygon_obj = []
    for i in range(len(polygon_xcoord_list)):
        polygon_obj.append((int(polygon_xcoord_list[i]), int(polygon_ycoord_list[i])))
    polygon = Polygon(polygon_obj)
    return polygon.area
    
    
def area_polygon_bbox(polygon_xcoord_list, polygon_ycoord_list):
    
    ''' Return the area of the largest bbox made from a polygon
    
    :param polygon_xcoord_list: list containing x coordinates of a polygon
    :param polygon_ycoord_list: list containing y coordinates of a polygon
    '''
    
    xmin, xmax, ymin, ymax = min(polygon_xcoord_list), max(polygon_xcoord_list), min(polygon_ycoord_list), max(polygon_ycoord_list)
    area = (int(ymax) - int(ymin)) * (int(xmax) - int(xmin))
    return area
  
  
def polygon_contribution(polygon_xcoord_list, polygon_ycoord_list):
    
    ''' Return the contribution of polygon in a bbox
    
    :param polygon_xcoord_list: list containing x coordinates of a polygon
    :param polygon_ycoord_list: list containing y coordinates of a polygon
    '''
    poly_area = area(polygon_xcoord_list, polygon_ycoord_list)
    bbox_area = area_polygon_bbox(polygon_xcoord_list, polygon_ycoord_list)
    return poly_area/ bbox_area
  
  
def convert(file_path):
  
    ''' Return a voc annotation that can be parsed by VOCBBoxParser
    
    :param file_path: file location of our xml file
    '''
    
    f = file_path
    loc = file_path.replace('Annotations', 'Images')
    img_path = loc.split('.')[0] + '.jpg'
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
    b3 = ET.SubElement(m1, 'depth') 
    b3.text = "3"

    # Add bndbox element and remove <pt>
    for obj in root.findall('object'):
        if is_drop(obj):
            polygon_remove(root, obj)
        else:
            (xcoords_list, ycoords_list) = find_obj_coordinates(obj)
            print(xcoords_list, ycoords_list)
            if (is_bbox(obj)+ is_drop(obj) == 0):
                if len(xcoords_list) >= 3:
                    ratio = polygon_contribution(xcoords_list, ycoords_list)
                    if ratio < 0.7:
                        polygon_remove(root, obj)
            to_bbox(xcoords_list, ycoords_list, obj)
        
    t = ET.ElementTree(root)
    with open (file_path, "wb") as files :
        t.write(files)
    # files.close()
    



# ------------ converting our xml to standard voc parser ready xml -------------
paths = get_files('/Users/huamuxin/Documents/fastai-test/Modified data/Annotations', extensions=['.xml']) # get the file paths
for path in paths:
    path=str(path)
    convert(path)


    
# ---------- Loop over all the polygons to find Area polygon/ Area max box -----
paths = get_files('/Users/huamuxin/Documents/fastai-test/Original Data/Annotations', extensions=['.xml']) # get the file paths
poly_contribs = []
imgs = []
c = []
for path in paths:
    f = str(path)
    tree = ET.parse(f)
    root = tree.getroot()
    count = 0
    for obj in root.iter('object'):
        if (is_bbox(obj)+ is_drop(obj) == 0):
            (xcoords_list, ycoords_list) = find_obj_coordinates(obj)
            if len(xcoords_list) >= 3:
              # dc-legacy-98-slim-black-white-red_product_9065556_color_2125.xml, only one x coordinate
                ratio = polygon_contribution(xcoords_list, ycoords_list)
                poly_contribs.append(ratio)
                count += 1
    c.append(count) # c has the number of polygons in an image
    if count != 0:
        files = [f] * count
        imgs.append(files)


# -------- Save the image names and ratios as dataframe ------------------
df = pd.DataFrame(poly_contribs, index = sum(imgs, []), 
                  columns = ['polygon_ratio'])
df.to_csv('polygon_contribs')


# ------------ Plot the histogram of ratios distribution -----------------
# import matplotlib.pyplot as plt
# plt.hist(poly_contribs, bins = 10, edgecolor = 'red')
# plt.show() # Don't know why plotting is not working here, maybe R
