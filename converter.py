from PIL import Image
import xml.etree.ElementTree as ET
import utils # get_files
import os
    
def convert(file_path):
  '''
  A function modifies our xml file into standard xml file
  Add image size, bndbox elements, remove pt element
  
  Args:-
      file_path : file location of our xml file
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
    xes, ys = [], []
    for obj in root.iter('object'):
        if len(obj) != 0:
            xml_bbox = obj.findall('polygon')
            for bbox in xml_bbox:
                pts = (obj.find('polygon')).findall('pt')
                for pt in pts:
                    xes.append(pt.find('x').text)
                    ys.append(pt.find('y').text)
        #         xml_bbox = obj.findall('polygon')
    #         for each in xml_bbox:
                obj.remove(bbox)
        box_num = int(len(ys)/4)
        # x coordinates lists
        xmins = [xes[i*4] for i in range(box_num)]
        xmaxs = [xes[i*4+1] for i in range(box_num)]
        
        # y coordinates lists
        ymins = [ys[i*4] for i in range(box_num)]
        ymaxs = [ys[i*4+2] for i in range(box_num)]
        
        # compare to get xmin, xmax, ymin, ymax, and write 
        for i in range(box_num):
            xmin, xmax = min(xmins[i], xmaxs[i]), max(xmins[i], xmaxs[i])
            ymin, ymax = min(ymins[i], ymaxs[i]), max(ymins[i], ymaxs[i])
            
            m = obj
            b1 = ET.SubElement(m, 'bndbox')
            c1 = ET.SubElement(b1, 'xmin')
            c1.text = str(xmins[i])
            c2 = ET.SubElement(b1, 'ymin')
            c2.text = str(ymins[i])
            c3 = ET.SubElement(b1, 'xmax')
            c3.text = str(xmaxs[i])
            c4 = ET.SubElement(b1, 'ymax')
            c4.text = str(ymaxs[i])
    t = ET.ElementTree(root)
    with open (file_path, "wb") as files :
            t.write(files)
    files.close()


paths = get_files('/Users/huamuxin/Documents/fastai-test/VOCParser_test/testing', extensions=['.xml']) # get the file paths
for path in paths:
    path=str(path)
    convert(path)
print('-----')
    
