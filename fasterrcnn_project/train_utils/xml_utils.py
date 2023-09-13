import os.path
import PIL.ImageFont as ImageFont
import xmltodict
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import glob
import os
import numpy as np
from numpy.linalg import norm


xml_root_dir = '/Users/huamuxin/Documents/fastai-test/Original Data/Annotations/'

def get_names_from_xml_path(xml_path):
    '''
    This function gets name from a xml file, this can be used to identify images.

    :param xml_path: A string. Xml file path.

    :return: A string. File name without extension that shared by a xml and an image.
    '''
    return os.path.basename(xml_path).split('.')[0]

def get_all_xmls_paths(dir):
    '''
    This function gets all names of xml files in a given directory.

    :param dir: A string. A directory that has xmls files

    :return: A list. All xml names under the directory.
    '''
    if dir[-1]=='/':
        paths = glob.glob(dir+'*.xml')
    else:
        paths = glob.glob(dir+'/.xml')
    return paths


def get_xml_dict(xml_path):
    '''
    This function gets an xml file path and returns a dictionary of corresponding annotations

    :param xml_path: A string. Path of an xml file.
    :return: A dictionary. This dictionary has all the information in the xml files
    '''

    with open(xml_path) as my_xml:
        xml_str = my_xml.read()

    xml = xmltodict.parse(xml_str)
    xml_dict = xml['annotation']

    return xml_dict

def get_labels(xml_dict):
    '''

    :param xml_dict:
    :return:
    '''
    objects = xml_dict['object']
    labels = []
    for object in objects:
        labels.append(object['name'])
    return labels

def get_types(xml_dict, no_type_pass=True):
    # This function is used to find out if we have any type other than bound_box.
    # Seems like most polygons are exclude.
    '''
    This function gets types in an xml file.

    :param xml_dict: A dict. A dictionary of xml file information, recommend output of get_xml_dict.
    :param no_type_pass: Ture/False. There are xmls has no objects, or object has no type attribute,
                         default is true to not recording image names for these cases.

    :return: A list. All types in the xml files, have duplicates.
    '''
    types_in_one = []
    try:
        objects = xml_dict['object']
        for i in range(len(objects)):
            try:
                type = (objects[i]['type'])
            except Exception as e:  # if no type attribute, we will handle by recording the filename or an empty string.
                if no_type_pass == False:
                    type = (objects['filename'])
                else:
                    type = ''
            finally:
                types_in_one.append(type)
    except Exception as e:
        if no_type_pass == False:
            types_in_one.append(xml_dict['filename'])
        else:
            pass
    return types_in_one


def all_types(xml_root_dir):
    '''
    This function returns unique object types in all xml files in the directory.

    :param xml_root_dir: A string. Directory of xml files.

    :return: A list. All unique types, mostly bounding_box.
    '''
    paths = get_all_xmls_paths(xml_root_dir)
    types = []
    for path in paths:
        d = get_xml_dict(path)
        types += get_types(d)
    types = list(set(types))
    return types

def get_coordinates(xml_dict):
    '''
    This function gets annotated coordinates in the xml_dict.

    :param xml_dict: A dict. A dictionary of xml file information, recommend output of get_xml_dict.

    :return: A dictionary. Has key 'x' and 'y', storing corresponding coordinates.
    '''
    try:
        objects = xml_dict['object']
        polygons = [obj['polygon']['pt'] for obj in objects]
        coor_dict = {'x' : [[i['x'] for i in polygon] for polygon in polygons],
                     'y' : [[i['y'] for i in polygon] for polygon in polygons]}
        return coor_dict
    except Exception as e:
        print(e, "Maybe no object in this xml")


def is_bounding_box(x, y):
    '''
    This function check if the coordinate is a bounding box.

    :param x: A list. Coordinate of x's.
    :param y: A list. Coordinate of y's.

    :return: Ture/False. True means the coordinate is a box
    '''
    if len(x) ==4 and len(y) ==4:
        if len(set(x)) == 2 and len(set(y)) == 2:
            return True
        else:
            return False
    else:
        return False

def add_polygon(img, x, y, labels, alpha=0.5, font = 'arial.ttf', font_size=24) :
    '''
    This function adds polygons to the image object.

    :param img: A PIL.Image.Image object.
    :param x: A list of x coordinates. X coordinates in integers.
    :param y: A list of y coordinates. Y coordinates in integers.
    :param alpha: A float' default to 0.5. Opacity of the polygons.
    :param labels: A list of all labels in the xml file.
    :param font: string. Font to be displayed.
    :param font_size: int.
    '''
    x = [list(map(int, i)) for i in x]
    y = [list(map(int, i)) for i in y]

    # try:
    #     font = ImageFont.truetype('font.ttf', 400)
    # except IOError:
    #     font = ImageFont.load_default()

    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    for i in range(len(x)):
        coordinate = [coord for coord in zip(x[i],y[i])]
        draw.polygon(coordinate, fill='wheat', outline=10)

        # text_box = font.getbbox(labels[i])
        # text_width = text_box[2] - text_box[0]
        # text_height = text_box[3] - text_box[1]
        # margin = int(0.05 * text_width)

        draw.text(coordinate[0],
            # (coordinate[0][0]+margin, coordinate[0][1]),
                   labels[i], fill="black", font_size=400)
    img3 = Image.blend(img, img2, alpha=alpha)
    return img3


def get_width_and_height(img_path):
    img = Image.open(img_path).convert('RGBA')
    width = img.size[0]
    height = img.size[1]
    area = width * height
    return {'width': width,
            'height': height,
            'area': area}

def visualize_current_xml(name,
                          xml_root_dir='/Users/huamuxin/Documents/fastai-test/Original Data/Annotations/',
                          img_root_dir='/Users/huamuxin/Documents/fastai-test/Original Data/Images/',
                          alpha=0.5):
    '''
    This function visualizes current annotation.

    :param name: A string. Name of a xml and image file. They must share the same name.
    :param xml_root_dir: A string. Xml root directory.
    :param img_root_dir: A string. Image root directory.
    '''
    img_path = img_root_dir + name + '.jpg'
    xml_path = xml_root_dir + name + '.xml'

    img = Image.open(img_path).convert('RGBA')
    xml_dict = get_xml_dict(xml_path)
    coordinates = get_coordinates(xml_dict)
    labels = get_labels(xml_dict)

    visualizing = add_polygon(img, coordinates['x'], coordinates['y'], alpha=alpha, labels=labels)
    visualizing.show()

def vec_angles(vec0, vec1):
    angle = np.arccos(np.dot(vector0, vector1)/norm(vector0)/norm(vector1))
    return np.degrees(angle)

def get_angles(x,y):
    # This is for oriented boxes
    '''
    This function gets the pojnt has a angle that most close to 90 degree.
    :param x:
    :param y:
    :return:
    '''
    coordinates = [i for i in zip(x,y)]
    points = [np.array(point, dtype=int) for point in coordinates]
    angles = []
    for i in range(4):
        vector0 = points[(i-1)%4] - points[i]
        vector1 = points[(i+1)%4] - points[i]
        angles.append(vec_angles(vector0, vector1)-90)

def crude_fix_box(x, y):
    '''
    This function fix a quadrilateral to a vertical box in a crude way, which is taking the maximum.

    :param x:
    :param y:
    :return:
    '''
    x_max, x_min = max(x), min(x)
    y_max, y_min = max(y), min(y)
    x = [x_max, x_max, x_min, x_min]
    y = [y_max, y_min, y_min, y_max]
    return {'x': x, 'y': y}


# def make_a_quadrilateral_to_box(x, y):

# def rorate_box_scheme(x, y):

# visualize_current_xml("1-state-hedde-2-whiskey-multi-leopard-haircalf_product_9144894_color_784454",
#                       alpha = 0.3)
#
# xml_paths = get_all_xmls_paths(xml_root_dir)[::10]
#
# names = [ get_names_from_xml_path(xml) for xml in xml_paths]
#
visualize_current_xml(names[4])

coordinates = get_coordinates(get_xml_dict(xml_paths[2]))
b = 3
coordinates = [i for i in zip(coordinates['x'][b], coordinates['y'][b])]
c
import cv2
img = cv2.imread()



