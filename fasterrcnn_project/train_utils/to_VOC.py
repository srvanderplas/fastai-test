from lxml import etree

'''This script modifies our xml files to VOC format'''

xml_path = '/Users/huamuxin/Documents/fastai-test/Original Data/Annotations/'
name = '1-state-hedde-2-whiskey-multi-leopard-haircalf_product_9144894_color_784454' # difficult
name2 = 'acorn-acorn-moc-summerweight-stonewash-black-canvas_product_8490840_color_580306' # polygon
p = xml_path + name + '.xml'
p2 = xml_path + name2 + '.xml'
class My_xml():
    def __init__(self, path = xml_path + name + '.xml'):
        self.path = path

    with open(p) as my_xml:
        xml_str = my_xml.read()

    xml = etree.fromstring(xml_str)
    xml_content = self.recur_parse_xml_to_dict(xml)['annotation']



import xmltodict
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

p2
with open(p2) as my_xml2:
    xml_str2 = my_xml2.read()
doc = xmltodict.parse(xml_str2)['annotation']
a_polygon = doc['object'][-3]['polygon']['pt']
b_polygon = doc['object'][-2]['polygon']['pt']
pic2 = '/Users/huamuxin/Documents/fastai-test/Original Data/Images/'+name2+'.jpg'
img = Image.open(pic2).convert('RGBA')
x = [[i['x'] for i in a_polygon], [i['x'] for i in b_polygon]]
y = [[i['y'] for i in a_polygon], [i['y'] for i in b_polygon]]

x = [list(map(int, i)) for i in x]
y = [list(map(int, i)) for i in y]


img2 = img.copy()
draw = ImageDraw.Draw(img2)
draw.polygon([i for i in zip(x[0],y[0])], fill=3, outline=1)
draw.polygon([i for i in zip(x[1],y[1])], fill=4, outline=2)
img2.show()

# size = img.size

