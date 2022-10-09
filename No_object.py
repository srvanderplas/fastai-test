import xml.etree.ElementTree as ET
import glob

no_obj_list = []

paths = get_files('/Users/huamuxin/Documents/fastai-test/Original Data/Annotations', extensions=['.xml']) # get the file paths

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
