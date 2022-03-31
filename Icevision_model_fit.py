from icevision.all import *
from icevision.models import *

#Needed a quick way to split our json files into test and train
#From the github repo https://github.com/e1-io/echo1-coco-split
# pip install echo1-coco-split
# 
# # Run the coco-split
# coco-split \
#     --has_annotations \
#     --valid_ratio .15 \
#     --test_ratio .05 \
#     --annotations_file ./instances_default.json

#Our object classes
CLASSES = ('Lines', 'Smooth', 'Grid', 'Crepe', 'Fuzzy', 'Leather', 'Other')
class_map = ClassMap(CLASSES)
len(class_map) 

#Parser that cucles through our images and annotations
train_parser = parsers.COCOBBoxParser(
    annotations_filepath = 'test.json',
    img_dir = 'shoe_imgs')
    
whole = SingleSplitSplitter()
train_records, *_ = train_parser.parse(data_splitter = whole)

show_records(train_records[1:2],ncols=1, font_size=50, label_color = '#ffff00')
plt.show()
