#Example followed from https://wandb.ai/maria_rodriguez/Surgical_instruments_models_/reports/Choosing-a-Model-for-Detecting-Surgical-Instruments--VmlldzoxMjI4NjQ0#outline
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

#Parser that cycles through our images and annotations
train_parser = parsers.COCOBBoxParser(
    annotations_filepath = 'train.json',
    img_dir = 'train')
test_parser = parsers.COCOBBoxParser(
    annotations_filepath = 'test.json',
    img_dir = 'test')    
    
#The parser defaults to splitting a dataset to 80 train / 20 valid.  
#If your dataset is already split into the different sets, assign a parser for each. 
#Using the SingleSplitSplitter() will keep sets intact.   

whole = SingleSplitSplitter()
train_records, *_ = train_parser.parse(data_splitter = whole)
test_records, *_ = test_parser.parse(data_splitter = whole)

#Showing us an example of the annotated data
show_records(train_records[1:2],ncols=1, font_size=50, label_color = '#ffff00')
plt.show()

#Allowing a relative big presize enables all or most of the image features to be incorporated in the processing.  
presize = 512 
image_size = 384 

#Transforms include scaling, flips, rotations, RGB shifts, lighting shifts, blurring, cropping and padding.
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=presize), tfms.A.Normalize()])
test_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=image_size), tfms.A.Normalize()])

#Was supposed to combine both transforms and orginal, I don't think were combined?
#--I had the same number of images as training and test files
train_ds = Dataset(train_records, train_tfms) 
test_ds = Dataset(test_records, test_tfms)

#Modeling dependencies
from icevision.models.checkpoint import *
from fastai.callback.wandb import *
from fastai.callback.tracker import SaveModelCallback

#The COCOMetric refers to the mean Average Precision, taking into account the 
#predicted bounding box location and the enclosed object's classification.
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

#Beginning first with the basic R-CNN model type
model_type = models.torchvision.faster_rcnn
model = model_type.model(num_classes=len(class_map))

#The dl iterativly feeds in the  data from the #PyTorch Dataset created in the previous step. 
#A bigger batch_size enables the learner to compare more representatives for each iteration
train_dl = model_type.train_dl(train_ds, batch_size=16, num_workers=0, shuffle=True) 
test_dl = model_type.valid_dl(test_ds, batch_size=16, num_workers=0, shuffle=False)

learn = model_type.fastai.learner(dls = [train_dl, test_dl],
                                  model = model, 
                                  metrics = metrics) 
                                  
learn.lr_find()
#If num_workers is anything other than 0, gets this warning, not sure if still is running or not
#[W ParallelNative.cpp:214] Warning: Cannot set number of intraop threads after 
#parallel work has started or after set_num_threads call when using native parallel 
#backend (function set_num_threads)

#Same with the fit
learn.fine_tune(100, 2e-04, freeze_epochs =1)

