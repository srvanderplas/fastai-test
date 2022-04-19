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
CLASSES = ('Lines', 'Smooth', 'Grid', 'Fuzzy', 'Leather', 'Other')#'Crepe'
class_map = ClassMap(CLASSES)
len(class_map) 

#COCO style -- uses json files
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

train_parser.class_map

#Showing us an example of the annotated data
show_records(train_records[7:8],ncols=1, font_size=50, label_color = '#ffff00')
plt.show()

#Allowing a relative big presize enables all or most of the image features to be incorporated in the processing.  
presize = 512 
image_size = 256

#Transforms include scaling, flips, rotations, RGB shifts, lighting shifts, blurring, cropping and padding.
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize = presize), tfms.A.Normalize()])
test_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=image_size), tfms.A.Normalize()])

#Was supposed to combine both transforms and orginal, I don't think were combined?
#--I had the same number of images as training and test files
train_ds = Dataset(train_records, train_tfms) 
test_ds = Dataset(test_records, test_tfms)


samples = [train_ds[2] for _ in range(3)]
show_samples(samples, ncols=3)
plt.show()
#Modeling dependencies
from icevision.models.checkpoint import *
from fastai.callback.wandb import *
from fastai.callback.tracker import SaveModelCallback

#The COCOMetric refers to the mean Average Precision, taking into account the 
#predicted bounding box location and the enclosed object's classification.
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

#Beginning first with the basic R-CNN model type
model_type = models.torchvision.faster_rcnn
backbone = model_type.backbones.resnet50_fpn(pretrained=True)
model = model_type.model(num_classes=len(class_map))
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(class_map)) 
backbone.__dict__

#The dl iterativly feeds in the  data from the #PyTorch Dataset created in the previous step. 
#A bigger batch_size enables the learner to compare more representatives for each iteration
train_dl = model_type.train_dl(train_ds, batch_size=4, num_workers=0, shuffle=True) 
test_dl = model_type.valid_dl(test_ds, batch_size=4, num_workers=0, shuffle=False)

learn = model_type.fastai.learner(dls = [train_dl, test_dl],
                                  model = model, 
                                  metrics = metrics) 
                                  
learn.lr_find()

model_type.show_results(model,test_ds)

infer_dl = model_type.infer_dl(test_ds, batch_size=1, shuffle=False)
preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)
show_preds(preds=preds[9:10])
plt.show()
#If num_workers is anything other than 0, gets this warning, not sure if still is running or not
#[W ParallelNative.cpp:214] Warning: Cannot set number of intraop threads after 
#parallel work has started or after set_num_threads call when using native parallel 
#backend (function set_num_threads)

#Same with the fit
learn.fine_tune(10, 0.0010000000474974513, freeze_epochs =1)
#SuggestedLRs(valley=0.0010000000474974513)

#-------------------------------------------------------------
#Utilzing now the VOC parser for our data. Needed to change the parser to fit our files

class VOCBBoxParser(Parser):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        class_map: Optional[ClassMap] = None,
        idmap: Optional[IDMap] = None,
    ):
        super().__init__(template_record=self.template_record(), idmap=idmap)
        self.class_map = class_map or ClassMap().unlock()
        self.images_dir = Path(images_dir)

        self.annotations_dir = Path(annotations_dir)
        self.annotation_files = get_files(self.annotations_dir, extensions=[".xml"])

    def __len__(self):
        return len(self.annotation_files)

    def __iter__(self):
        yield from self.annotation_files

    def template_record(self) -> BaseRecord:
        return BaseRecord(
            (
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

    def record_id(self, o) -> Hashable:
        return str(Path(self._filename).stem)

    def prepare(self, o):
        tree = ET.parse(str(o))
        self._root = tree.getroot()
        self._filename = self._root.find("filename").text
        # self._size = self._root.find("size")

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.filepath(o))
            # record.set_img_size(self.img_size(o))

        record.detection.set_class_map(self.class_map)
        record.detection.add_labels(self.labels(o))
        record.detection.add_bboxes(self.bboxes(o))

    def filepath(self, o) -> Union[str, Path]:
        return self.images_dir / self._filename

    # def img_size(self, o) -> ImgSize:
    #     width = int(self._size.find("width").text)
    #     height = int(self._size.find("height").text)
    #     return ImgSize(width=width, height=height)

    def labels(self, o) -> List[Hashable]:
        labels = []
        for object in self._root.iter("object"):
            label = object.find("name").text
            labels.append(label)

        return labels

    def bboxes(self, o) -> List[BBox]:
        def to_int(x):
            return int(float(x))

        bboxes = []
        for object in self._root.iter("object"):
            xml_bbox = object.find("polygon")
            x = to_int(xml_bbox.find("x").text)
            y = to_int(xml_bbox.find("y").text)

            bbox = BBox.from_xyxy(x, y, x, y)
            bboxes.append(bbox)

        return bboxes
      
      
import xml.etree.ElementTree as ET
#VOC style -- uses xml files 
parser = VOCBBoxParser(annotations_dir="Original Data/Annotations", 
  images_dir="Original Data/Images", 
  class_map = ClassMap(["bowtie", "chevron", "circle", "line", "polygon",
    "quad", "star", "text", "triangle", "exclude"]))

#Need to figure out how to only get the singular classes out of the multi ones
#Unsure about the parsing function right now. 
train_records, valid_records = parser.parse()
parser.class_map


