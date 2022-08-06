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
show_records(train_records[8:10],ncols=2, font_size=50, label_color = '#ffff00')
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
show_preds(preds=preds[8:11])
plt.show()
#If num_workers is anything other than 0, gets this warning, not sure if still is running or not
#[W ParallelNative.cpp:214] Warning: Cannot set number of intraop threads after 
#parallel work has started or after set_num_threads call when using native parallel 
#backend (function set_num_threads)

#Same with the fit
learn.fine_tune(10, 0.0010000000474974513, freeze_epochs =1)
#SuggestedLRs(valley=0.0010000000474974513)

#-------------------------------------------------------------
# After modifying our data, this parser works!
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
        self._size = self._root.find("size")

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.filepath(o))
            record.set_img_size(self.img_size(o))

        record.detection.set_class_map(self.class_map)
        record.detection.add_labels(self.labels(o))
        record.detection.add_bboxes(self.bboxes(o))

    def filepath(self, o) -> Union[str, Path]:
        return self.images_dir / self._filename

    def img_size(self, o) -> ImgSize:
        width = int(self._size.find("width").text)
        height = int(self._size.find("height").text)
        return ImgSize(width=width, height=height)

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
            xml_bbox = object.find("bndbox")
            xmin = to_int(xml_bbox.find("xmin").text)
            ymin = to_int(xml_bbox.find("ymin").text)
            xmax = to_int(xml_bbox.find("xmax").text)
            ymax = to_int(xml_bbox.find("ymax").text)

            bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
            bboxes.append(bbox)

        return bboxes
      
      
import xml.etree.ElementTree as ET
from icevision import *
import icevision.models.torchvision.faster_rcnn as faster_rcnn
#VOC style -- uses xml files 

class_map = ClassMap(['logo', 'polygon', 'chevron', 'circle', 'text', 'quad', 'other', 'triangle', 'exclude',
           'star', 'bowtie', 'line', 'ribbon'])
           
parser = VOCBBoxParser(annotations_dir=r"Modified data/Modified Annotation", 
                       images_dir=r"Modified data/Images", 
                       class_map = class_map)


#Unsure about the parsing function right now. 
train_records, valid_records = parser.parse()
show_records(train_records[479:480], ncols = 1, class_map = class_map)
plt.show()

presize = 512
size = 256

train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

# Train and Validation Dataset Objects
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

# DataLoaders EFFICIENTDET
# train_dl = efficientdet.train_dl(train_ds, batch_size=16
#                                 , num_workers=0 # same error, have to set this to 0
#                                 , shuffle=True)
# valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16
#                                 , num_workers=0
#                                 , shuffle=False)


train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=0, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=0, shuffle=False)

# DataLoaders Faster_rcnn
# show_records(train_records[3:4], ncols=3, class_map=class_map)

# Rewrite EfficientDetBackboneConfig to make it work in model function, EFFICIENTDET
# class BackboneConfig(ABC):
#     @abstractmethod
#     def __call__(self, *args: Any, **kwargs: Any) -> "BackboneConfig":
#         """Completes configuration for this backbone.
#         Called by the end user. All heavy lifting (such as creating a NN backbone)
#         should be done here.
#         """
# 
# class EfficientDetBackboneConfig(BackboneConfig):
#     def __init__(self, model_name: str):
#         self.model_name = model_name
#         self.pretrained = True
# 
#     def __call__(self, pretrained: bool = True) -> "EfficientDetBackboneConfig":
#         """Completes the configuration of the backbone
#         # Arguments
#             pretrained: If True, use a pretrained backbone (on COCO).
#         """
#         self.pretrained = pretrained
#         return self

# Model EFFICIENTDET
# backbone = torchvision.models.mobilenet_v2(pretrained = True).features
# model = efficientdet.model(
#     # backbone = efficientdet.backbones.tf_d0,
#     model_name="efficientdet_d0",
#     create_model('efficientnet_b0'),
#     num_classes=len(class_map),
#     img_size=size)

# model = efficientdet.model(backbone = backbone(), model_name = 'efficientdet_d4',
#      num_classes=len(class_map), img_size=size)
     # TypeError: model() missing 1 required positional argument: 'backbone' 
     
# model = efficientdet.model(
#     model_name=efficientdet.tf_b0,
#     backbone = "tf_efficientdet_lite0",
#     num_classes=len(class_map), img_size=size
# )

# model = efficientdet.model(efficientdet.tf_d0, num_classes=len(class_map), img_size=size)
# 
# model = efficientdet.model(
#     EfficientDetBackboneConfig('efficientdet_d0'),
#     num_classes=len(class_map),
#     img_size=size)

# backbone = backbones.resnet_fpn.resnet18(pretrained=True)
# backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', True)
# 
# backbone = icevision.backbones.resnest_fpn.resnest50_fpn(pretrained = True)
# backbone = backbones.resnet18(pretrained= True)
# 
# backbone = backbones.resnet_fpn.resnet_fpn_backbone('resnet18',pretrained=True)

model = faster_rcnn.model(num_classes=len(class_map)) # default fastercnn_resnet50_fpn

 
# Define metrics
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

# Fastai Learner

# learn = efficientdet.fastai.learner(
#     dls=[train_dl, valid_dl], 
#     model=model,
#     metrics=metrics)

learn = faster_rcnn.fastai.learner(
    dls=[train_dl, valid_dl], model=model, metrics=metrics
)

# Fastai Training
# Learning Rate Finder
learn.freeze()
learn.lr_find()

# learn.fine_tune(20, lr=1e-4)
learn.fine_tune(50, 1e-2, freeze_epochs=2)

# Inference
infer_dl = faster_rcnn.infer_dl(valid_ds, batch_size=16)

# Predict

# samples, preds = faster_rcnn.predict_dl(model, infer_dl)
# no predict_dl function anymore.
# preds3 = faster_rcnn.predict_from_dl(model, infer_dl) # no img info in the ground_truth
# samples4, preds4 = faster_rcnn.predict_from_dl(model, infer_dl) #batch_size=16
# 
# samples5, preds5 = faster_rcnn.predict(model, valid_ds) # pred from dataset, overflow makes R aborted


from icevision.utils import *

def predict_from_dl(
    predict_fn,
    model,
    infer_dl,
    keep_images = False,
    show_pbar = True,
    **predict_kwargs) :
    all_preds = []
    for batch, records in pbar(infer_dl, show=show_pbar):
        preds = predict_fn(
            model=model,
            batch=batch,
            records=records,
            keep_images=keep_images,
            **predict_kwargs,
        )
        all_preds.extend(preds)

    return all_preds
  
def predict_batch(
    model: nn.Module,
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
):
    model.eval()
    device = device or model_device(model)
    batch = [o.to(device) for o in batch]

    raw_preds = model(*batch)
    return convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
    )  

def predict(
    model: nn.Module,
    dataset: Dataset,
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    batch, records = build_infer_batch(dataset)
    return predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
        device=device,
    )
    
    
def convert_raw_predictions(
    batch,
    raw_preds,
    records: Sequence[BaseRecord],
    detection_threshold: float,
    keep_images: bool = False,
):
    return [
        convert_raw_prediction(
            sample=sample,
            raw_pred=raw_pred,
            record=record,
            detection_threshold=detection_threshold,
            keep_image=keep_images,
        )
        for sample, raw_pred, record in zip(zip(*batch), raw_preds, records)
    ]


def convert_raw_prediction(
    sample,
    raw_pred: dict,
    record: BaseRecord,
    detection_threshold: float,
    keep_image: bool = False,
):
    above_threshold = raw_pred["scores"] >= detection_threshold

    # convert predictions
    labels = raw_pred["labels"][above_threshold]
    labels = labels.detach().cpu().numpy()

    scores = raw_pred["scores"][above_threshold]
    scores = scores.detach().cpu().numpy()

    boxes = raw_pred["boxes"][above_threshold]
    bboxes = []
    for box_tensor in boxes:
        xyxy = box_tensor.cpu().numpy()
        bbox = BBox.from_xyxy(*xyxy)
        bboxes.append(bbox)

    # build prediction
    pred = BaseRecord(
        (
            ScoresRecordComponent(),
            ImageRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )
    pred.detection.set_class_map(record.detection.class_map)
    pred.detection.set_scores(scores)
    pred.detection.set_labels_by_id(labels)
    pred.detection.set_bboxes(bboxes)
    pred.detection.above_threshold = above_threshold

    if keep_image:
        tensor_image, *_ = sample
        record.set_img(tensor_to_image(tensor_image))

    return Prediction(pred=pred, ground_truth=record)
  
  
p = predict_from_dl(predict_fn = predict_batch, model=model, infer_dl=infer_dl, keep_images=True)
