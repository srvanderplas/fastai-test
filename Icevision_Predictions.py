#Example found at https://medium.com/@yrodriguezmd/creating-a-labelled-dataset-using-a-pretrained-model-17036f03e411
#Under A. Loading, click on the link at the end "See an example here"

#Dependencies
from icevision.all import *
from icevision.models import *
import icevision
import mmcv

#Locate, fetch,and open images
IMG_PATH = Path('shoe_imgs2')
img_files = get_image_files(IMG_PATH)
img = PIL.Image.open(img_files[0])

#Resizing images
img.to_thumb(150,150)

#Our object classes, would work if if the pre-trained model weights were updated
CLASSES = ('Lines', 'Smooth', 'Grid', 'Crepe', 'Fuzzy', 'Leather', 'Other')
class_map = ClassMap(CLASSES)
len(class_map)  

#testing where and what classes we should expect to see 
print(class_map.get_by_name('Smooth'))         # output: 2
print(class_map.get_by_name('Grid'))

#Import pre-trained model
from icevision.models.checkpoint import *

#Utilizing retinanet model - ResNet and the Feature Pyramid Network. 
#This code is different than example. Locations of these models have changed in the package since example was created.
#It addresses the imbalanced caused by backgrounds, as well as improves detection in different scales.
model_type = models.torchvision.retinanet
backbone = model_type.backbones.resnet50_fpn
model = model_type.model(backbone=backbone(pretrained=True),
                         num_classes=len(class_map))
backbone.__dict__                      
    
#This  generates predictions. This turns off the batch normalization and drop 
#out steps which are normally utilized during training and not during inference.   
model.eval()                  

#Opens our images and fixes any color issues. Important that there aren't very many photos
#in the file. Using our entire dataset crashed my computer. 15 images worked.
imgs_array = [PIL.Image.open(file) for file in img_files]
imgs_array = [image.convert('RGB') for image in imgs_array]

#Transformations. Limited to resizing, padding and normalization, so don't expect to
#see significantly transformed images like for training.
img_size = 384
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])
infer_ds = Dataset.from_images(imgs_array, valid_tfms, class_map = class_map)
samples = [infer_ds[0] for _ in range(3)]
show_samples(samples, 
             denormalize_fn = denormalize_imagenet, ncols=3)
plt.show()
             
#predict_from_dl uses the model to generate classification and bounding box predictions for the objects in the images.
#Could be useful for phots not already labeled.
#Note the result will not have predictions here until weights are updated.
infer_dl = model_type.infer_dl(infer_ds, batch_size=4, shuffle=False)
preds_saved = model_type.predict_from_dl(model, infer_dl,
                                          keep_images=True,)
                                          
show_preds(preds_saved, font_size=30, label_color='#ffff00')    
plt.show()

#Generating annotation file, will give prediciton metrics
show_sample(preds_saved[3])
preds_saved[3].pred

#More on this example for creating a dataset with the annotations. 


    
