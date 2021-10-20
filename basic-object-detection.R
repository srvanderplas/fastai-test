
reticulate::conda_python('r-reticulate')
library(fastai)
library(magrittr)

# In terminal run the following
# conda activate /home/susan/.local/share/r-miniconda/envs/r-reticulate/
# wget https://raw.githubusercontent.com/airctic/icevision/master/install_colab.sh"
# bash install_colab.sh

fastai:::icevision()

# get file
url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
download.file(url,destfile = "odFridgeObjects.zip")

# Parser
class_map = icevision_ClassMap(c("milk_bottle", "carton", "can", "water_bottle"))
parser = parsers_voc(annotations_dir= "odFridgeObjects/annotations/",
                     images_dir= "odFridgeObjects/images",
                     class_map=class_map)
records = parser$parse()

# Records
train_records = records[[1]]
valid_records = records[[2]]

# Transforms
train_tfms = icevision_Adapter(list(icevision_aug_tfms(size=384, presize=512),
                                    icevision_Normalize()))
valid_tfms = icevision_Adapter(list(icevision_resize_and_pad(384),icevision_Normalize()))

# Datasets
train_ds = icevision_Dataset(train_records, train_tfms)
valid_ds = icevision_Dataset(valid_records, valid_tfms)

# See batch

train_ds %>% show_samples(idx=c(5,10,20,50,100,99), class_map=class_map,
                          denormalize_fn=denormalize_imagenet(),ncols = 3)

