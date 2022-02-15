library(jsonlite)
# library(tibble)
# library(tidyr)
# library(stringr)
library(magrittr)
library(zeallot) # multi-assign operator %<-%
# library(jpeg)
# suppressMessages(library(dplyr))

library(reticulate)
reticulate::use_virtualenv('r-reticulate')
library(fastai)
library(rjson)
library(imager)


use_python('/Users/jaydenstack/Library/r-miniconda/envs/r-reticulate/bin/python', required = TRUE)
use_condaenv('fastai_in_R')
py_config()

result <- fromJSON(file = 'shoe_textures.json')

cats <- list()
trn_fns <- list()
trn_ids <- list()

for(i in 1:length(result$categories)){
  cats[i] = result$categories[[i]]$name
}
for(i in 1:length(result$images)){
  trn_fns[i] = result$images[[i]]$file_name
}
for(i in 1:length(result$images)){
  trn_ids[i] = result$images[[i]]$id
}

for(i in 1:length(trn_fns)){
  trn_fns[[i]] <- stringr::str_replace(trn_fns[[i]], "https://", "https://www.")
}

im <- load.image(trn_fns[[1]])
plot(im)

DrawOutine <- function(o, lw){
set_path
}

tmrfs = 
md = ImageDataLoaders_from_csv("shoes_num")
#We already have this in the lbl_bbox part
# for (o in 1:length(result$annotations)){
#     bb[o] = as.list(result$annotations[[o]]$bbox)
#     #bb = array(dim = [bb[2], bb[1], bb[4 + bb[1] -1, bb[2] + bb[1] -1]])
# }

im <- load.image(images[1])
im2 = resize(im, round(width(im)/100),round(height(im)/10))
plot(im2)
dev.new(width = 8, height = 6, pointsize = 10)

plot(im2,main="Thumbnail")
plot(im)
plot(cars)
plot(boats)

im


# Modifying code from https://cran.r-project.org/web/packages/fastai/vignettes/obj_detect.html

c(images, lbl_bbox) %<-% get_annotations('shoe_textures.json')
names(lbl_bbox) <- images

# # download image files
# # First add www to all urls
images <- stringr::str_replace(images, "https://", "https://www.")
dir.create("shoe_imgs")
image_paths <- stringr::str_replace(images, "https://www.srvanderplas.com/shoes/all_photos/", "shoe_imgs/")
purrr::walk2(images, image_paths, download.file)
#
image_nums <- 1:length(image_paths)
file.copy(image_paths, file.path("shoes_num", paste0(image_nums, ".jpg")))


names(lbl_bbox) <- paste0(image_nums, ".jpg")
img2bbox = lbl_bbox

im = load.image("shoes_num/3.jpg")
plot(im)

get_y = list(function(o) img2bbox[[o$name]][[1]],
             function(o) as.list(img2bbox[[o$name]][[2]]))

coco = DataBlock(blocks = list(ImageBlock(), BBoxBlock(), BBoxLblBlock()),
                 get_items = get_image_files(),
                 splitter = RandomSplitter(valid_pct = .2, seed = 42),
                 get_y = get_y,
                 item_tfms = Resize(128),
                 batch_tfms = aug_transforms(),
                 n_inp = 1)

dls = coco %>% dataloaders('shoes_num', bs = 5)
dls %>% show_batch(max_n = 6, nrows=1)

# Retinanet components
encoder = create_body(resnet34(), pretrained = TRUE)
get_c(dls)

arch = RetinaNet(encoder, get_c(dls), final_bias=-4)
create_head(124, 4)
arch$smoothers
arch$classifier
arch$box_regressor
ratios = c(1/2,1,2)
scales = c(1,2**(-1/3), 2**(-2/3))

crit = RetinaNetFocalLoss(scales = scales, ratios = ratios)

nn = nn()

retinanet_split = function(m) {
  L(m$encoder,nn$Sequential(m$c5top6, m$p6top7, m$merges,
                            m$smoothers, m$classifier, m$box_regressor))$map(params())
}


# Unfreeze and train

learn = Learner(dls, arch, loss_func = crit, splitter = retinanet_split)
learn$freeze()
learn$summary()
learn %>% fit(10, fastai::slice(1e-5, 1e-4))
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   TypeError: unsupported operand type(s) for *: 'TensorMultiCategory' and 'TensorImage'

# learn = Learner(dls, arch, loss_func = crit)
# learn$freeze()
# learn %>% fit(10, fastai::slice(1e-5, 1e-4))
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   TypeError: no implementation found for 'torch.nn.functional.smooth_l1_loss' on types that implement __torch_function__: [<class 'fastai.torch_core.TensorImage'>, <class 'fastai.vision.core.TensorBBox'>]
