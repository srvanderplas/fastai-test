library(jsonlite)
# library(tibble)
# library(tidyr)
# library(stringr)
library(magrittr)
library(zeallot) # multi-assign operator %<-%
# library(jpeg)
# suppressMessages(library(dplyr))

library(reticulate)
reticulate::conda_python('r-reticulate')
library(fastai)


# Modifying code from https://cran.r-project.org/web/packages/fastai/vignettes/obj_detect.html

c(images, lbl_bbox) %<-% get_annotations('shoe_textures.json')
names(lbl_bbox) <- images

# # download image files
# # First add www to all urls
images <- stringr::str_replace(images, "https://", "https://www.")
# dir.create("shoe_imgs")
image_paths <- stringr::str_replace(images, "https://www.srvanderplas.com/shoes/all_photos/", "shoe_imgs/")
# # purrr::walk2(images, image_paths, download.file)
#
image_nums <- 1:length(image_paths)
# file.copy(image_paths, file.path("shoes_num", paste0(image_nums, ".jpg")))


names(lbl_bbox) <- paste0(image_nums, ".jpg")
img2bbox = lbl_bbox

get_y = list(function(o) img2bbox[[o$name]][[1]],
             function(o) as.list(img2bbox[[o$name]][[2]]))

coco = DataBlock(blocks = list(ImageBlock(), BBoxBlock(), BBoxLblBlock()),
                 get_items = get_image_files(),
                 splitter = RandomSplitter(),
                 get_y = get_y,
                 item_tfms = Resize(128),
                 batch_tfms = aug_transforms(),
                 n_inp = 1)

dls = coco %>% dataloaders('shoes_num', bs = 2)
dls %>% show_batch(max_n = 6)


# Retinanet components
encoder = create_body(resnet34(), pretrained = TRUE)

arch = RetinaNet(encoder, get_c(dls), final_bias=-4)

ratios = c(1/2,1,2)
scales = c(1,2**(-1/3), 2**(-2/3))

crit = RetinaNetFocalLoss(scales = scales, ratios = ratios)

nn = nn()

retinanet_split = function(m) {
  L(m$encoder,nn$Sequential(m$c5top6, m$p6top7, m$merges,
                            m$smoothers, m$classifier, m$box_regressor))$map(params())
}

# Unfreeze and train

# learn = Learner(dls, arch, loss_func = crit, splitter = retinanet_split)
# learn$freeze()
# learn %>% fit(10, fastai::slice(1e-5, 1e-4))
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   TypeError: unsupported operand type(s) for *: 'TensorMultiCategory' and 'TensorImage'


# learn = Learner(dls, arch, loss_func = crit)
# learn$freeze()
# learn %>% fit(10, fastai::slice(1e-5, 1e-4))
# Error in py_call_impl(callable, dots$args, dots$keywords) :
#   TypeError: no implementation found for 'torch.nn.functional.smooth_l1_loss' on types that implement __torch_function__: [<class 'fastai.torch_core.TensorImage'>, <class 'fastai.vision.core.TensorBBox'>]
