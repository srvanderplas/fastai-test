library(jsonlite)
reticulate::conda_python('r-reticulate')
library(fastai)
library(magrittr)
library(zeallot) # multi-assign operator %<-%
fastai:::icevision()

# Modifying code from https://cran.r-project.org/web/packages/fastai/vignettes/obj_detect.html
tmp <- read_json("project-9-at-2022-01-20-18-13-56892e24.json")

c(images, lbl_bbox) %<-% get_annotations('result.json')
names(lbl_bbox) <- images
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



dls = coco %>% dataloaders('coco_tiny/train')
dls %>% show_batch(max_n = 12)
