# ---- Setup ----
# reticulate::install_miniconda()
# reticulate::conda_create('r-reticulate')
## This creates a r-reticulate conda instance inside
## ~/.local/share/r-miniconda/envs/r-reticulate (on my machine)

# devtools::install_github('eagerai/fastai')
## This step took forever but did originally finish. You may need to use gpu = F
## because I have a cuda-capable gpu on my desktop...
# fastai::install_fastai(gpu = T, cuda_version = '10', overwrite = FALSE)

library(fastai)
URLs_PETS() # Must restart session for this to work, not sure why?
path = 'oxford-iiit-pet'
path_anno = 'oxford-iiit-pet/annotations'
path_img = 'oxford-iiit-pet/images'
fnames = get_image_files(path_img)

dls = ImageDataLoaders_from_name_re(
  path, fnames, pat='(.+)_\\d+.jpg$',
  item_tfms = RandomResizedCrop(460, min_scale=0.75), bs = 10,
  batch_tfms = list(aug_transforms(size = 299, max_warp = 0),
                    Normalize_from_stats( imagenet_stats() )
  ),
  device = 'cuda'
)

dls %>% show_batch()


learn = cnn_learner(dls, resnet50(), metrics = error_rate)
learn %>% fit_one_cycle(n_epoch = 8)


learn$unfreeze()

learn %>% fit_one_cycle(3, lr_max = fastai::slice(1e-6,1e-4))


interp = ClassificationInterpretation_from_learner(learn)
interp %>% most_confused()

