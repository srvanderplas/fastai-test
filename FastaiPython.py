# All imports from all portions of the script so far
# ------------------------------------------------------------------------------
from utils import *
from os import listdir

import torch
from torch import nn, LongTensor, FloatTensor
import torch.nn.functional as F

from fastai.basics import ifnone
from fastai.vision.all import *
from fastai.vision.models.unet import _get_sz_change_idxs, hook_outputs
from fastai.layers import init_default, ConvLayer
from fastai.callback.hook import model_sizes
# 

import pdb
import torchvision
from object_detection_metrics.BoundingBox import BoundingBox, BBType, BBFormat
from object_detection_metrics.BoundingBoxes import BoundingBoxes
from object_detection_metrics.Evaluator import Evaluator

# ---- Model Architecture ------------------------------------------------------
def conv2d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias=False, init=nn.init.kaiming_normal_):
    "Create and initialize `nn.Conv2d` layer."
    if padding is None: padding = ks // 2
    return init_default(nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias), init)

class LateralUpsampleMerge(nn.Module):
    "Merge the features coming from the downsample path (in `hook`) with the upsample path."
    def __init__(self, ch, ch_lat, hook):
        super().__init__()
        self.hook = hook
        self.conv_lat = conv2d(ch_lat, ch, ks=1, bias=True)
    
    def forward(self, x):
        return self.conv_lat(self.hook.stored) + F.interpolate(x, self.hook.stored.shape[-2:], mode='nearest')


class RetinaNet(nn.Module):
    "Implements RetinaNet from https://arxiv.org/abs/1708.02002"
    def __init__(self, encoder:nn.Module, n_classes, final_bias=0., chs=256, n_anchors=9, flatten=True):
        super().__init__()
        self.n_classes,self.flatten = n_classes,flatten
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sz_change_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        self.encoder = encoder
        self.c5top5 = conv2d(sfs_szs[-1][1], chs, ks=1, bias=True)
        self.c5top6 = conv2d(sfs_szs[-1][1], chs, stride=2, bias=True)
        self.p6top7 = nn.Sequential(nn.ReLU(), conv2d(chs, chs, stride=2, bias=True))
        self.merges = nn.ModuleList([LateralUpsampleMerge(chs, sfs_szs[idx][1], hook) 
                                     for idx,hook in zip(sfs_idxs[-2:-4:-1], self.sfs[-2:-4:-1])])
        self.smoothers = nn.ModuleList([conv2d(chs, chs, 3, bias=True) for _ in range(3)])
        self.classifier = self._head_subnet(n_classes, n_anchors, final_bias, chs=chs)
        self.box_regressor = self._head_subnet(4, n_anchors, 0., chs=chs)
        
    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256):
        "Helper function to create one of the subnet for regression/classification."
        layers = [ConvLayer(chs, chs, bias=True, norm_type=None) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        layers[-1].weight.data.fill_(0)
        return nn.Sequential(*layers)
    
    def _apply_transpose(self, func, p_states, n_classes):
        #Final result of the classifier/regressor is bs * (k * n_anchors) * h * w
        #We make it bs * h * w * n_anchors * k then flatten in bs * -1 * k so we can contenate
        #all the results in bs * anchors * k (the non flatten version is there for debugging only)
        if not self.flatten: 
            sizes = [[p.size(0), p.size(2), p.size(3)] for p in p_states]
            return [func(p).permute(0,2,3,1).view(*sz,-1,n_classes) for p,sz in zip(p_states,sizes)]
        else:
            return torch.cat([func(p).permute(0,2,3,1).contiguous().view(p.size(0),-1,n_classes) for p in p_states],1)
    
    def forward(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        for merge in self.merges: p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        return [self._apply_transpose(self.classifier, p_states, self.n_classes), 
                self._apply_transpose(self.box_regressor, p_states, 4),
                [[p.size(2), p.size(3)] for p in p_states]]
    
    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

# ------------------------------------------------------------------------------

# ---- Loss function -----------------------------------------------------------
def activ_to_bbox(acts, anchors, flatten=True):
    "Extrapolate bounding boxes on anchors from the model activations."
    if flatten:
        acts.mul_(acts.new_tensor([[0.1, 0.1, 0.2, 0.2]])) #Can't remember where those scales come from, but they help regularize
        centers = anchors[...,2:] * acts[...,:2] + anchors[...,:2]
        sizes = anchors[...,2:] * torch.exp(acts[...,:2])
        return torch.cat([centers, sizes], -1)
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res

def bbox_to_activ(bboxes, anchors, flatten=True):
    "Return the target of the model on `anchors` for the `bboxes`."
    if flatten:
        t_centers = (bboxes[...,:2] - anchors[...,:2]) / anchors[...,2:] 
        t_sizes = torch.log(bboxes[...,2:] / anchors[...,2:] + 1e-8) 
        return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
    else: return [activ_to_bbox(act,anc) for act,anc in zip(acts, anchors)]
    return res

def encode_class(idxs, n_classes):
    target = idxs.new_zeros(len(idxs), n_classes).float()
    mask = idxs != 0
    i1s = LongTensor(list(range(len(idxs))))
    target[i1s[mask],idxs[mask]-1] = 1
    return target

def create_anchors(sizes, ratios, scales, flatten=True):
    "Create anchor of `sizes`, `ratios` and `scales`."
    aspects = [[[s*math.sqrt(r), s*math.sqrt(1/r)] for s in scales] for r in ratios]
    aspects = torch.tensor(aspects).view(-1,2)
    anchors = []
    for h,w in sizes:
        #4 here to have the anchors overlap.
        sized_aspects = 4 * (aspects * torch.tensor([2/h,2/w])).unsqueeze(0)
        base_grid = create_grid((h,w)).unsqueeze(1)
        n,a = base_grid.size(0),aspects.size(0)
        ancs = torch.cat([base_grid.expand(n,a,2), sized_aspects.expand(n,a,2)], 2)
        anchors.append(ancs.view(h,w,a,4))
    return torch.cat([anc.view(-1,4) for anc in anchors],0) if flatten else anchors

def create_grid(size):
    "Create a grid of a given `size`."
    H, W = size
    grid = FloatTensor(H, W, 2)
    linear_points = torch.linspace(-1+1/W, 1-1/W, W) if W > 1 else torch.tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(-1+1/H, 1-1/H, H) if H > 1 else torch.tensor([0.])
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1,2)

def cthw2tlbr(boxes):
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[:,:2] - boxes[:,2:]/2
    bot_right = boxes[:,:2] + boxes[:,2:]/2
    return torch.cat([top_left, bot_right], 1)

def tlbr2cthw(boxes):
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[:,:2] + boxes[:,2:])/2
    sizes = boxes[:,2:] - boxes[:,:2]
    return torch.cat([center, sizes], 1)

def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    "Match `anchors` to targets. -1 is match to background, -2 is ignore."
    matches = anchors.new(anchors.size(0)).zero_().long() - 2
    if targets.numel() == 0: return matches
    ious = IoU_values(anchors, targets)
    vals,idxs = torch.max(ious,1)
    matches[vals < bkg_thr] = -1
    matches[vals > match_thr] = idxs[vals > match_thr]
    return matches

def intersection(anchors, targets):
    "Compute the sizes of the intersections of `anchors` by `targets`."
    ancs, tgts = cthw2tlbr(anchors), cthw2tlbr(targets)
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(a,t,4), tgts.unsqueeze(0).expand(a,t,4)
    top_left_i = torch.max(ancs[...,:2], tgts[...,:2])
    bot_right_i = torch.min(ancs[...,2:], tgts[...,2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0) 
    return sizes[...,0] * sizes[...,1]

def IoU_values(anchs, targs):
    "Compute the IoU values of `anchors` by `targets`."
    inter = intersection(anchs, targs)
    anc_sz, tgt_sz = anchs[:,2] * anchs[:,3], targs[:,2] * targs[:,3]
    union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter
    return inter/(union+1e-8)

class RetinaNetFocalLoss(nn.Module):
    def __init__(self, gamma:float=2., alpha:float=0.25,  pad_idx:int=0, scales=None, ratios=None, reg_loss=F.smooth_l1_loss):
        super().__init__()
        self.gamma,self.alpha,self.pad_idx,self.reg_loss = gamma,alpha,pad_idx,reg_loss
        self.scales = ifnone(scales, [1,2**(-1/3), 2**(-2/3)])
        self.ratios = ifnone(ratios, [1/2,1,2])
        
    def _change_anchors(self, sizes) -> bool:
        if not hasattr(self, 'sizes'): return True
        for sz1, sz2 in zip(self.sizes, sizes):
            if sz1[0] != sz2[0] or sz1[1] != sz2[1]: return True
        return False
    
    def _create_anchors(self, sizes, device:torch.device):
        self.sizes = sizes
        self.anchors = create_anchors(sizes, self.ratios, self.scales).to(device)
    
    def _unpad(self, bbox_tgt, clas_tgt):
        i = torch.min(torch.nonzero(clas_tgt - self.pad_idx)) if sum(clas_tgt) > 0 else 0
        return tlbr2cthw(bbox_tgt[i:]), clas_tgt[i:]-1+self.pad_idx
    
    def _focal_loss(self, clas_pred, clas_tgt):
        encoded_tgt = encode_class(clas_tgt, clas_pred.size(1))
        ps = torch.sigmoid(clas_pred.detach())
        weights = encoded_tgt * (1-ps) + (1-encoded_tgt) * ps
        alphas = (1-encoded_tgt) * self.alpha + encoded_tgt * (1-self.alpha)
        weights.pow_(self.gamma).mul_(alphas)
        clas_loss = F.binary_cross_entropy_with_logits(clas_pred, encoded_tgt, weights, reduction='sum')
        return clas_loss
        
    def _one_loss(self, clas_pred, bbox_pred, clas_tgt, bbox_tgt):
        bbox_tgt, clas_tgt = self._unpad(bbox_tgt, clas_tgt)
        matches = match_anchors(self.anchors, bbox_tgt)
        bbox_mask = matches>=0
        if bbox_mask.sum() != 0:
            bbox_pred = bbox_pred[bbox_mask]
            bbox_tgt = bbox_tgt[matches[bbox_mask]]
            bb_loss = self.reg_loss(bbox_pred, bbox_to_activ(bbox_tgt, self.anchors[bbox_mask]))
        else: bb_loss = 0.
        matches.add_(1)
        clas_tgt = clas_tgt + 1
        clas_mask = matches>=0
        clas_pred = clas_pred[clas_mask]
        clas_tgt = torch.cat([clas_tgt.new_zeros(1).long(), clas_tgt])
        clas_tgt = clas_tgt[matches[clas_mask]]
        return bb_loss + self._focal_loss(clas_pred, clas_tgt)/torch.clamp(bbox_mask.sum(), min=1.)
    
    def forward(self, output, bbox_tgts, clas_tgts):
        clas_preds, bbox_preds, sizes = output
        if self._change_anchors(sizes): self._create_anchors(sizes, clas_preds.device)
        n_classes = clas_preds.size(2)
        return sum([self._one_loss(cp, bp, ct, bt)
                    for (cp, bp, ct, bt) in zip(clas_preds, bbox_preds, clas_tgts, bbox_tgts)])/clas_tgts.size(0)

class SigmaL1SmoothLoss(nn.Module):
    def forward(self, pred, targ):
        reg_diff = torch.abs(targ - pred)
        reg_loss = torch.where(torch.le(reg_diff, 1/9), 4.5 * torch.pow(reg_diff, 2), reg_diff - 1/18)
        return reg_loss.mean()

# ------------------------------------------------------------------------------


# [Model working video, 34:27-47:34](youtube.com/watch?v=5bSVug1YB3s&t=2922s)

# ---- Data Setup --------------------------------------------------------------
#Listing the images as a list for us to get later
IMG_PATH = Path('shoe_imgs')
imgs = os.listdir(IMG_PATH)

# I read that this was a Mac thing that sometimes these ._ files would show up 
# after the above code is run? If these files are not in the list, 
# no need to run. There should be 1299 images
# imgs.remove('._1.jpg')
# imgs.remove('._3.jpg')
# len(imgs)

#Getting the urls and bboxes.
urls, lbl_bbox = get_annotations('data/shoes/shoe_textures.json')
lbl_bbox[0]

#Since we already downloaded our images to the repo, no need to change the urls.
#Function to quickly get the img and associated bounding box
img2bbox = dict(zip(imgs, lbl_bbox))

#Checking that the first bbox lines up with the first image, should match with lbl_bbox[0]
first = {k: img2bbox[k] for k in list(img2bbox)[:1]}
first

# ------------------------------------------------------------------------------

# ---- Data Cleaning -----------------------------------------------------------

#1. Pass in a file name and grab the file. 
#2. Look for that file name's bounding box coordinates
#3. Look for the bounding box class
getters = [lambda o: IMG_PATH/o, lambda o: img2bbox[o][0], lambda o: img2bbox[o][1]]

#Item and batch transforms that keep aspect ratio similar for images
item_tfms = [Resize(224)]
#Keypoint augmentations--don't use transforms that will cause outputs to go off screen
batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize.from_stats(*imagenet_stats)]

#Get our images, no matter what the input, return our list of imgs
def get_train_ims(noop): return imgs

#blocks - inputs and outputs - input image and then output bounding box and its label
#n_inp = 1 is number of inputs is just the one image
model = DataBlock(
blocks = (ImageBlock, BBoxBlock, BBoxLblBlock),
splitter = RandomSplitter(),
get_items = get_train_ims,
getters = getters,
item_tfms = item_tfms,
batch_tfms = batch_tfms,
n_inp=1
)


dls = model.dataloaders('shoe_imgs', bs = 4, num_workers = 0, device = torch.device('cuda'))
dls.c = 7
#Number of classes in this set
get_c(dls)
dls.show_batch()
plt.show()
# ------------------------------------------------------------------------------


# ---- Model Fitting -----------------------------------------------------------
#Cloning the repo that didn't work initially, took necessary functions and initiated them below instead
# !git clone 'https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0.git'
# from imports import *

#Create body of the model
encoder = create_body(resnet34, pretrained=True)
get_c(dls)

# -4 Bias is used because the lecturer said "It was pretty good"
arch = RetinaNet(encoder, get_c(dls), final_bias=-4)

# Example of creating a head of a model-notice it has linear layers at the end that will only output 4 features.
create_head(124, 4)

# Important to note that what we have will get us multiple results. 
# This model head has a smoother, classifier and box_regressor to get everything needed
arch.smoothers
arch.classifier
arch.box_regressor

# For RetinaNet to work, define aspect ratio's and scales of what image should be. 
# As such we will use -1/3 and -2/3 so that anxhor boxes will fit. 
ratios = [1/2, 1, 2]
scales = [1,2**(-1/3), 2**(-2/3)]

# Loss function specified in below chunk
crit = RetinaNetFocalLoss(scales=scales, ratios=ratios)

# Splitting the pre-trained model further. Split into encoder and everything else
def retinanet_split(m): return L(m.encoder,nn.Sequential(m.c5top6, m.p6top7, m.merges, m.smoothers, m.classifier, m.box_regressor)).map(params)


####IMPORTANT####
# I ran into the same error as our R trial when I tried to fit 
# the first time. This code fixed it, not really sure what it does
# Found at https://forums.fast.ai/t/typeerror-no-implementation-found-for-torch-nn-functional-smooth-l1-loss-on-types-that-implement-torch-function-class-fastai-torch-core-tensorimage-class-fastai-vision-core-tensorbbox/90897

TensorImage.register_func(torch.nn.functional.smooth_l1_loss, TensorImage, TensorBBox)
TensorMultiCategory.register_func(TensorMultiCategory.mul, TensorMultiCategory, TensorImage)
TensorImage.register_func(torch.nn.functional.binary_cross_entropy_with_logits, TensorImage, TensorMultiCategory)

# Trying to clear up memory used
# import tensorflow as tf
# tf.keras.backend.clear_session()
torch.cuda.empty_cache()



# ------------------------------------------------------------------------------

# for images,label in enumerate(dls):
#   print(images)
#   return
# 
# 
# 
# # Get classification matrix
# interp = ClassificationInterpretation.from_learner(learn)


# ------------------------------------------------------------------------------
# Object detection metrics code copied from:
# https://github.com/jaidmin/Practical-Deep-Learning-for-Coders-2.0/blob/master/Computer%20Vision/06_Object_Detection_changed.ipynb
def _retinanet_split(m): return L(m.encoder,nn.Sequential(m.c5top6, m.p6top7, m.merges, m.smoothers, m.classifier, m.box_regressor)).map(params)

class ThresholdingAndNMS(Callback):
    def __init__(self, threshold=0.3):
        self.threshold=threshold
    def after_loss(self):
        if self.training: return # only do this expensive computation during validation/show_results
        box_pred, cls_pred = self.learn.pred
        scores = torch.sigmoid(cls_pred)
        anchors = self.learn.loss_func.anchors
        recovered_boxes = torch.clamp(cthw2tlbr(activ_to_bbox(box_pred, anchors).view(-1,4)).view(*box_pred.shape), min=-1, max=1)
        cls_clean, box_clean = [],[]
        
        one_batch_boxes = []
        one_batch_scores = []
        one_batch_cls_pred = [] 
        for i in range(cls_pred.shape[0]):
            cur_box_pred = recovered_boxes[i]
            cur_scores = scores[i]
            max_scores, cls_idx = torch.max(cur_scores, dim=1)
            thresh_mask = max_scores > self.threshold
            
            cur_keep_boxes = cur_box_pred[thresh_mask]
            cur_keep_scores = cur_scores[thresh_mask]
            cur_keep_cls_idx = cls_idx[thresh_mask]
            
            one_img_boxes = []
            one_img_scores = []
            one_img_cls_pred = []
            for c in range(NUM_CLASSES):
                cls_mask   = cur_keep_cls_idx==c
                if cls_mask.sum()==0:
                    continue
                cls_boxes  = cur_keep_boxes[cls_mask]
                cls_scores = cur_keep_scores[cls_mask].max(dim=1)[0]
                nms_keep_idx = torchvision.ops.nms(cls_boxes,cls_scores, iou_threshold=0.5)
                one_img_boxes += [*cls_boxes[nms_keep_idx]]
                one_img_scores += [*cur_keep_scores[nms_keep_idx]]
                one_img_cls_pred += [*tensor([c]*len(nms_keep_idx))]
                
            one_batch_boxes.append(one_img_boxes)
            one_batch_scores.append(one_img_scores)
            one_batch_cls_pred.append(one_img_cls_pred)
        
        
        
        #padded_boxes, padded_cls_pred = pad_and_merge(one_batch_boxes, one_batch_cls_pred)
        #print(f"padded_boxes: {padded_boxes.shape} - padded_cls_pred: {padded_cls_pred.shape}")
        #self.learn.pred = to_device((padded_boxes, padded_cls_pred), cls_pred.device)
        padded_boxes, padded_scores = pad_and_merge_scores(one_batch_boxes, one_batch_scores)
        #print(f"padded_boxes: {padded_boxes.shape} - padded_scores: {padded_scores.shape}")
        self.learn.pred = to_device((padded_boxes, padded_scores), cls_pred.device)
def pad_and_merge_scores(boxes_batch, scores_batch):
    max_n_boxes = max([len(boxes_img) for boxes_img in boxes_batch])
    
    padded_boxes = torch.zeros(len(boxes_batch), max_n_boxes, 4).float()
    padded_scores = torch.zeros(len(boxes_batch), max_n_boxes, NUM_CLASSES).float()
    padded_scores[:,:] = 10 # set all to 10, if its a padded box, this is very ugly, the metric will remove 
    # these rows
    
    for i, (boxes_img, scores_img) in enumerate(zip(boxes_batch, scores_batch)):
        for j, (box, score) in enumerate(zip(boxes_img, scores_img)):
            padded_boxes[i,j] = box
            padded_scores[i,j] = score
    return (TensorBBox(padded_boxes), TensorMultiCategory(padded_scores))
def tlbr2xyxy(box, img_size=(224,224)):
    h,w = img_size  # ????
    # assume shape = (4)
    # converting from pytorch -1 to 1 -> 0 to 1
    #print(f"box shape: {box.shape}")
    box = box.squeeze()
    box = (box + 1) / 2
    x1 = int(box[0]*w)
    x2 = int(box[2]*w)
    y1 = int(box[1]*h)
    y2 = int(box[3]*h)
    return [x1,y1,x2,y2]

class mAP(Metric):
    def __init__(self):
        self.boxes = BoundingBoxes()
        self.count = 0
        self.res = None
    
    def reset(self):
        self.boxes.removeAllBoundingBoxes()
        self.count = 0
    
    def accumulate(self, learn):
        # add predictions and ground truths
        #pdb.set_trace()
        pred_boxes, pred_scores = learn.pred
        # remove padded boxes in batch
        pred_cls = pred_scores.argmax(dim=-1)
        gt_boxes, gt_cls = learn.yb
        #pdb.set_trace()
        for img_box_pred, img_score_pred, img_box_gt, img_cls_gt in zip(pred_boxes, pred_scores, gt_boxes, gt_cls): 
            
            pred_nonzero_idxs = (img_score_pred.sum(dim=-1) < 5).float().nonzero()
            #pdb.set_trace()
            if not pred_nonzero_idxs.numel() == 0:
                img_cls_pred = img_score_pred[pred_nonzero_idxs].argmax(dim=-1)
                #pdb.set_trace()
                #add predictions for this img
                for box_pred, cls_pred, score_pred in zip(img_box_pred[pred_nonzero_idxs], img_cls_pred, img_score_pred[pred_nonzero_idxs]):
                    b = BoundingBox(self.count, learn.dls.vocab[cls_pred.item()+1], *tlbr2xyxy(box_pred), 
                                bbType=BBType.Detected, format=BBFormat.XYX2Y2, classConfidence=score_pred.squeeze()[cls_pred.item()])
                    self.boxes.addBoundingBox(b)
                    #print(f"adding detection {learn.dls.vocab[cls_pred.item()]}")
             #       pdb.set_trace()
            
            gt_nonzero_idxs   = img_cls_gt.nonzero()#.squeeze()
            for box_gt, cls_gt in zip(img_box_gt[gt_nonzero_idxs], img_cls_gt[gt_nonzero_idxs]):
                b = BoundingBox(self.count, learn.dls.vocab[cls_gt.item()], *tlbr2xyxy(box_gt), 
                            bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2)
                self.boxes.addBoundingBox(b)
                #print(f"adding gt {learn.dls.vocab[cls_gt.item()]}")
          #      pdb.set_trace()
            # increment counter
            self.count += 1
    
    @property
    def value(self):
        if len(self.boxes.getBoundingBoxes()) == 0:
            return 0
        self.res = Evaluator().GetPascalVOCMetrics(self.boxes)
        return np.mean([cat["AP"] for cat in self.res])
    
    @property
    def name(self):
        return "mAP"
      
class LookUpMetric(Metric):
    def __init__(self, reference_metric, metric_name, lookup_idx):
        store_attr(self, "reference_metric,metric_name,lookup_idx")
    
    def reset(self):
        pass
    def accumulate(self, learn):
        pass
    
    @property
    def value(self):
        if self.reference_metric.res is None:
            _ = self.reference_metric.value
        return self.reference_metric.res[self.lookup_idx]["AP"]
    
    @property
    def name(self):
        return self.metric_name + "AP"
    
map_metric = mAP()
metrics = [map_metric]

# ------------------------------------------------------------------------------



# ---- Fitting the model -----------------------------------------------------
#BOOM! It worked!
learn = Learner(dls, arch, loss_func=crit, splitter=retinanet_split, 
                cbs=[ThresholdingAndNMS()], metrics=metrics)
# learn.to_fp16()
learn.freeze()
#Find learning rate--I just used the learning rates from the tutorial, ours might be different. 
# learn.lr_find()
# learn.fit_one_cycle(8, slice(1e-5, 1e-4))
learn.fit(8, lr = 7.585775892948732e-05) # too many values to unpack error

learn.fine_tune(4)

learn.save("8-epochs")
#Save the learned model? Look at documentation for this
#learn.save will work
#save_model("models/fastai", learn, opt=True, with_opt=True)
#load_model(file, model, opt, with_opt=True, device=None, strict=True)

#model.save("model_name")
#Load it back in--tf.keras.models.load_model(path to saved model)

# ------------------------------------------------------------------------------
