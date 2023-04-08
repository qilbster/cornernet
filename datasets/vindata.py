import os

from torch.utils import data
import cv2
import json
import math
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from utils.image import random_crop, crop_image
from utils.image import color_jittering_, lighting_
from utils.image import draw_gaussian, gaussian_radius

CLASS_NAMES = ['__background__','Atelectasis','Calcification','Cardiomegaly','Consolidation',
	'ILD','Infiltration','Lung Opacity','Nodule/Mass','Other lesion','Pleural effusion',
	'Pleural thickening','Pneumothorax','Pulmonary fibrosis', 'Aortic enlargement']

CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]

VINBIG_MEAN = [0.55221958, 0.55221958, 0.55221958] # I took these from our sample_jpegs
VINBIG_STD = [0.01493061, 0.01493061, 0.01493061]

train_df = pd.read_csv('data/train_no_dup.csv')

# Extracting image records only in the sample folder
jpeg_paths = os.listdir('data/jpeg_sample_train')
jpeg_ids = [path.replace(".jpg", "") for path in jpeg_paths]
train_df = train_df[train_df['image_id'].isin(jpeg_ids)]


class VINBIG(Dataset):
  def __init__(self, data_dir, split, data_df=train_df, split_ratio=1.0, gaussian=True, img_size=511):
    super(VINBIG, self).__init__()
    self.split = split
    self.gaussian = gaussian

    self.down_ratio = 4
    self.img_size = {'h': img_size, 'w': img_size}
    self.fmap_size = {'h': (img_size + 1) // self.down_ratio, 'w': (img_size + 1) // self.down_ratio}
    self.padding = 128

    self.data_rng = np.random.RandomState(123)
    self.rand_scales = np.arange(0.6, 1.4, 0.1)
    self.gaussian_iou = 0.3

    self.data_dir = data_dir
    self.img_dir = os.path.join(self.data_dir, 'jpeg_sample_train')
    if split == 'test':
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json')
    else:
      self.annot_path = os.path.join(self.data_dir, 'train.csv')

    self.num_classes = 15
    self.class_name = CLASS_NAMES
    self.valid_ids = CLASS_IDS
    self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

    self.max_objs = 52
    # self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
    # self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
    self.mean = np.array(VINBIG_MEAN, dtype=np.float32)[None, None, :]
    self.std = np.array(VINBIG_STD, dtype=np.float32)[None, None, :]

    self.data_df = data_df
    self.images = data_df['image_id'].unique()

    if 0 < split_ratio < 1:
      split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
      self.images = self.images[:split_size]

    self.num_samples = len(self.images)

    print('Loaded %d %s samples' % (self.num_samples, split))

  def __getitem__(self, index):
    img_id = self.images[index]
    image = cv2.imread(os.path.join(self.img_dir, img_id + '.jpg'))

    labels = self.data_df.query('image_id == @img_id')['class_id'].values
    bboxes = self.data_df.query('image_id == @img_id')[['x_min','y_min','x_max','y_max']].values
    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([0])

    sorted_inds = np.argsort(labels, axis=0)
    bboxes = bboxes[sorted_inds]
    labels = labels[sorted_inds]

    # random crop (for training) or center crop (for validation)
    if self.split == 'train':
      image, bboxes = random_crop(image,
                                  bboxes,
                                  random_scales=self.rand_scales,
                                  new_size=self.img_size,
                                  padding=self.padding)
    else:
      image, border, offset = crop_image(image,
                                         center=[image.shape[0] // 2, image.shape[1] // 2],
                                         new_size=[max(image.shape[0:2]), max(image.shape[0:2])])
      bboxes[:, 0::2] += border[2]
      bboxes[:, 1::2] += border[0]

    # resize image and bbox
    height, width = image.shape[:2]
    image = cv2.resize(image, (self.img_size['w'], self.img_size['h']))
    bboxes[:, 0::2] *= self.img_size['w'] / width
    bboxes[:, 1::2] *= self.img_size['h'] / height

    # discard non-valid bboxes
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.img_size['w'] - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.img_size['h'] - 1)
    keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0,
                               (bboxes[:, 3] - bboxes[:, 1]) > 0)
    bboxes = bboxes[keep_inds]
    labels = labels[keep_inds]

    # randomly flip image and bboxes
    if self.split == 'train' and np.random.uniform() > 0.5:
      image[:] = image[:, ::-1, :]
      bboxes[:, [0, 2]] = image.shape[1] - bboxes[:, [2, 0]] - 1

    image = image.astype(np.float32) / 255.

    # randomly change color and lighting
    # if self.split == 'train':
    #   color_jittering_(self.data_rng, image)
    #   lighting_(self.data_rng, image, 0.1, self.eig_val, self.eig_vec)

    image -= self.mean
    image /= self.std
    image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

    hmap_tl = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)
    hmap_br = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)

    regs_tl = np.zeros((self.max_objs, 2), dtype=np.float32)
    regs_br = np.zeros((self.max_objs, 2), dtype=np.float32)

    inds_tl = np.zeros((self.max_objs,), dtype=np.int64)
    inds_br = np.zeros((self.max_objs,), dtype=np.int64)

    num_objs = np.array(min(bboxes.shape[0], self.max_objs))
    ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
    ind_masks[:num_objs] = 1

    for i, ((xtl, ytl, xbr, ybr), label) in enumerate(zip(bboxes, labels)):
      fxtl = (xtl * self.fmap_size['w'] / self.img_size['w'])
      fytl = (ytl * self.fmap_size['h'] / self.img_size['h'])
      fxbr = (xbr * self.fmap_size['w'] / self.img_size['w'])
      fybr = (ybr * self.fmap_size['h'] / self.img_size['h'])

      ixtl = int(fxtl)
      iytl = int(fytl)
      ixbr = int(fxbr)
      iybr = int(fybr)

      if self.gaussian:
        width = xbr - xtl
        height = ybr - ytl

        width = math.ceil(width * self.fmap_size['w'] / self.img_size['w'])
        height = math.ceil(height * self.fmap_size['h'] / self.img_size['h'])

        radius = max(0, int(gaussian_radius((height, width), self.gaussian_iou)))

        draw_gaussian(hmap_tl[label], [ixtl, iytl], radius)
        draw_gaussian(hmap_br[label], [ixbr, iybr], radius)
      else:
        hmap_tl[label, iytl, ixtl] = 1
        hmap_br[label, iybr, ixbr] = 1

      regs_tl[i, :] = [fxtl - ixtl, fytl - iytl]
      regs_br[i, :] = [fxbr - ixbr, fybr - iybr]
      inds_tl[i] = iytl * self.fmap_size['w'] + ixtl
      inds_br[i] = iybr * self.fmap_size['w'] + ixbr

    return {'image': image,
            'hmap_tl': hmap_tl, 'hmap_br': hmap_br,
            'regs_tl': regs_tl, 'regs_br': regs_br,
            'inds_tl': inds_tl, 'inds_br': inds_br,
            'ind_masks': ind_masks}

  def __len__(self):
    return self.num_samples


class VINBIG_EVAL(VINBIG):
  def __init__(self, data_dir, split, test_scales=(1,), test_flip=False):
    super(VINBIG_EVAL, self).__init__(data_dir, split, gaussian=False)
    self.test_scales = test_scales
    self.test_flip = test_flip

  def __getitem__(self, index):
    img_id = self.images[index]
    image = cv2.imread(os.path.join(self.img_dir, img_id, '.jpg'))
    height, width = image.shape[0:2]

    out = {}
    for scale in self.test_scales:
      new_height = int(height * scale)
      new_width = int(width * scale)

      in_height = new_height | 127
      in_width = new_width | 127

      fmap_height, fmap_width = (in_height + 1) // self.down_ratio, (in_width + 1) // self.down_ratio
      height_ratio = fmap_height / in_height
      width_ratio = fmap_width / in_width

      resized_image = cv2.resize(image, (new_width, new_height))
      resized_image, border, offset = crop_image(image=resized_image,
                                                 center=[new_height // 2, new_width // 2],
                                                 new_size=[in_height, in_width])

      resized_image = resized_image / 255.
      resized_image -= self.mean
      resized_image /= self.std
      resized_image = resized_image.transpose((2, 0, 1))[None, :, :, :]  # [H, W, C] to [C, H, W]

      if self.test_flip:
        resized_image = np.concatenate((resized_image, resized_image[..., ::-1].copy()), axis=0)

      out[scale] = {'image': resized_image,
                    'border': border,
                    'size': [new_height, new_width],
                    'fmap_size': [fmap_height, fmap_width],
                    'ratio': [height_ratio, width_ratio]}

    return img_id, out

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self.valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

          detection = {"image_id": int(image_id),
                       "category_id": int(category_id),
                       "bbox": bbox_out,
                       "score": float("{:.2f}".format(score))}

          detections.append(detection)
    return detections

  def run_eval(self, results, save_dir):
    detections = self.convert_eval_format(results)

    if save_dir is not None:
      result_json = os.path.join(save_dir, "results.json")
      json.dump(detections, open(result_json, "w"))

    coco_dets = self.coco.loadRes(detections)
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

  @staticmethod
  def collate_fn(batch):
    out = []
    for img_id, sample in batch:
      out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
      if k == 'image' else np.array(sample[s][k])[None, ...] for k in sample[s]} for s in sample}))
    return out


if __name__ == '__main__':
  # import pickle
  from tqdm import tqdm

  dataset = COCO('E:\coco_debug', 'train')
  # loader = torch.utils.data.DataLoader(dataset, batch_size=2,
  #                                            shuffle=False, num_workers=0,
  #                                            pin_memory=True, drop_last=True)
  for d in tqdm(dataset):
    pass
  # for d in tqdm(loader):
  #   pass

  dataset = COCO_eval('../data', 'val', test_flip=True, test_scales=[0.5, 0.75, 1, 1.25, 1.5])
  # for d in tqdm(dataset):
  #   pass
  loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=True,
                                             collate_fn=dataset.collate_fn)

  for b in tqdm(loader):
    torch.save(b, '../_debug/imgs2.t7')
    break
    pass

if __name__ == 'x__main__':
  # import pickle
  from tqdm import tqdm

  dataset = COCO('E:\coco_debug', 'train')
  data = dataset[0]
  torch.save(data, '../_debug/db.t7')

  # for d in tqdm(dataset):
  #   pass
