import cv2
import numpy as np
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.polys import Polygon


def read_kor_path_img(path):
    arr = np.fromfile(path, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def bb2yolo(bb, img_w, img_h, class_nums):
    label = class_nums[bb.label]
    x = (bb.center_x / img_w).round(6)
    y = (bb.center_y / img_h).round(6)
    w = (bb.width / img_w).round(6)
    h = (bb.height / img_h).round(6)
    instance = (label, x, y, w, h)
    return instance

def label2bbs(label, encoding_dict, dataset):
    bbs = []
    if dataset == 'car_person':
        for annotation in label:
            cls = list(annotation['attributes'].values())
            if len(cls):
                cls, *_ = list(annotation['attributes'].values())
                if cls in encoding_dict.keys():
                    encoded_cls = encoding_dict[cls]
                    bb = BoundingBox.from_point_soup(annotation['points'])
                    bb.label = encoded_cls
                    bbs.append(bb)
    elif dataset == 'cityscapes':
        for obj in label['objects']:
            cls = obj['label']
            if cls in encoding_dict.keys():
                encoded_cls = encoding_dict[cls]
                bb = Polygon(obj['polygon']).to_bounding_box()
                bb.label = encoded_cls
                bbs.append(bb)
    return bbs

def get_partial_idxs(n_imgs, chunk_size):
    img_idx_chunks = []
    img_idx_arr = np.arange(n_imgs)
    np.random.shuffle(img_idx_arr)
    for i in range(0, n_imgs, chunk_size):
        partial_idx = img_idx_arr[i:i+chunk_size]
        img_idx_chunks.append(partial_idx)
    return img_idx_chunks
