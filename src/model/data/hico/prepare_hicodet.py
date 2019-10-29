"""
Prepare data for Part-GPNN model. 
Need: 
Node feature at different scales
Edge feature for valid edges
Adjacency matrix GT (parse graph GT)
Edge weight (corresponds to node level)
Edge label GT
"""

import os
import json
import pickle
import warnings
from collections import defaultdict

import numpy as np
import scipy.io as sio
import skimage.io
import cv2


part_ids = {'Torso': [1, 2],
            'Right Hand': [3],
            'Left Hand': [4],
            'Left Foot': [5],
            'Right Foot': [6],
            'Upper Leg Right': [7, 9],
            'Upper Leg Left': [8, 10],
            'Lower Leg Right': [11, 13],
            'Lower Leg Left': [12, 14],
            'Upper Arm Left': [15, 17],
            'Upper Arm Right': [16, 18],
            'Lower Arm Left': [19, 21],
            'Lower Arm Right': [20, 22], 
            'Head': [23, 24],
            'Upper Body': [1, 2, 3, 4, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 
            'Lower Body': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
            'Left Arm': [4, 15, 17, 19, 21], 
            'Right Arm': [3, 16, 18, 20, 22], 
            'Left Leg': [5, 8, 10, 12, 14], 
            'Right Leg': [6, 7, 9, 11, 13], 
            'Full Body': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
            }



__PART_WEIGHT_L1 = 0.1 # hand
__PART_WEIGHT_L2 = 0.3 # arm
__PART_WEIGHT_L3 = 0.5 # upper body
__PART_WEIGHT_L4 = 1.0 # human
part_weights = {'Torso': __PART_WEIGHT_L1,
            'Right Hand': __PART_WEIGHT_L1,
            'Left Hand': __PART_WEIGHT_L1,
            'Left Foot': __PART_WEIGHT_L1,
            'Right Foot': __PART_WEIGHT_L1,
            'Upper Leg Right': __PART_WEIGHT_L1,
            'Upper Leg Left': __PART_WEIGHT_L1,
            'Lower Leg Right': __PART_WEIGHT_L1,
            'Lower Leg Left': __PART_WEIGHT_L1,
            'Upper Arm Left': __PART_WEIGHT_L1,
            'Upper Arm Right': __PART_WEIGHT_L1,
            'Lower Arm Left': __PART_WEIGHT_L1,
            'Lower Arm Right': __PART_WEIGHT_L1, 
            'Head': __PART_WEIGHT_L1,
            'Upper Body': __PART_WEIGHT_L3, 
            'Lower Body': __PART_WEIGHT_L3, 
            'Left Arm': __PART_WEIGHT_L2, 
            'Right Arm': __PART_WEIGHT_L2, 
            'Left Leg': __PART_WEIGHT_L2, 
            'Right Leg': __PART_WEIGHT_L2,
            'Full Body': __PART_WEIGHT_L4
            }

part_names = list(part_ids.keys())

part_graph = {'Torso': [],
            'Right Hand': [],
            'Left Hand': [],
            'Left Foot': [],
            'Right Foot': [],
            'Upper Leg Right': [],
            'Upper Leg Left': [],
            'Lower Leg Right': [],
            'Lower Leg Left': [],
            'Upper Arm Left': [],
            'Upper Arm Right': [],
            'Lower Arm Left': [],
            'Lower Arm Right': [], 
            'Head': [],
            'Upper Body': ['Head', 'Torso', 'Left Arm', 'Right Arm'],
            'Lower Body': ['Left Leg', 'Right Leg'],
            'Left Arm': ['Upper Arm Left', 'Lower Arm Left', 'Left Hand'],
            'Right Arm': ['Upper Arm Right', 'Lower Arm Right', 'Right Hand'],
            'Left Leg': ['Upper Leg Left', 'Lower Leg Left', 'Left Foot'],
            'Right Leg': ['Upper Leg Right', 'Lower Leg Right', 'Right Foot'],
            'Full Body': ['Head', 'Torso', 'Upper Body', 'Lower Body']
            }

def get_intersection(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))

def compute_area(box):
    side1 = box[2]-box[0]
    side2 = box[3]-box[1]
    if side1 > 0 and side2 > 0:
        return side1 * side2
    else:
        return 0.0

def compute_iou(box1, box2):
    intersection_area = compute_area(get_intersection(box1, box2))
    iou = intersection_area / (compute_area(box1) + compute_area(box2) - intersection_area)
    return iou

def get_node_index(bbox, det_boxes, index_list):
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    max_iou_index = -1

    for i_node in index_list:
        # check bbox overlap
        iou = compute_iou(bbox, det_boxes[i_node])
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i_node
    return max_iou_index

def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def img_to_torch(img):
    """
    input: H x W x C img iterables with range 0-255
    output: C x H x W img tensor with range 0-1, normalized
    """
    img = np.array(img) / 255.
    img = (img - mean) / std
    if len(img.shape) == 3:
        img = np.expand_dims(img.transpose([2,0,1]), axis=0)
    elif len(img.shape) == 4:
        img = img.transpose([0,3,1,2])
    elif len(img.shape) == 5:
        img = img.transpose([0,1,4,2,3])
    img = torch.autograd.Variable(torch.Tensor(img)).cuda()
    return img

meta_dir = '/home/tengyu/Documents/PartGPNN/gpnn/tmp/vcoco/vcoco_features'
img_dir = '/mnt/hdd-12t/share/HICO/hico_20160224_det/images'
densepose_path = '/mnt/hdd-12t/tengyu/DensePose/infer_out/hico-det/'
checkpoint_dir = '/home/tengyu/Documents/github/Part-GPNN/data/hico/model_resnet/finetune_resnet'
save_data_path = '/home/tengyu/Documents/github/Part-GPNN/data/hico/feature_resnet'
mmdetection_path = '/mnt/hdd-12t/tengyu/PartGPNN/gpnn/data/hico/mmdetection'
hico_anno_dir = '/mnt/hdd-12t/share/HICO/hico_20160224_det/annotation'

input_h, input_w = 224, 224

part_eye = np.eye(21)
obj_eye = np.eye(81)

for imageset in ['train', 'test']:

    hake_annotation = json.JSONDecoder().decode(open(os.path.join(os.path.dirname(__file__), 'annotation', 'hico-%sing-set-image-level.json')).read())
    hico_annotation = sio.loadmat(os.path.join(hico_anno_dir, 'anno.mat'))
    hico_bbox_annotation = sio.loadmat(os.path.join(hico_anno_dir, 'anno_bbox.mat'))

    for filename in annotation:

        

        data = {
            'adj_mat'        : adj_mat, 
            'action_labels'  : gt_action_labels, 
            'strength_level' : gt_strength_level, 
            'part_num'       : part_num, 
            'obj_num'        : obj_num, 
            'human_num'      : human_num, 
            'part_human_id'  : part_human_ids, 
            'part_classes'   : part_classes, 
            'obj_classes'    : obj_classes_all, 
            'part_boxes'     : part_boxes, 
            'obj_boxes'      : obj_boxes_all, 
            'filename'       : filename, 
            'node_num'       : node_num, 
            'img_w'          : img_w, 
            'img_h'          : img_h
        }
        pickle.dump(data, open(os.path.join(save_data_path, filename + '.data'), 'wb'))
        