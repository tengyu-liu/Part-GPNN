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
import metadata

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
    hico_bbox_annotation = sio.loadmat(os.path.join(hico_anno_dir, 'anno_bbox.mat'))
    mmdetection_result = pickle.load(open(os.path.join(mmdetection_path, 'outputs', 'hico-det.%s.pkl'%imageset), 'rb'))

    for img_i in range(hico_bbox_annotation.shape[1]):
        filename = hico_bbox_annotation[0,img_i][0][0]

        # check if human detection exists
        if not os.path.exists(os.path.join(densepose_path, imageset, filename + '.pkl')):
            warnings.warn('human detection missing for ' + filename)
            continue

        obj_boxes_all = np.empty((0,4))
        obj_classes_all = list()
        part_boxes_all = np.empty((0,4))
        part_classes_all = list()
        part_human_id = list()
        edge_boxes_all = np.empty((0,4))
        edge_human_id = list()

        # object detection
        for c in range(2, len(metadata.coco_classes)):
            for detection in mmdetection_result[filename][c-1]:
                if detection[4] > 0.7:
                    y0,x0,y1,x1 = detection[0], detection[1], detection[2], detection[3]
                    obj_boxes_all = np.vstack((obj_boxes_all, np.array(detection[:4])[np.newaxis, ...]))
                    obj_classes_all.append(c-1)
        if len(obj_classes_all) == 0:
            warnings.warn('object detection missing for ' + filename)

        # human detection
        densepose_boxes, densepose_bodies = pickle.load(open(os.path.join(densepose_path, imageset, filename + '.pkl'), 'rb'))
        for human_id in range(len(densepose_boxes[1])):
            if densepose_boxes[1][human_id][4] < 0.7:
                continue
            for part_id, part_name in enumerate(part_names):
                x, y = np.where(np.isin(densepose_bodies[1][human_id], part_ids[part_name]))
                x = x + densepose_boxes[1][human_id][1]
                y = y + densepose_boxes[1][human_id][0]
                if len(x) > 0:
                    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
                    part_boxes_all = np.vstack([part_boxes_all, np.array([[y0,x0,y1,x1]])])
                    part_classes_all.append(part_id)
                    part_human_id.append(human_id)
                    # Add edges
                    for obj_box in obj_boxes_all:
                        edge_box = combine_box(obj_box, part_boxes_all[-1,:])
                        edge_human_id.append(human_id)
                        edge_boxes_all = np.vstack([edge_boxes_all, [edge_box]])



        bbox_annotation = hico_bbox_annotation[0,img_i]

        for hoi_i in range(len(hico_bbox_annotation[0,img_i][2][0])):
            invis = hico_bbox_annotation[0,img_i][2][0][hoi_i][4][0,0]
            if invis == 1: continue
            action = metadata.hoi_to_action[hico_bbox_annotation[0,img_i][2][0][hoi_i][0][0,0]-1]
            bbox_h = hico_bbox_annotation[0,img_i][2][0][hoi_i][1]
            bbox_o = hico_bbox_annotation[0,img_i][2][0][hoi_i][2]
            h_idx = hico_bbox_annotation[0,img_i][2][0][hoi_i][3][0,0]
            o_idx = hico_bbox_annotation[0,img_i][2][0][hoi_i][3][0,1]
            x0_h,y0_h,x1_h,y1_h = int(bbox_h['x1'][0,0][0,0]), int(bbox_h['y1'][0,0][0,0]), int(bbox_h['x2'][0,0][0,0]), int(bbox_h['y2'][0,0][0,0])
            x0_o,y0_o,x1_o,y1_o = int(bbox_o['x1'][0,0][0,0]), int(bbox_o['y1'][0,0][0,0]), int(bbox_o['x2'][0,0][0,0]), int(bbox_o['y2'][0,0][0,0])
            # x0,y0,x1,y1 = min(x0_h, x0_o), min(y0_h, y0_o), max(x1_h, x1_o), max(y1_h, y1_o)





        data = {
            'node_features'  : node_features,
            'edge_features'  : edge_features,
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
        