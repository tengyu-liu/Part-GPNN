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

import matplotlib.pyplot as plt

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

def get_node_index(bbox, det_boxes):
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    max_iou_index = -1

    for i_node in range(len(det_boxes)):
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

if False:
    meta_dir = '/home/tengyu/Documents/PartGPNN/gpnn/tmp/vcoco/vcoco_features'
    img_dir = '/mnt/hdd-12t/share/HICO/hico_20160224_det/images'
    densepose_path = '/mnt/hdd-12t/tengyu/DensePose/infer_out/hico-det/'
    checkpoint_dir = '/home/tengyu/Documents/github/Part-GPNN/data/hico/model_resnet/finetune_resnet'
    save_data_path = '/home/tengyu/Documents/github/Part-GPNN/data/hico/feature_resnet'
    mmdetection_path = '/mnt/hdd-12t/tengyu/PartGPNN/gpnn/data/hico/mmdetection'
    hico_anno_dir = '/mnt/hdd-12t/share/HICO/hico_20160224_det/annotation'
else:
    meta_dir = '/home/tengyu/Documents/PartGPNN/gpnn/tmp/vcoco/vcoco_features'
    img_dir = '/home/tengyu/Data/hico/hico_20160224_det/images'
    densepose_path = '/home/tengyu/Documents/densepose/DensePoseData/infer_out/hico-det/'
    checkpoint_dir = '/home/tengyu/Documents/github/Part-GPNN/data/hico/model_resnet/finetune_resnet'
    save_data_path = '/home/tengyu/Documents/github/Part-GPNN/data/hico/feature_resnet'
    mmdetection_path = '/home/tengyu/Documents/mmdetection/outputs'
    hico_anno_dir = '/home/tengyu/Data/hico/hico_20160224_det'

input_h, input_w = 224, 224

part_eye = np.eye(21)
obj_eye = np.eye(81)

for imageset in ['test', 'train']:

    hake_annotation = json.JSONDecoder().decode(open(os.path.join(os.path.dirname(__file__), 'annotation', 'hico-%sing-set-image-level.json'%imageset)).read())
    hico_bbox_annotation = sio.loadmat(os.path.join(hico_anno_dir, 'anno_bbox.mat'))['bbox_{}'.format(imageset)]
    mmdetection_result = pickle.load(open(os.path.join(mmdetection_path, 'hico-det.%s.pkl'%imageset), 'rb'))

    for img_i in range(hico_bbox_annotation.shape[1]):
        filename = hico_bbox_annotation[0,img_i][0][0]

        # check if human detection exists
        if not os.path.exists(os.path.join(densepose_path, imageset, filename + '.pkl')):
            warnings.warn('human detection missing for ' + filename)
            continue

        # load image
        try:
            image = skimage.io.imread(os.path.join(img_dir, '%s2015'%imageset, filename))
        except:
            warnings.warn('Image missing ' + filename)
            raise
            continue

        img_w = image.shape[0]
        img_h = image.shape[1]

        if len(image.shape) == 2:
            image = np.tile(np.expand_dims(image, axis=-1), [1, 1, 3])

        obj_boxes_all = np.empty((0,4))
        obj_classes_all = list()
        part_boxes_all = np.empty((0,4))
        part_classes_all = list()
        human_boxes = []
        human_ids = []
        part_human_ids = list()
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
        densepose_boxes, densepose_bodies = pickle.load(open(os.path.join(densepose_path, imageset, filename + '.pkl'), 'rb'), encoding='latin-1')
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
                    part_human_ids.append(human_id)
                    if part_names[part_id] == 'Full Body':
                        human_boxes.append([y0,x0,y1,x1])
                        human_ids.append(human_id)

        # Load annotation
        action_labels = defaultdict(list)
        bbox_annotation = hico_bbox_annotation[0,img_i]

        for hoi_i in range(len(hico_bbox_annotation[0,img_i][2][0])):
            invis = hico_bbox_annotation[0,img_i][2][0][hoi_i][4][0,0]
            if invis == 1: continue
            action = metadata.hoi_to_action[hico_bbox_annotation[0,img_i][2][0][hoi_i][0][0,0]-1]
            if metadata.action_classes[action] == 'no_interaction':
                continue
            bbox_h = hico_bbox_annotation[0,img_i][2][0][hoi_i][1]
            bbox_o = hico_bbox_annotation[0,img_i][2][0][hoi_i][2]
            h_idx = hico_bbox_annotation[0,img_i][2][0][hoi_i][3][0,0]
            o_idx = hico_bbox_annotation[0,img_i][2][0][hoi_i][3][0,1]
            x0_h,y0_h,x1_h,y1_h = int(bbox_h['x1'][0,0][0,0]), int(bbox_h['y1'][0,0][0,0]), int(bbox_h['x2'][0,0][0,0]), int(bbox_h['y2'][0,0][0,0])
            x0_o,y0_o,x1_o,y1_o = int(bbox_o['x1'][0,0][0,0]), int(bbox_o['y1'][0,0][0,0]), int(bbox_o['x2'][0,0][0,0]), int(bbox_o['y2'][0,0][0,0])
            # x0,y0,x1,y1 = min(x0_h, x0_o), min(y0_h, y0_o), max(x1_h, x1_o), max(y1_h, y1_o)
            human_index = get_node_index([x0_h, y0_h, x1_h, y1_h], human_boxes)
            object_index = get_node_index([x0_o, y0_o, x1_o, y1_o], obj_boxes_all)
            if human_index < 0 or object_index < 0:
                continue
            action_labels[(human_ids[human_index], object_index)].append(action)
                    
        # Prepare data
        part_num = len(part_boxes_all)
        obj_num = len(obj_boxes_all)
        human_num = len(human_boxes)
        node_num = part_num + obj_num
        node_features = np.zeros([node_num, 1000])
        edge_features = np.zeros([node_num, node_num, 1000])
        adj_mat = np.zeros([node_num, node_num])
        gt_strength_level = np.zeros([node_num, node_num])
        gt_action_label = np.zeros([node_num, node_num, len(metadata.hoi_to_action)])

        # for i_node in range(node_num):
        #     if i_node < part_num:
        #         box = [int(round(x)) for x in part_boxes_all[i_node]]
        #         print(box)
        #         patch = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
        #         plt.subplot(121)
        #         plt.imshow(image)
        #         plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]])
        #         plt.subplot(122)
        #         plt.imshow(patch)
        #         print(part_names[part_classes_all[i_node]])
        #         plt.show()
        #     else:
        #         box = [int(round(x)) for x in obj_boxes_all[i_node - part_num]]
        #         patch = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
        #         plt.subplot(121)
        #         plt.imshow(image)
        #         plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]])
        #         plt.subplot(122)
        #         plt.imshow(patch)
        #         print(metadata.coco_classes[obj_classes_all[i_node - part_num] + 1])
        #         plt.show()

        # continue

        # for human_index, obj_index in action_labels.keys():
        #     plt.imshow(image)
        #     box = human_boxes[human_ids.index(human_index)]
        #     plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]])
        #     box = obj_boxes_all[obj_index]
        #     plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]])
        #     print([metadata.action_classes[i] for i in action_labels[(human_index, obj_index)]])
        #     plt.show()

        # continue

        # extract node features
        for i_node in range(node_num):
            if i_node < part_num:
                box = part_boxes_all[i_node]
            else:
                box = obj_boxes_all[i_node - part_num]
            box = np.array(box).astype(int)
            img_patch = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
            img_patch = transform(cv2.resize(img_patch, (input_h, input_w), interpolation=cv2.INTER_LINEAR))

            img_patch = torch.autograd.Variable(img_patch).unsqueeze(0).cuda()
            feat, pred = feature_network(img_patch)
            node_features[i_node] = feat.data.cpu().numpy()
        
        part_boxes_all = np.array(part_boxes_all)
        obj_boxes_all = np.array(obj_boxes_all)

        if len(part_boxes_all) == 0 or len(obj_boxes_all) == 0:
            warnings.warn('Zero detection result for {}'.format(filename))
            continue

        node_features_appd = np.zeros([node_features.shape[0], 6 + 21 + 81])
        node_features_appd[:part_num,0] = (part_boxes_all[:,2] - part_boxes_all[:,0]) / img_w # relative w
        node_features_appd[:part_num,1] = (part_boxes_all[:,3] - part_boxes_all[:,1]) / img_h # relative h
        node_features_appd[:part_num,2] = ((part_boxes_all[:,2] + part_boxes_all[:,0]) / 2) / img_w # relative cx
        node_features_appd[:part_num,3] = ((part_boxes_all[:,3] + part_boxes_all[:,1]) / 2) / img_h # relative cy
        node_features_appd[:part_num,4] = (part_boxes_all[:,2] - part_boxes_all[:,0]) * (part_boxes_all[:,3] - part_boxes_all[:,1]) / (img_w * img_h) # relative area
        node_features_appd[:part_num,5] = (part_boxes_all[:,2] - part_boxes_all[:,0]) / (part_boxes_all[:,3] - part_boxes_all[:,1]) # aspect ratio
        node_features_appd[:part_num,6:6+21] = part_eye[part_classes_all]
        
        node_features_appd[part_num:,0] = (obj_boxes_all[:,2] - obj_boxes_all[:,0]) / img_w # relative w
        node_features_appd[part_num:,1] = (obj_boxes_all[:,3] - obj_boxes_all[:,1]) / img_h # relative h
        node_features_appd[part_num:,2] = ((obj_boxes_all[:,2] + obj_boxes_all[:,0]) / 2) / img_w # relative cx
        node_features_appd[part_num:,3] = ((obj_boxes_all[:,3] + obj_boxes_all[:,1]) / 2) / img_h # relative cy
        node_features_appd[part_num:,4] = (obj_boxes_all[:,2] - obj_boxes_all[:,0]) * (obj_boxes_all[:,3] - obj_boxes_all[:,1]) / (img_w * img_h) # relative area
        node_features_appd[part_num:,5] = (obj_boxes_all[:,2] - obj_boxes_all[:,0]) / (obj_boxes_all[:,3] - obj_boxes_all[:,1]) # aspect ratio
        node_features_appd[part_num:,6+21:] = obj_eye[obj_classes_all]

        node_features_appd[np.isnan(node_features_appd)] = 0
        node_features_appd[np.isinf(node_features_appd)] = 0

        node_features = np.concatenate([node_features, node_features_appd], axis=-1)

        # extract edge features
        edge_patch_mapping = {}
        edge_patch_feat = []
        for i_node in range(part_num):
            # we only consider edges connecting at least one part. inter-object edges are not considered
            i_box = part_boxes_all[i_node]
            for j_node in range(i_node + 1, node_num):
                j_box = None

                # j_node is a child of i_node
                if (j_node < part_num and \
                part_human_ids[i_node] == part_human_ids[j_node] and 
                part_names[part_classes_all[j_node]] in part_graph[part_names[part_classes_all[i_node]]]):
                    edge_patch_mapping[(i_node, j_node)] = len(edge_patch_feat)
                    edge_patch_feat.append(node_features[i_node, :1000])

                # j_node is obj and i_node is part
                if j_node >= part_num:
                    j_box = obj_boxes_all[j_node - part_num]
                    box = combine_box(i_box, j_box)
                    box = np.array(box).astype(int)
                    img_patch = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
                    img_patch = transform(cv2.resize(img_patch, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
                    img_patch = torch.autograd.Variable(torch.unsqueeze(img_patch, dim=0)).cuda()
                    feat, pred = feature_network(img_patch)

                    edge_patch_mapping[(i_node, j_node)] = len(edge_patch_feat)
                    edge_patch_feat.append(feat.data.cpu().numpy())

        # Organize edge features
        for i_node in range(node_num):
            for j_node in range(node_num):
                if i_node == j_node:
                    edge_features[i_node, j_node, :1108] = node_features[i_node, :1108]
                    edge_features[i_node, j_node, 1108:] = node_features_appd[j_node]
                    adj_mat[i_node, j_node] = 1
                else:
                    key = (min(i_node, j_node), max(i_node, j_node))
                    if key in edge_patch_mapping:
                        edge_features[i_node, j_node, :1000] = edge_patch_feat[edge_patch_mapping[key]]
                        edge_features[i_node, j_node, 1000:1108] = node_features_appd[i_node]
                        edge_features[i_node, j_node, 1108:] = node_features_appd[j_node]
                        adj_mat[i_node, j_node] = 1
                        # Compute GT Labels and GT signal strength on each edge
                        if i_node < part_num and j_node >= part_num:
                            gt_strength_level[i_node, j_node] = part_weights[part_names[part_classes_all[i_node]]]
                            for label in action_labels[(part_human_ids[i_node], j_node - part_num)]:
                                gt_action_labels[i_node, j_node, label] = 1

                        if j_node < part_num and i_node >= part_num:
                            gt_strength_level[i_node, j_node] = part_weights[part_names[part_classes_all[j_node]]]
                            for label in action_labels[(part_human_ids[j_node], i_node - part_num)]:
                                gt_action_labels[i_node, j_node, label] = 1

        data = {
            'node_features'  : node_features,
            'edge_features'  : edge_features,
            'adj_mat'        : adj_mat, 
            'action_labels'  : gt_action_labels, 
            'strength_level' : gt_strength_level, 
            'part_num'       : part_num, 
            'obj_num'        : obj_num, 
            'human_num'      : human_num, 
            'node_num'       : node_num, 
            'part_human_id'  : part_human_ids, 
            'part_classes'   : part_classes_all, 
            'obj_classes'    : obj_classes_all, 
            'part_boxes'     : part_boxes_all, 
            'obj_boxes'      : obj_boxes_all, 
            'filename'       : filename, 
            'img_w'          : img_w, 
            'img_h'          : img_h
        }
        
        pickle.dump(data, open(os.path.join(save_data_path, filename + '.data'), 'wb'))
        