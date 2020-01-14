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
import skimage.io

import cv2
import feature_model
import metadata
import torch
import torch.autograd
import torchvision.models
import vsrl_utils as vu

part_ids = {'Right Shoulder': [2],
            'Left Shoulder': [5],
            'Knee Right': [10],
            'Knee Left': [13],
            'Ankle Right': [11],
            'Ankle Left': [14],
            'Elbow Left': [6],
            'Elbow Right': [3],
            'Hand Left': [7],
            'Hand Right': [4],
            'Head': [0],
            'Hip': [8],
            'Upper Body': [2,5,6,3,7,4,0,8],
            'Lower Body': [10,13,11,14,8],
            'Left Arm': [5,6,7],
            'Right Arm': [2,3,4],
            'Left Leg': [8,10,11],
            'Right Leg': [8,13,14],
            'Full Body': [2,5,10,13,11,14,6,3,7,4,0,8], 
            }


__PART_WEIGHT_L1 = 0.1 # hand
__PART_WEIGHT_L2 = 0.3 # arm
__PART_WEIGHT_L3 = 0.5 # upper body
__PART_WEIGHT_L4 = 1.0 # human
part_weights = {'Right Shoulder': __PART_WEIGHT_L1,
            'Left Shoulder': __PART_WEIGHT_L1,
            'Knee Right': __PART_WEIGHT_L1,
            'Knee Left': __PART_WEIGHT_L1,
            'Ankle Right': __PART_WEIGHT_L1,
            'Ankle Left': __PART_WEIGHT_L1,
            'Elbow Left': __PART_WEIGHT_L1,
            'Elbow Right': __PART_WEIGHT_L1,
            'Hand Left': __PART_WEIGHT_L1,
            'Hand Right': __PART_WEIGHT_L1,
            'Head': __PART_WEIGHT_L1,
            'Hip': __PART_WEIGHT_L1,
            'Upper Body': __PART_WEIGHT_L3,
            'Lower Body': __PART_WEIGHT_L3,
            'Left Arm': __PART_WEIGHT_L2,
            'Right Arm': __PART_WEIGHT_L2,
            'Left Leg': __PART_WEIGHT_L2,
            'Right Leg': __PART_WEIGHT_L2,
            'Full Body': __PART_WEIGHT_L4}

part_names = list(part_ids.keys())

part_graph = {'Right Shoulder': [],
            'Left Shoulder': [],
            'Knee Right': [],
            'Knee Left': [],
            'Ankle Right': [],
            'Ankle Left': [],
            'Elbow Left': [],
            'Elbow Right': [],
            'Hand Left': [],
            'Hand Right': [],
            'Head': [],
            'Hip': [],
            'Upper Body': ['Head', 'Hip', 'Left Arm', 'Right Arm'],
            'Lower Body': ['Hip', 'Left Leg', 'Right Leg'],
            'Left Arm': ['Left Shoulder', 'Elbow Left', 'Hand Left'],
            'Right Arm': ['Right Shoulder', 'Elbow Right', 'Hand Right'],
            'Left Leg': ['Hip', 'Knee Left', 'Ankle Left'],
            'Right Leg': ['Hip', 'Knee Right', 'Ankle Right'],
            'Full Body': ['Upper Body', 'Lower Body']
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

meta_dir = os.path.join(os.path.dirname(__file__), '../../../data/vcoco_features')
img_dir = '/home/tengyu/dataset/mscoco/images'
checkpoint_dir = '/home/tengyu/github/Part-GPNN/data/model_resnet_noisy/finetune_resnet'
vcoco_root = '/home/tengyu/dataset/v-coco/data'
save_data_path = '/home/tengyu/github/Part-GPNN/data/feature_resnet_tengyu2'

os.makedirs(save_data_path, exist_ok=True)

feature_network = feature_model.Resnet152(num_classes=len(metadata.action_classes))
feature_network.cuda()
best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
checkpoint = torch.load(best_model_file)
for k in list(checkpoint['state_dict'].keys()):
    if k[:7] == 'module.':
        checkpoint['state_dict'][k[7:]] = checkpoint['state_dict'][k]
        del checkpoint['state_dict'][k]

feature_network.load_state_dict(checkpoint['state_dict'])

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
])

input_h, input_w = 224, 224

part_eye = np.eye(21)
obj_eye = np.eye(81)

vcoco_mapping = {'train': 'train', 'test': 'val', 'val': 'train'}

for imageset in ['train', 'test', 'val']:
    coco = vu.load_coco(vcoco_root)
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), vcoco_root)
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)
    
    image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()

for imageset in ['train', 'test', 'val']:
    coco = vu.load_coco(vcoco_root)
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), vcoco_root)
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)
    
    image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()

    for i_image, image_id in enumerate(image_ids):
        filename = coco.loadImgs(ids=[image_id])[0]['file_name']
        d = filename.split('_')[1][:-4]

        print('%d/%d: %s'%(i_image, len(image_ids), filename))

        # if os.path.exists(os.path.join(save_data_path, filename + '.data')):
        #     continue

        try:
            openpose = json.load(open(os.path.join(os.path.dirname(__file__), '../../../data/openpose/%s2014openpose/%s_keypoints.json'%(vcoco_mapping[imageset], filename[:-4]))))
        except:
            warnings.warn('OpenPose missing ' + os.path.join(os.path.dirname(__file__), '../../../data/openpose/%s2014openpose/%s_keypoints.json'%(vcoco_mapping[imageset], filename[:-4])))
            continue
        try:
            image_meta = pickle.load(open(os.path.join(meta_dir, filename + '.p'), 'rb'), encoding='latin1')
        except:
            warnings.warn('Meta missing ' + filename)
            continue
        try:
            image = skimage.io.imread(os.path.join(img_dir, '%s2014'%d, filename))
        except:
            warnings.warn('Image missing ' + filename)
            continue

        img_w = image.shape[0]
        img_h = image.shape[1]

        if len(image.shape) == 2:
            image = np.tile(np.expand_dims(image, axis=-1), [1, 1, 3])

        obj_boxes_all = image_meta['boxes'][image_meta['human_num']:]
        obj_classes_all = image_meta['classes'][image_meta['human_num']:]

        part_human_ids = []
        part_classes = []
        part_boxes = []

        human_boxes = []

        # human_boxes contains parts at different levels        
        for human_id, human in enumerate(openpose['people']):
            keypoints = np.array(human['pose_keypoints_2d']).reshape([-1,3])
            try:
                h, w, _ = np.max(keypoints[keypoints[:,2] >= 0.7], axis=0) - np.min(keypoints[keypoints[:,2] >= 0.7], axis=0)
            except:
                human_boxes.append([0,0,0,0])
                continue
            if w < 60 or h < 60:
                human_boxes.append([0,0,0,0])
                continue
            for part_id, part_name in enumerate(part_names):
                yxs = keypoints[part_ids[part_name]]
                yxs = yxs[yxs[:,2] > 0.7]
                if len(yxs) == 0:
                    continue
                y0 = int(np.clip(yxs[:,0].min() - w * 0.1, 0, img_w))
                x0 = int(np.clip(yxs[:,1].min() - w * 0.1, 0, img_h))
                y1 = int(np.clip(yxs[:,0].max() + w * 0.1, 0, img_w))
                x1 = int(np.clip(yxs[:,1].max() + w * 0.1, 0, img_h))
                _box = [y0,x0,y1,x1]
                part_boxes.append(_box)
                part_classes.append(part_id)
                part_human_ids.append(human_id)
                if part_names[part_id] == 'Full Body':
                    human_boxes.append([y0,x0,y1,x1])

        part_num = len(part_boxes)
        obj_num = len(obj_boxes_all)
        human_num = len(human_boxes)
        node_num = part_num + obj_num

        labels = defaultdict(list)
        roles = defaultdict(list)

        # Extract GT labels
        for x in vcoco_all:
            if x['label'][i_image, 0] == 1:
                try:
                    action_index = metadata.action_index[x['action_name']]
                    role_bbox = x['role_bbox'][i_image, :] * 1.
                    role_bbox = role_bbox.reshape((-1, 4))
                    bbox = role_bbox[0, :]
                    human_index = get_node_index(bbox, human_boxes, np.arange(human_num))   # node_index uses human box
                    if human_index == -1:
                        warnings.warn('human detection missing')
                        continue
                    assert human_index < human_num
                    for i_role in range(1, len(x['role_name'])):
                        bbox = role_bbox[i_role, :]
                        if np.isnan(bbox[0]):
                            continue
                        obj_index = get_node_index(bbox, obj_boxes_all, np.arange(obj_num))# + human_num
                        if obj_index == - 1:
                            warnings.warn('object detection missing')
                            continue
                        assert obj_index >= 0 and obj_index < obj_num
                        roles[(human_index, obj_index)].append(metadata.role_index[x['role_name'][i_role]] - 1)
                    if len(x['role_name']) == 1:
                        labels[(human_index, -1)].append(action_index - 1)
                except IndexError:
                    warnings.warn('Labels missing for {}'.format(filename))
                    raise
                    pass

        try:
            data = pickle.load(open(os.path.join(save_data_path, filename + '.data'), 'rb'))
        except FileNotFoundError:
            warnings.warn('Data missing for {}'.format(filename))
            continue

        assert part_num == data['part_num']
        assert obj_num == data['obj_num']
        assert np.all(part_human_ids == data['part_human_id'])
        assert np.all(part_classes == data['part_classes'])
        assert np.all(obj_classes_all == data['obj_classes'])
        assert np.all(part_boxes == data['part_boxes'])
        assert np.all(obj_boxes_all == data['obj_boxes'])

        gt_strength_level = data['strength_level']
        gt_action_labels = data['action_labels']

        # Organize edge features
        for i_node in range(node_num):
            for j_node in range(node_num):
                # Compute GT Labels and GT signal strength on each edge
                if i_node < part_num and j_node < part_num and part_human_ids[i_node] == part_human_ids[j_node]:
                    gt_strength_level[i_node, j_node] = max(part_weights[part_names[part_classes[i_node]]], part_weights[part_names[part_classes[j_node]]])
                    for label in labels[(part_human_ids[i_node], -1)]:
                        gt_action_labels[i_node, j_node, label] = 1

        data['action_labels'] = gt_action_labels
        pickle.dump(data, open(os.path.join(save_data_path, filename + '.data'), 'wb'))
        