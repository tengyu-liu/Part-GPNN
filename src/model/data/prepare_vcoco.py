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
import sys
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

local = False
 
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

def get_box(_box, human_boxes_all, used_human):
    max_iou = 0.5
    best_box = None
    best_i = None
    for i, box in enumerate(human_boxes_all):
        if i in used_human:
            continue
        iou = compute_iou(_box, box)
        if iou > max_iou:
            max_iou = iou
            best_box = box
            best_i = i
    return best_i, human_boxes_all[best_i]

def draw_box(box, color='blue'):
    x0,y0,x1,y1 = box
    plt.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], c=color)

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
if local:
    img_dir = '/home/tengyu/Data/mscoco/coco'
    vcoco_root = '/home/tengyu/Data/mscoco/v-coco/data'
    save_data_path = '/home/tengyu/Documents/github/Part-GPNN/data/feature_resnet_tengyu2'
else:
    img_dir = '/home/tengyu/dataset/mscoco/images'
    checkpoint_dir = '/home/tengyu/github/Part-GPNN/data/model_resnet_noisy/finetune_resnet'
    vcoco_root = '/home/tengyu/dataset/v-coco/data'
    save_data_path = '/home/tengyu/github/Part-GPNN/data/feature_resnet_tengyu2'

obj_action_pair = pickle.load(open(os.path.join(os.path.dirname(__file__), 'obj_action_pairs.pkl'), 'rb'))

os.makedirs(save_data_path, exist_ok=True)

if not local:
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

    for i_image, image_id in enumerate(image_ids):
        filename = coco.loadImgs(ids=[image_id])[0]['file_name']
        d = filename.split('_')[1][:-4]

        print('%d/%d: %s'%(i_image, len(image_ids), filename), end='\n', flush=True)

        try:
            openpose = json.load(open(os.path.join(os.path.dirname(__file__), '../../../data/openpose/%s2014openpose/%s_keypoints.json'%(vcoco_mapping[imageset], filename[:-4]))))
            # openpose = json.load(open(os.path.join('../../data/openpose/train2014openpose/%s_keypoints.json'%(filename[:-4]))))
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
        human_boxes_all = image_meta['boxes'][:image_meta['human_num']].astype(int)

        part_human_ids = []
        part_classes = []
        part_boxes = []

        human_boxes = []

        used_human = []
        # human_boxes contains parts at different levels        
        for human in openpose['people']:
            import matplotlib.pyplot as plt
            keypoints = np.array(human['pose_keypoints_2d']).reshape([-1,3])
            # plt.imshow(image)
            # plt.scatter(keypoints[keypoints[:,2] > 0.5, 0], keypoints[keypoints[:,2] > 0.5, 1])
            try:
                h, w, _ = np.max(keypoints[keypoints[:,2] >= 0.5], axis=0) - np.min(keypoints[keypoints[:,2] >= 0.5], axis=0)
            except:
                continue
            if w < 30 or h < 30:
                continue
            for part_id, part_name in enumerate(part_names):
                yxs = keypoints[part_ids[part_name]]
                yxs = yxs[yxs[:,2] > 0.5]
                yxs = yxs[yxs[:,0] >= 0]
                yxs = yxs[yxs[:,0] < img_h]
                yxs = yxs[yxs[:,1] >= 0]
                yxs = yxs[yxs[:,1] < img_w]
                # plt.scatter(yxs[:,0], yxs[:,1], c='black', s=2)
                if len(yxs) == 0:
                    continue
                y0 = int(np.clip(yxs[:,0].min() - h * 0.1, 0, img_h))
                x0 = int(np.clip(yxs[:,1].min() - w * 0.1, 0, img_w))
                y1 = int(np.clip(yxs[:,0].max() + h * 0.1, 0, img_h))
                x1 = int(np.clip(yxs[:,1].max() + w * 0.1, 0, img_w))
                _box = [y0,x0,y1,x1]
                # draw_box(_box)
                part_boxes.append(_box)
                part_classes.append(part_id)
                part_human_ids.append(len(human_boxes))
                if part_name == 'Full Body':
                    if len(used_human) == len(human_boxes_all):
                        human_boxes.append(_box)
                        continue
                    i_box, body_box = get_box(_box, human_boxes_all, used_human)
                    if i_box is None:
                        human_boxes.append(_box)
                        continue
                    human_boxes.append(body_box)
                    part_boxes[-1] = body_box
                    used_human.append(i_box)
            # draw_box(human_boxes[-1])
            # plt.show()
        for i in range(len(human_boxes_all)):
            if i not in used_human:
                part_human_ids.append(len(human_boxes))
                human_boxes.append(human_boxes_all[i])
                part_boxes.append(human_boxes_all[i])
                part_classes.append(part_names.index('Full Body'))
                    
                # import matplotlib.pyplot as plt
                # print(part_names[part_id])
                # # print(image.shape, x0, x1, y0, y1)
                # # print('YXS', yxs)
                # # print('WH', w, h)
                # plt.subplot(121)
                # plt.imshow(image)
                # plt.plot([y0,y0,y1,y1,y0], [x0,x1,x1,x0,x0])
                # plt.subplot(122)
                # plt.imshow(image[x0:x1 + 1, y0:y1 + 1, :])
                # plt.show()

        part_num = len(part_boxes)
        obj_num = len(obj_boxes_all)
        human_num = len(human_boxes)
        node_num = part_num + obj_num

        if os.path.exists(os.path.join(save_data_path, filename + '.data')):
            data = pickle.load(open(os.path.join(save_data_path, filename + '.data'), 'rb'))
            labels = data['labels']
            roles = data['roles']
            if local:
                node_features = data['node_features']
                edge_features = data['edge_features']
                adj_mat = data['adj_mat']
                gt_action_labels = data['action_labels']
                gt_action_roles = data['action_roles']
                gt_strength_level = data['strength_level']
        else:
            labels = defaultdict(list)
            roles = defaultdict(list)
            if local:
                node_features = np.zeros([node_num, 1108])
                edge_features = np.zeros([node_num, node_num, 1216])
                adj_mat = np.zeros((node_num, node_num))
                gt_strength_level = np.zeros([node_num, node_num])
                gt_action_labels = np.zeros([node_num, node_num, len(metadata.action_classes) - 1])
                gt_action_roles = np.zeros([node_num, node_num, len(metadata.action_classes) - 1, len(metadata.roles) - 1])

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
                            warnings.warn('object bbox missing')
                            labels[(human_index, -1)].append(action_index - 1)
                            continue
                        obj_index = get_node_index(bbox, obj_boxes_all, np.arange(obj_num))# + human_num
                        if obj_index == - 1:
                            warnings.warn('object detection missing')
                            labels[(human_index, -1)].append(action_index - 1)
                        else:
                            assert obj_index >= 0 and obj_index < obj_num
                            labels[(human_index, obj_index)].append(action_index - 1)
                            labels[(human_index, -1)].append(action_index - 1)
                            roles[(human_index, obj_index)].append(metadata.role_index[x['role_name'][i_role]] - 1)
                    if len(x['role_name']) == 1:
                        labels[(human_index, -1)].append(action_index - 1)
                except IndexError:
                    warnings.warn('Labels missing for {}'.format(filename))
                    raise
                    pass
        
        data = {
            'labels'         : labels,
            'roles'          : roles,
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
            'img_h'          : img_h,
            'img_id'         : image_id, 
            'd'              : d
        }
        
        if local:
            for human_index, obj_index in labels.keys():
                for i_part, i_human in enumerate(part_human_ids):
                    if obj_index != -1:
                        if i_human == human_index:
                            gt_action_labels[i_part, obj_index + part_num, labels[human_index, obj_index]] = 1
                            gt_action_roles[i_part, obj_index + part_num, labels[human_index, obj_index], roles[human_index, obj_index]] = 1
                    for j_part, j_human in enumerate(part_human_ids):
                        if i_human == human_index and j_human == human_index:
                            gt_action_labels[i_part, j_part, labels[human_index, obj_index]] = 1
        
            data['node_features']  = node_features
            data['edge_features']  = edge_features
            data['adj_mat']        = adj_mat
            data['action_labels']  = gt_action_labels
            data['action_roles']   = gt_action_roles
            data['strength_level'] = gt_strength_level

        pickle.dump(data, open(os.path.join(save_data_path, filename + '.data'), 'wb'))

if local:
    exit()

for fn in os.listdir(save_data_path):
    print(fn)
    data = pickle.load(open(os.path.join(save_data_path, fn), 'rb'))
    d = data['d']
    filename = data['filename']
    part_num = data['part_num']
    obj_num = data['obj_num']
    human_num = data['human_num']
    part_human_ids = data['part_human_id']
    part_classes = data['part_classes']
    obj_classes_all = data['obj_classes']
    part_boxes = data['part_boxes']
    obj_boxes_all = data['obj_boxes']
    filename = data['filename']
    node_num = data['node_num']
    img_w = data['img_w']
    img_h = data['img_h']
    image_id = data['img_id']
    labels = data['labels']
    roles = data['roles']

    image = skimage.io.imread(os.path.join(img_dir, '%s2014'%d, filename))
    if len(image.shape) == 2:
        image = np.tile(np.expand_dims(image, axis=-1), [1, 1, 3])

    node_features = np.zeros([node_num, 1000])
    edge_features = np.zeros((node_num, node_num, 1216))
    adj_mat = np.zeros((node_num, node_num))
    gt_strength_level = np.zeros([node_num, node_num])
    gt_action_labels = np.zeros([node_num, node_num, len(metadata.action_classes) - 1])
    gt_action_roles = np.zeros([node_num, node_num, len(metadata.action_classes) - 1, len(metadata.roles) - 1])
    
    # Extract node features
    if not local:
        for i_node in range(node_num):
            if i_node < part_num:
                box = part_boxes[i_node]
            else:
                box = obj_boxes_all[i_node - part_num]
            box = np.array(box).astype(int)
            img_patch = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
            img_patch = transform(cv2.resize(img_patch, (input_h, input_w), interpolation=cv2.INTER_LINEAR))

            img_patch = torch.autograd.Variable(img_patch).unsqueeze(0).cuda()
            feat, pred = feature_network(img_patch)
            node_features[i_node] = feat.data.cpu().numpy()
        
        part_boxes = np.array(part_boxes)
        obj_boxes_all = np.array(obj_boxes_all)

        if len(part_boxes) == 0 or len(obj_boxes_all) == 0:
            warnings.warn('Zero detection result for {}'.format(filename))
            continue

    node_features_appd = np.zeros([node_features.shape[0], 6 + 21 + 81])
    if not local:
        node_features_appd[:part_num,0] = (part_boxes[:,2] - part_boxes[:,0]) / img_w # relative w
        node_features_appd[:part_num,1] = (part_boxes[:,3] - part_boxes[:,1]) / img_h # relative h
        node_features_appd[:part_num,2] = ((part_boxes[:,2] + part_boxes[:,0]) / 2) / img_w # relative cx
        node_features_appd[:part_num,3] = ((part_boxes[:,3] + part_boxes[:,1]) / 2) / img_h # relative cy
        node_features_appd[:part_num,4] = (part_boxes[:,2] - part_boxes[:,0]) * (part_boxes[:,3] - part_boxes[:,1]) / (img_w * img_h) # relative area
        node_features_appd[:part_num,5] = (part_boxes[:,2] - part_boxes[:,0]) / (part_boxes[:,3] - part_boxes[:,1]) # aspect ratio
        node_features_appd[:part_num,6:6+21] = part_eye[part_classes]
        
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

    # Extract edge features
    edge_patch_mapping = {}
    edge_patch_feat = []
    for i_node in range(part_num):
        # we only consider edges connecting at least one part. inter-object edges are not considered
        i_box = part_boxes[i_node]
        for j_node in range(i_node + 1, node_num):
            j_box = None
            if (j_node < part_num and \
            part_human_ids[i_node] == part_human_ids[j_node] and 
            part_names[part_classes[j_node]] in part_graph[part_names[part_classes[i_node]]]):
                j_box = part_boxes[j_node]
            if j_node >= part_num:
                j_box = obj_boxes_all[j_node - part_num]
            if j_box is not None:
                if local:
                    edge_patch_mapping[(i_node, j_node)] = len(edge_patch_feat)
                    edge_patch_feat.append(0)
                else:
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
                if i_node < part_num:
                    gt_strength_level[i_node, i_node] = part_weights[part_names[part_classes[i_node]]]
                    for label in labels[(part_human_ids[i_node], -1)]:
                        gt_action_labels[i_node, i_node, label] = 1
            else:
                # Compute GT Labels and GT signal strength on each edge
                if i_node < part_num and j_node < part_num and part_human_ids[i_node] == part_human_ids[j_node]:
                    gt_strength_level[i_node, j_node] = max(part_weights[part_names[part_classes[i_node]]], part_weights[part_names[part_classes[j_node]]])
                    for label in labels[(part_human_ids[i_node], -1)]:
                        gt_action_labels[i_node, j_node, label] = 1

                if i_node < part_num and j_node >= part_num:
                    gt_strength_level[i_node, j_node] = part_weights[part_names[part_classes[i_node]]]
                    for label in labels[(part_human_ids[i_node], j_node - part_num)]:
                        gt_action_labels[i_node, j_node, label] = 1
                    for role in roles[(part_human_ids[i_node], j_node - part_num)]:
                        gt_action_roles[i_node, j_node, label, role] = 1

                if j_node < part_num and i_node >= part_num:
                    gt_strength_level[i_node, j_node] = part_weights[part_names[part_classes[j_node]]]
                    for label in labels[(part_human_ids[j_node], i_node - part_num)]:
                        gt_action_labels[i_node, j_node, label] = 1
                    for role in roles[(part_human_ids[j_node], i_node - part_num)]:
                        gt_action_roles[i_node, j_node, label, role] = 1

                key = (min(i_node, j_node), max(i_node, j_node))
                if key in edge_patch_mapping:
                    edge_features[i_node, j_node, :1000] = edge_patch_feat[edge_patch_mapping[key]]
                    edge_features[i_node, j_node, 1000:1108] = node_features_appd[i_node]
                    edge_features[i_node, j_node, 1108:] = node_features_appd[j_node]
                    adj_mat[i_node, j_node] = 1

    pairwise_action_mask = np.zeros([node_num, node_num, 27])
    for i_obj in range(part_num, node_num):
        pairwise_action_mask[:, i_obj, :] = obj_action_pair[[obj_classes_all[i_obj - part_num]]]
        pairwise_action_mask[i_obj, :, :] = obj_action_pair[[obj_classes_all[i_obj - part_num]]]
    for i_part in range(part_num):
        for j_part in range(part_num):
            if part_human_ids[i_part] == part_human_ids[j_part]:
                pairwise_action_mask[i_part, j_part, [2,5,16,21,24]] = 1.0
    if np.sum(gt_action_labels) == 0:
        assert sum(len(x) for x in labels.values()) == 0
    data = {
        'node_features'  : node_features,
        'edge_features'  : edge_features,
        'adj_mat'        : adj_mat, 
        'action_labels'  : gt_action_labels,
        'action_roles'   : gt_action_roles,
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
        'img_h'          : img_h,
        'img_id'         : image_id, 
        'pairwise_action_mask' : pairwise_action_mask
    }
    
    pickle.dump(data, open(os.path.join(save_data_path, fn), 'wb'))
    