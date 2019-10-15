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
import pickle
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
            'Lower Arm Right': [] 
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


meta_dir = '/home/tengyu/Data/mscoco/v-coco/processed/None'
img_dir = '/home/tengyu/Data/mscoco/coco'
# mmdetection_path = ''
densepose_path = '/home/tengyu/Documents/densepose/DensePoseData/infer_out'
checkpoint_dir = '/home/tengyu/Documents/github/Part-GPNN/data/model_resnet_noisy/finetune_resnet'
vcoco_root = '/home/tengyu/Data/mscoco/v-coco/data'
save_data_path = '/home/tengyu/Documents/github/Part-GPNN/data/feature_resnet_tengyu'

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

part_eye = np.eye(14)
obj_eye = np.eye(81)

for imageset in ['train', 'test', 'val']:
    coco = vu.load_coco(os.path.join(vcoco_root))
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), os.path.join(vcoco_root))
    image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()

    for i_image, image_id in enumerate(image_ids):
        filename = coco.loadImgs(ids=[image_id])[0]['file_name']

        print('%d/%d: %s'%(i_image, len(image_ids), filename))

        if imageset != filename.split('_')[1]:
            continue

        image_meta = pickle.load(open(os.path.join(meta_dir, filename + '.p'), 'rb'), encoding='latin1')
        image = skimage.io.imread(os.path.join(img_dir, '%s2014'%imageset, filename))
        img_w = image.shape[0]
        img_h = image.shape[1]

        obj_boxes_all = instance['boxes'][instance['human_num']:]
        obj_classes_all = instance['classes'][instance['human_num']:]

        boxes, bodies = pickle.load(open(os.path.join(densepose_path, '%s/%s.pkl'%(vcoco_mapping[imageset], img_name)), 'rb'), encoding='latin-1')

        part_human_ids = []
        part_classes = []
        part_boxes = []

        human_boxes = []

        for human_id in range(len(boxes[1])):
            if boxes[1][human_id][4] < 0.7:
                continue

            # human_boxes contains parts at different levels        
            for part_id, part_name in enumerate(part_names):
                x, y = np.where(np.isin(bodies[1][human_id], part_ids[part_name]))
                x = x + boxes[1][human_id][1]
                y = y + boxes[1][human_id][0]
                if len(x) > 0:
                    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
                    part_boxes.append([y0,x0,y1,x1])
                    part_classes.append(part_id)
                    # plt.plot([y0,y0,y1,y1,y0], [x0,x1,x1,x0,x0])
                    part_human_ids.append(human_id)

                    if part_names[part_id] == 'Full Body':
                        human_boxes.append([y0,x0,y1,x1])

                    for obj_box in obj_boxes_all:
                        edge_box = combine_box(obj_box, part_boxes_all[-1,:])
                        edge_human_id.append(human_id)
                        edge_boxes_all = np.vstack([edge_boxes_all, [edge_box]])

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
                        obj_index = get_node_index(bbox, obj_boxes_all, np.arange(obj_num)) + human_num
                        if obj_index == human_num - 1:
                            warnings.warn('object detection missing')
                            continue
                        assert obj_index >= human_num and obj_index < human_num + obj_num

                        labels[(human_index, obj_index)].append(action_index - 1)
                        roles[(human_index, obj_index)].append(metadata.role_index[x['role_name'][i_role]] - 1)
                except IndexError:
                    warnings.warn('Labels missing for {}'.format(filename))
                    raise
                    pass

        edge_features = np.zeros((node_num, node_num, 1202))
        adj_mat = np.zeros((node_num, node_num))
        gt_strength_level = np.zeros([node_num, node_num])
        gt_action_labels = np.zeros([node_num, node_num, len(metadata.action_classes) - 1])
        gt_action_roles = np.zeros([node_num, node_num, len(metadata.roles) - 1])
        
        # Extract node features
        node_patches = []
        for i_node in range(node_num):
            if i_node < part_num:
                box = part_boxes[i_node]
            else:
                box = obj_boxes_all[i_node - part_num]
            img_patch = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
            img_patch = transform(cv2.resize(img_patch, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
            node_patches.append(img_patch)
        node_patches = torch.autograd.Variable(node_patches).cuda()
        feat, pred = feature_network(node_patches)
        node_features = feat.data.cpu().numpy()

        node_features_appd = np.zeros([node_features.shape[0], 6 + 14 + 81])
        node_features_appd[:part_num,0] = (part_boxes[:,2] - part_boxes[:,0]) / img_w # relative w
        node_features_appd[:part_num,1] = (part_boxes[:,3] - part_boxes[:,1]) / img_h # relative h
        node_features_appd[:part_num,2] = ((part_boxes[:,2] + part_boxes[:,0]) / 2) / img_w # relative cx
        node_features_appd[:part_num,3] = ((part_boxes[:,3] + part_boxes[:,1]) / 2) / img_h # relative cy
        node_features_appd[:part_num,4] = (part_boxes[:,2] - part_boxes[:,0]) * (part_boxes[:,3] - part_boxes[:,1]) / (img_w * img_h) # relative area
        node_features_appd[:part_num,5] = (part_boxes[:,2] - part_boxes[:,0]) / (part_boxes[:,3] - part_boxes[:,1]) # aspect ratio
        node_features_appd[:part_num,6:6+14] = part_eye[part_classes]
        
        node_features_appd[part_num:,0] = (obj_boxes_all[:,2] - obj_boxes_all[:,0]) / img_w # relative w
        node_features_appd[part_num:,1] = (obj_boxes_all[:,3] - obj_boxes_all[:,1]) / img_h # relative h
        node_features_appd[part_num:,2] = ((obj_boxes_all[:,2] + obj_boxes_all[:,0]) / 2) / img_w # relative cx
        node_features_appd[part_num:,3] = ((obj_boxes_all[:,3] + obj_boxes_all[:,1]) / 2) / img_h # relative cy
        node_features_appd[part_num:,4] = (obj_boxes_all[:,2] - obj_boxes_all[:,0]) * (obj_boxes_all[:,3] - obj_boxes_all[:,1]) / (img_w * img_h) # relative area
        node_features_appd[part_num:,5] = (obj_boxes_all[:,2] - obj_boxes_all[:,0]) / (obj_boxes_all[:,3] - obj_boxes_all[:,1]) # aspect ratio
        node_features_appd[part_num:,6+14:] = obj_eye[obj_classes_all]

        node_features_appd[np.isnan(node_features_appd)] = 0
        node_features_appd[np.isinf(node_features_appd)] = 0

        node_features = np.concatenate([node_features, node_features_appd], axis=-1)

        # Extract edge features
        edge_patch_mapping = {}
        edge_patch_feat = []
        for i_node in range(part_num):
            edge_patches = []
            # we only consider edges connecting at least one part. inter-object edges are not considered
            i_box = part_boxes[i_node]
            for j_node in range(i_node + 1, node_num):
                if (j_node < part_num and \
                part_human_ids[i_node] == part_human_ids[j_node] and 
                part_names[part_classes[j_node]] in part_graph[part_names[part_classes[i_node]]]) or \
                j_node >= part_num:
                    j_box = part_boxes[j_node]
                    box = combine_box(i_box, j_box)
                    img_patch = image[box[1] : box[3] + 1, box[0] : box[2] + 1, :]
                    img_patch = transform(cv2.resize(img_patch, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
                    edge_patch_mapping[(i_node, j_node)] = len(edge_patches) + sum(len(x) for x in edge_patch_feat)
                    edge_patches.append(img_patch)

            edge_patches = torch.autograd.Variable(edge_patches)
            feat, pred = feature_network(edge_patches)
            edge_patch_feat.append(feat.data.cpu().numpy())
        
        edge_patch_feat = np.vstack(edge_patch_feat)

        # Organize edge features
        for i_node in range(node_num):
            for j_node in range(node_num):
                if i_node == j_node:
                    edge_features[i_node, j_node] = node_features[i_node]
                    edge_features[i_node, j_node, 1000:1101] = node_features_appd[i_node]
                    edge_features[i_node, j_node, 1101:] = node_features_appd[j_node]
                    adj_mat[i_node, j_node] = 1
                else:
                    key = (min(i_node, j_node), max(i_node, j_node))
                    if key in edge_patch_mapping:
                        edge_features[i_node, j_node, :1000] = edge_patch_feat[edge_patch_mapping[key]]
                        edge_features[i_node, j_node, 1000:1101] = node_features_appd[i_node]
                        edge_features[i_node, j_node, 1101:] = node_features_appd[j_node]
                        adj_mat[i_node, j_node] = 1
                        # Compute GT Labels and GT signal strength on each edge
                        if i_node < part_num and j_node >= part_num:
                            gt_strength_level[i_node, j_node] = part_weights[part_names[part_classes[i_node]]]
                            for label in labels[(part_human_ids[i_node], j_node - part_num)]:
                                gt_action_labels[i_node, j_node, label] = 1
                            for role in roles[(part_human_ids[i_node], j_node - part_num)]:
                                gt_action_roles[i_node, j_node, role] = 1

                        if j_node < part_num and i_node >= part_num:
                            gt_strength_level[i_node, j_node] = part_weights[part_names[part_classes[j_node]]]
                            for label in labels[(part_human_ids[j_node], i_node - part_num)]:
                                gt_action_labels[i_node, j_node, label] = 1
                            for role in roles[(part_human_ids[j_node], i_node - part_num)]:
                                gt_action_roles[i_node, j_node, role] = 1

        data = {
            'node_features'  : node_features,
            'edge_features'  : edge_features,
            'adj_mat'        : adj_mat, 
            'action_labels'  : gt_action_labels,
            'action_roles'   : gt_action_roles,
            'strength_level' : gt_strength_level
        }
        pickle.dump(data, open(os.path.join(save_data_path, filename + '.data'), 'wb'))
        