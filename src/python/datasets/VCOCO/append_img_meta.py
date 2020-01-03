"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

"""

# from __future__ import print_function
import os
import time
import pickle
import warnings

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import cv2

import vcoco_config
import vsrl_eval
import vsrl_utils as vu
import metadata


part_names = ['Right Shoulder',
            'Left Shoulder',
            'Knee Right',
            'Knee Left',
            'Ankle Right',
            'Ankle Left',
            'Elbow Left',
            'Elbow Right',
            'Hand Left',
            'Hand Right',
            'Head',
            'Hip']

part_dist = []
part_dist.append([0, 2, 2, 2, 3, 3, 3, 1, 4, 2, 1, 1])
part_dist.append([2, 0, 2, 2, 3, 3, 1, 3, 2, 4, 1, 1])
part_dist.append([2, 2, 0, 2, 1, 3, 3, 3, 4, 4, 3, 1])
part_dist.append([2, 2, 2, 0, 3, 1, 3, 3, 4, 4, 3, 1])
part_dist.append([3, 3, 1, 3, 0, 4, 4, 4, 5, 5, 4, 2])
part_dist.append([3, 3, 3, 1, 4, 0, 4, 4, 5, 5, 4, 2])
part_dist.append([3, 1, 3, 3, 4, 4, 0, 4, 1, 5, 2, 2])
part_dist.append([1, 3, 3, 3, 4, 4, 4, 0, 5, 1, 2, 2])
part_dist.append([4, 2, 4, 4, 5, 5, 1, 5, 0, 6, 3, 3])
part_dist.append([2, 4, 4, 4, 5, 5, 5, 1, 6, 0, 3, 3])
part_dist.append([1, 1, 3, 3, 4, 4, 2, 2, 3, 3, 0, 2])
part_dist.append([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 0])
part_dist = np.array(part_dist)

def parse_classes(det_classes):
    obj_nodes = False
    human_num = 0
    obj_num = 0
    for i in range(det_classes.shape[0]):
        if not obj_nodes:
            if det_classes[i] == 1:
                human_num += 1
            else:
                obj_nodes = True
                obj_num += 1
        else:
            if det_classes[i] > 1 or det_classes[i] == -1:
                obj_num += 1
            else:
                break

    node_num = human_num + obj_num
    edge_num = det_classes.shape[0] - node_num
    return human_num, obj_num, edge_num


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


def group_boxes(part_boxes, part_human_id):
    last_human_id = None
    human_boxes = []
    new_human_id = []
    new_part_human_id = []
    human_id = -1
    for i_part, part_box in enumerate(part_boxes):
        if last_human_id is None:
            human_id += 1
            human_box = part_box
        elif last_human_id != part_human_id[i_part]:
            human_boxes.append(human_box)
            new_human_id.append(human_id)
            human_id += 1
            human_box = part_box
        else:
            human_box = combine_box(human_box, part_box)
        last_human_id = part_human_id[i_part]
        new_part_human_id.append(human_id)
    human_boxes.append(human_box)
    new_human_id.append(human_id)
    return human_boxes, new_part_human_id, new_human_id

vcoco_mapping = {'train': 'train', 'test': 'val', 'val': 'train'}

def parse_features(paths, imageset):
    # roi_size = 49  # Deformable ConvNet
    # roi_size = 512 * 49  # VGG conv feature
    # roi_size = 4096  # VGG fully connected feature
    roi_size = 1000  # ResNet fully connected feature
    feature_size = 1000
    feature_type = 'resnet'
    action_class_num = len(metadata.action_classes)
    no_action_index = metadata.action_index['none']
    no_role_index = metadata.role_index['none']
    feature_path = os.path.join(paths.data_root, 'features_{}_noisy'.format(feature_type))
    save_data_path = os.path.join(paths.data_root, 'processed', feature_type)
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    coco = vu.load_coco(os.path.join(paths.vcoco_data_root, 'data'))
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), os.path.join(paths.vcoco_data_root, 'data'))
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)

    image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
    all_results = list()
    unique_image_ids = list()

    part_eye = np.eye(14)
    obj_eye = np.eye(81)

    for i_image, image_id in enumerate(image_ids):
        filename = coco.loadImgs(ids=[image_id])[0]['file_name']
        # print(os.path.join(paths.data_root, 'coco', vcoco_mapping[imageset]+'2014', filename))
        img = cv2.imread(os.path.join(paths.data_root, 'coco', vcoco_mapping[imageset]+'2014', filename))

        img_w = img.shape[0]
        img_h = img.shape[1]
        try:
            data = pickle.load(open(os.path.join(paths.data_root, 'processed', feature_type, '{}.p'.format(filename)), 'rb'))
            # edge_features = np.load(os.path.join(paths.data_root, 'processed', feature_type, '{}_edge_features.npy').format(filename))
            node_features = np.load(os.path.join(paths.data_root, 'processed', feature_type, '{}_node_features.npy').format(filename))
        except:
            continue

        obj_boxes = data['obj_boxes']
        part_boxes = data['part_boxes']
        part_num = data['part_num']
        obj_num = data['obj_num']
        obj_classes = data['obj_classes']
        part_classes = data['part_classes']

        # append bbox and class to node features
        # assert img_w > 0
        # assert img_h > 0
        # assert np.all((part_boxes[:,3] - part_boxes[:,1]) > 0)
        # assert np.all((obj_boxes[:,3] - obj_boxes[:,1]) > 0)

        node_features_appd = np.zeros([node_features.shape[0], 6 + 14 + 81])

        node_features_appd[:part_num,0] = (part_boxes[:,2] - part_boxes[:,0]) / img_w # relative w
        node_features_appd[:part_num,1] = (part_boxes[:,3] - part_boxes[:,1]) / img_h # relative h
        node_features_appd[:part_num,2] = ((part_boxes[:,2] + part_boxes[:,0]) / 2) / img_w # relative cx
        node_features_appd[:part_num,3] = ((part_boxes[:,3] + part_boxes[:,1]) / 2) / img_h # relative cy
        node_features_appd[:part_num,4] = (part_boxes[:,2] - part_boxes[:,0]) * (part_boxes[:,3] - part_boxes[:,1]) / (img_w * img_h) # relative area
        node_features_appd[:part_num,5] = (part_boxes[:,2] - part_boxes[:,0]) / (part_boxes[:,3] - part_boxes[:,1]) # aspect ratio
        node_features_appd[:part_num,6:6+14] = part_eye[part_classes]

        node_features_appd[part_num:,0] = (obj_boxes[:,2] - obj_boxes[:,0]) / img_w # relative w
        node_features_appd[part_num:,1] = (obj_boxes[:,3] - obj_boxes[:,1]) / img_h # relative h
        node_features_appd[part_num:,2] = ((obj_boxes[:,2] + obj_boxes[:,0]) / 2) / img_w # relative cx
        node_features_appd[part_num:,3] = ((obj_boxes[:,3] + obj_boxes[:,1]) / 2) / img_h # relative cy
        node_features_appd[part_num:,4] = (obj_boxes[:,2] - obj_boxes[:,0]) * (obj_boxes[:,3] - obj_boxes[:,1]) / (img_w * img_h) # relative area
        node_features_appd[part_num:,5] = (obj_boxes[:,2] - obj_boxes[:,0]) / (obj_boxes[:,3] - obj_boxes[:,1]) # aspect ratio
        node_features_appd[part_num:,6+14:] = obj_eye[obj_classes]

        node_features_appd[np.isnan(node_features_appd)] = 0
        node_features_appd[np.isinf(node_features_appd)] = 0

        np.save(os.path.join(paths.data_root, 'processed', feature_type, '{}_node_features_appd.npy').format(filename), node_features_appd)

        # try:
        #     instance = pickle.load(open(os.path.join(save_data_path, '{}.p'.format(filename)), 'rb'))
        #     instance['part_adj_mat'] = part_adj_mat
        #     pickle.dump(instance, open(os.path.join(save_data_path, '{}.p'.format(filename)), 'wb'))
        # except IOError:
        #     warnings.warn('.p file missing for {}'.format(filename))
        #     continue

def collect_data(paths):
    imagesets = ['train', 'val', 'test']
    for imageset in imagesets:
        parse_features(paths, imageset)
        # break


def main():
    start_time = time.time()
    paths = vcoco_config.Paths()
    paths.eval_root = 'evaluation/vcoco/features'
    if not os.path.exists(paths.eval_root):
        os.makedirs(paths.eval_root)
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
