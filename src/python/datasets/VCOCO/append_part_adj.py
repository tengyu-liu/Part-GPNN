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


def parse_features(paths, imageset):
    # roi_size = 49  # Deformable ConvNet
    # roi_size = 512 * 49  # VGG conv feature
    # roi_size = 4096  # VGG fully connected feature
    roi_size = 1000  # ResNet fully connected feature
    feature_size = 1000
    feature_type = 'resnet_noisy'
    action_class_num = len(metadata.action_classes)
    no_action_index = metadata.action_index['none']
    no_role_index = metadata.role_index['none']
    feature_path = os.path.join(paths.data_root, 'features_{}'.format(feature_type))
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

    for i_image, image_id in enumerate(image_ids):
        filename = coco.loadImgs(ids=[image_id])[0]['file_name']
        print(filename)
        try:
            part_classes = np.load(os.path.join(feature_path, '{}_part_classes.npy'.format(filename)))
            part_human_id = np.load(os.path.join(feature_path, '{}_part_human_id.npy'.format(filename)))
            
        except IOError:
            warnings.warn('Features and detection results missing for {}'.format(filename))
            continue

        # 1. compute number of human, objects, and edges
        part_num = len(part_classes)

        # 2. Create placeholders for edge_feature, node_feature, adj_mat, node_labels, node_roles
        unique_image_ids.append(image_id)
        part_adj_mat = np.zeros((part_num, part_num))

        # Create part-level adj mats
        for i_part in range(part_num):
            for j_part in range(part_num):
                if part_human_id[i_part] == part_human_id[j_part]:
                    if part_dist[part_classes[i_part], part_classes[j_part]] == 1:
                        part_adj_mat[i_part, j_part] = 1

        try:
            instance = pickle.load(open(os.path.join(save_data_path, '{}.p'.format(filename)), 'rb'))
            instance['part_adj_mat'] = part_adj_mat
            pickle.dump(instance, open(os.path.join(save_data_path, '{}.p'.format(filename)), 'wb'))
        except IOError:
            warnings.warn('.p file missing for {}'.format(filename))
            continue

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
