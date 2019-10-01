"""
Created on Mar 13, 2017

@author: Siyuan Qi

Description of the file.

"""

from __future__ import print_function
import os
import time
import pickle
import warnings

import numpy as np
import scipy.io

import hico_config
import metadata


def get_intersection(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))

def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))

def compute_area(box):
    return (box[2]-box[0])*(box[3]-box[1])

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

def get_obj_index(classname, bbox, det_classes, det_boxes, node_num):
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    max_iou_index = -1

    for i_node in range(node_num):
        # print(classname, metadata.hico_classes[metadata.coco_to_hico[det_classes[i_node]]])
        if classname == metadata.hico_classes[metadata.coco_to_hico[det_classes[i_node]]]:
            # check bbox overlap
            intersection_area = compute_area(get_intersection(bbox, det_boxes[i_node]))
            iou = intersection_area/(compute_area(bbox)+compute_area(det_boxes[i_node])-intersection_area)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = i_node
    return max_iou_index

def get_human_index(bbox, det_boxes, node_num):
    bbox = np.array(bbox, dtype=np.float32)
    max_iou = 0.5  # Use 0.5 as a threshold for evaluation
    max_iou_index = -1

    for i_node in range(node_num):
        # check bbox overlap
        intersection_area = compute_area(get_intersection(bbox, det_boxes[i_node]))
        iou = intersection_area/(compute_area(bbox)+compute_area(det_boxes[i_node])-intersection_area)
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i_node
    return max_iou_index

def read_features(data_root, tmp_root, bbox, list_action):
    # roi_size = 49  # Deformable ConvNet
    # roi_size = 512 * 49  # VGG conv feature
    roi_size = 200  # VGG fully connected feature
    # hoi_class_num = 600
    action_class_num = 117
    # feature_path = os.path.join(data_root, 'processed', 'features_background_49')
    save_data_path = os.path.join(data_root, 'processed', 'hico_data_background_49')
    feature_path = os.path.join(data_root, 'processed', 'features_roi_vgg')
    #save_data_path = os.path.join(data_root, 'processed', 'hico_data_roi_vgg')
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    for i_image in range(bbox['filename'].shape[1]):
        filename = os.path.splitext(bbox['filename'][0, i_image][0])[0]
        print(filename)

        try:
            obj_classes = np.load(os.path.join(feature_path, '{}.jpg_obj_classes.npy'.format(filename)))
            obj_boxes = np.load(os.path.join(feature_path, '{}.jpg_obj_boxes.npy'.format(filename)))
            obj_features = np.load(os.path.join(feature_path, '{}.jpg_obj_features.npy'.format(filename)))
            part_classes = np.load(os.path.join(feature_path, '{}.jpg_part_classes.npy'.format(filename)))
            part_boxes = np.load(os.path.join(feature_path, '{}.jpg_part_boxes.npy'.format(filename)))
            part_human_id = np.load(os.path.join(feature_path, '{}.jpg_part_human_id.npy'.format(filename)))
            part_features = np.load(os.path.join(feature_path, '{}.jpg_part_features.npy'.format(filename)))
            edge_human = np.load(os.path.join(feature_path, '{}.jpg_edge_human_id.npy'.format(filename)))
            edge_boxes = np.load(os.path.join(feature_path, '{}.jpg_edge_boxes.npy'.format(filename)))
            edge_features_in = np.load(os.path.join(feature_path, '{}.jpg_edge_features.npy'.format(filename)))
        except IOError:
            continue

        part_num = len(part_boxes)
        obj_num = len(obj_boxes)
        human_num = len(list(set(part_human_id)))
        edge_num = len(edge_boxes)
        # p_node_num = part_num + obj_num
        node_num = human_num + obj_num
        assert edge_num == part_num * obj_num

        if part_num == 0:
            warnings.warn('human detection missing for {}'.format(filename))
            continue

        human_boxes, part_human_ids, human_ids = group_boxes(part_boxes, part_human_id)

        edge_features = np.zeros((part_num+obj_num, part_num+obj_num, roi_size))
        node_features = np.zeros((part_num+obj_num, roi_size*2))
        adj_mat = np.zeros((human_num+obj_num, human_num+obj_num))
        node_labels = np.zeros((node_num, action_class_num))

        # Node features
        for i_node in range(node_num):
            # node_features[i_node, :] = np.reshape(det_features[i_node, ...], roi_size)
            if i_node < part_num:
                node_features[i_node, :roi_size] = np.reshape(part_features[i_node, ...], roi_size)
            else:
                node_features[i_node, roi_size:] = np.reshape(obj_features[i_node - part_num, ...], roi_size)

        # Edge features
        i_edge = 0
        for i_part in range(part_num):
            for i_obj in range(obj_num):
                edge_features[i_part, part_num + i_obj, :] = edge_features_in[i_edge, :]
                edge_features[part_num + i_obj, i_part, :] = edge_features_in[i_edge, :]
                i_edge += 1

        # Adjacency matrix and node labels
        for i_hoi in range(bbox['hoi'][0, i_image]['id'].shape[1]):
            try:
                classname = 'person'
                x1 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['x1'][0, 0][0, 0]
                y1 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['y1'][0, 0][0, 0]
                x2 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['x2'][0, 0][0, 0]
                y2 = bbox['hoi'][0, i_image]['bboxhuman'][0, i_hoi]['y2'][0, 0][0, 0]
                human_index = get_human_index([x1, y1, x2, y2], human_boxes, human_num)

                hoi_id = bbox['hoi'][0, i_image]['id'][0, i_hoi][0, 0]
                classname = list_action['nname'][hoi_id, 0][0]
                x1 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['x1'][0, 0][0, 0]
                y1 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['y1'][0, 0][0, 0]
                x2 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['x2'][0, 0][0, 0]
                y2 = bbox['hoi'][0, i_image]['bboxobject'][0, i_hoi]['y2'][0, 0][0, 0]
                obj_index = get_obj_index(classname, [x1, y1, x2, y2], obj_classes, obj_boxes, obj_num)

                action_id = metadata.hoi_to_action[hoi_id]
                if human_index != -1 and obj_index != -1:
                    adj_mat[human_index, obj_index] = 1
                    adj_mat[obj_index, human_index] = 1
                    node_labels[human_index, action_id] = 1
                    node_labels[obj_index, action_id] = 1
            except IndexError:
                pass

        instance = dict()
        instance['human_num'] = human_num
        instance['part_num'] = part_num
        instance['obj_num'] = obj_num
        instance['img_name'] = filename
        instance['human_boxes'] = human_boxes
        instance['part_boxes'] = part_boxes
        instance['obj_boxes'] = obj_boxes
        instance['obj_classes'] = obj_classes
        instance['part_classes'] = part_classes
        instance['adj_mat'] = adj_mat
        instance['part_human_id'] = part_human_ids
        instance['node_labels'] = node_labels
        np.save(os.path.join(save_data_path, '{}_edge_features'.format(filename)), edge_features)
        np.save(os.path.join(save_data_path, '{}_node_features'.format(filename)), node_features)
        pickle.dump(instance, open(os.path.join(save_data_path, '{}.p'.format(filename)), 'wb'))


def collect_data(paths):
    anno_bbox = scipy.io.loadmat(os.path.join(paths.data_root, 'anno_bbox.mat'))
    bbox_train = anno_bbox['bbox_train']
    bbox_test = anno_bbox['bbox_test']
    list_action = anno_bbox['list_action']

    read_features(paths.data_root, paths.tmp_root, bbox_train, list_action)
    read_features(paths.data_root, paths.tmp_root, bbox_test, list_action)


def main():
    paths = hico_config.Paths()
    start_time = time.time()
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
