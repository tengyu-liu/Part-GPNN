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


part_names = [
    'Torso',            # 0 
    'Right Hand',       # 1
    'Left Hand',        # 2
    'Left Foot',        # 3
    'Right Foot',       # 4
    'Upper Leg Right',  # 5
    'Upper Leg Left',   # 6
    'Lower Leg Right',  # 7
    'Lower Leg Left',   # 8
    'Upper Arm Left',   # 9
    'Upper Arm Right',  # 10
    'Lower Arm Left',   # 11
    'Lower Arm Right',  # 12
    'Head']             # 13

part_dist = []
part_dist.append([0, 3, 3, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2, 1])
part_dist.append([3, 0, 6, 6, 6, 4, 4, 5, 5, 4, 2, 5, 1, 4])
part_dist.append([3, 6, 0, 6, 6, 4, 4, 5, 5, 2, 4, 1, 5, 4])
part_dist.append([3, 6, 6, 0, 6, 4, 2, 5, 1, 4, 4, 5, 5, 4])
part_dist.append([3, 6, 6, 6, 0, 2, 4, 1, 5, 4, 4, 5, 5, 4])
part_dist.append([1, 4, 4, 4, 2, 0, 2, 1, 3, 2, 2, 3, 3, 2])
part_dist.append([1, 4, 4, 2, 4, 2, 0, 3, 1, 2, 2, 3, 3, 2])
part_dist.append([2, 5, 5, 5, 1, 1, 3, 0, 4, 3, 3, 4, 4, 3])
part_dist.append([2, 5, 5, 1, 5, 3, 1, 4, 0, 3, 3, 4, 4, 3])
part_dist.append([1, 4, 2, 4, 4, 2, 2, 3, 3, 0, 2, 1, 3, 2])
part_dist.append([1, 2, 4, 4, 4, 2, 2, 3, 3, 2, 0, 3, 1, 2])
part_dist.append([2, 5, 1, 5, 5, 3, 3, 4, 4, 1, 3, 0, 4, 3])
part_dist.append([2, 1, 5, 5, 5, 3, 3, 4, 4, 3, 1, 4, 0, 3])
part_dist.append([1, 4, 4, 4, 4, 2, 2, 3, 3, 2, 2, 3, 3, 0])
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
    feature_type = 'None'
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

    for i_image, image_id in enumerate(image_ids):
        filename = coco.loadImgs(ids=[image_id])[0]['file_name']
        print(filename)
        """
        For each image, 
        1. Compute number of human, objects, nodes, groups, edges
        3. Extract node features 
        4. Extract edge features
        
        5. Compute ground truth adj_mat, node_label, node_role
        """
        # if not os.path.exists(os.path.join(save_data_path, '{}.p'.format(filename))) or type(pickle.load(open(os.path.join(save_data_path, '{}.p'.format(filename)), 'rb'))) == dict:
        #     continue
        # else:
        #     os.remove(os.path.join(save_data_path, '{}.p'.format(filename)))
        if not os.path.exists(os.path.join(save_data_path, '{}.p'.format(filename))):
            try:
                obj_classes = np.load(os.path.join(feature_path, '{}_obj_classes.npy'.format(filename)))
                obj_boxes = np.load(os.path.join(feature_path, '{}_obj_boxes.npy'.format(filename)))
                part_classes = np.load(os.path.join(feature_path, '{}_part_classes.npy'.format(filename)))
                part_boxes = np.load(os.path.join(feature_path, '{}_part_boxes.npy'.format(filename)))
                part_human_id = np.load(os.path.join(feature_path, '{}_part_human_id.npy'.format(filename)))
                edge_human = np.load(os.path.join(feature_path, '{}_edge_human_id.npy'.format(filename)))
                edge_boxes = np.load(os.path.join(feature_path, '{}_edge_boxes.npy'.format(filename)))
                if feature_type != 'None':
                    obj_features = np.load(os.path.join(feature_path, '{}_obj_features.npy'.format(filename)))
                    part_features = np.load(os.path.join(feature_path, '{}_part_features.npy'.format(filename)))
                    edge_features_in = np.load(os.path.join(feature_path, '{}_edge_features.npy'.format(filename)))
                
            except IOError:
                warnings.warn('Features and detection results missing for {}'.format(filename))
                continue

            # 1. compute number of human, objects, and edges
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

            # 2. Create placeholders for edge_feature, node_feature, adj_mat, node_labels, node_roles
            unique_image_ids.append(image_id)
            if feature_type != 'None':
                edge_features = np.zeros((part_num + obj_num, part_num + obj_num , feature_size))
                node_features = np.zeros((part_num + obj_num, feature_size*2))
            adj_mat = np.zeros((human_num + obj_num, human_num + obj_num))
            node_labels = np.zeros((node_num, action_class_num))
            node_roles = np.zeros((node_num, 3))
            node_labels[:, no_action_index] = 1
            node_roles[:, no_role_index] = 1

            # Group human boxes
            human_boxes, part_human_ids, human_ids = group_boxes(part_boxes, part_human_id)
            det_boxes = np.vstack([human_boxes, obj_boxes])

            if feature_type != 'None':
                # 3. Extract node features  # part : obj
                for i_node in range(part_num + obj_num):
                    # node_features[i_node, :] = np.reshape(det_features[i_node, ...], roi_size)
                    if i_node < part_num:
                        node_features[i_node, :roi_size] = np.reshape(part_features[i_node, ...], roi_size)
                    else:
                        node_features[i_node, roi_size:] = np.reshape(obj_features[i_node - part_num, ...], roi_size)

                # 4. Extract edge features
                i_edge = 0
                for i_part in range(part_num):
                    for i_obj in range(obj_num):
                        edge_box = edge_boxes[i_edge]
                        part_box = part_boxes[i_part]
                        obj_box = obj_boxes[i_obj]
                        assert np.linalg.norm(combine_box(part_box, obj_box) - edge_box) == 0

                        edge_features[i_part, part_num + i_obj, :] = np.reshape(edge_features_in[i_edge, ...], roi_size)
                        edge_features[part_num + i_obj, i_part, :] = edge_features[i_part, part_num + i_obj, :]
                        i_edge += 1
        else:
            instance = pickle.load(open(os.path.join(save_data_path, '{}.p'.format(filename)), 'rb'))
            image_id = instance['img_id']
            human_num = instance['human_num']
            obj_num = instance['obj_num']
            part_num = instance['part_num']
            filename = instance['img_name']
            human_boxes = instance['human_boxes']
            part_boxes = instance['part_boxes']
            obj_boxes = instance['obj_boxes']
            obj_classes = instance['obj_classes']
            part_classes = instance['part_classes']
            adj_mat = instance['adj_mat']
            part_human_ids = instance['part_human_id']
            node_labels = instance['node_labels']
            node_roles = instance['node_roles']
            if feature_type != 'None':
                edge_features = np.load(os.path.join(save_data_path, '{}_edge_features.npy'.format(filename)))
                node_features = np.load(os.path.join(save_data_path, '{}_node_features.npy'.format(filename)))

        # 5. Compute ground truth adj_mat, node_label, node_role
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
                    node_labels[human_index, action_index] = 1
                    node_labels[human_index, no_action_index] = 0

                    for i_role in range(1, len(x['role_name'])):
                        bbox = role_bbox[i_role, :]
                        if np.isnan(bbox[0]):
                            continue
                        obj_index = get_node_index(bbox, obj_boxes, np.arange(obj_num)) + human_num
                        if obj_index == human_num - 1:
                            warnings.warn('object detection missing')
                            continue
                        assert obj_index >= human_num and obj_index < human_num + obj_num
                        node_labels[obj_index, action_index] = 1
                        node_labels[obj_index, no_action_index] = 0
                        node_roles[obj_index, metadata.role_index[x['role_name'][i_role]]] = 1
                        node_roles[obj_index, no_role_index] = 0
                        adj_mat[human_index, obj_index] = 1
                        adj_mat[obj_index, human_index] = 1
                except IndexError:
                    warnings.warn('Labels missing for {}'.format(filename))
                    raise
                    pass
        
        instance = dict()
        instance['img_id'] = image_id
        instance['human_num'] = human_num
        instance['obj_num'] = obj_num
        instance['part_num'] = part_num
        instance['img_name'] = filename
        instance['human_boxes'] = human_boxes
        instance['part_boxes'] = part_boxes
        instance['obj_boxes'] = obj_boxes
        instance['edge_boxes'] = edge_boxes
        instance['obj_classes'] = obj_classes
        instance['part_classes'] = part_classes
        instance['adj_mat'] = adj_mat
        instance['part_human_id'] = part_human_ids
        instance['node_labels'] = node_labels
        instance['node_roles'] = node_roles
        if feature_type != 'None':
            np.save(os.path.join(save_data_path, '{}_edge_features.npy'.format(filename)), edge_features)
            np.save(os.path.join(save_data_path, '{}_node_features.npy'.format(filename)), node_features)
        pickle.dump(instance, open(os.path.join(save_data_path, '{}.p'.format(filename)), 'wb'))

    print('total image', len(unique_image_ids), 'total results', len(all_results))

def visualize_roi(paths, imageset, filename, roi):
    image_path = os.path.join(paths.data_root, '../v-coco/coco/images', '{}2014'.format(imageset), filename)
    assert os.path.exists(image_path)
    original_img = scipy.misc.imread(image_path, mode='RGB')
    roi_image = original_img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
    plt.imshow(roi_image)
    plt.show()


def append_result(all_results, node_labels, node_roles, image_id, boxes, human_num, obj_num, adj_mat):
    for i in range(human_num):
        if node_labels[i, metadata.action_index['none']] > 0.5:
            continue
        instance_result = dict()
        instance_result['image_id'] = image_id
        instance_result['person_box'] = boxes[i, :]
        for action_index, action in enumerate(metadata.action_classes):
            if action == 'none' or node_labels[i, action_index] < 0.5:
                continue
            result = instance_result.copy()
            result['{}_agent'.format(action)] = node_labels[i, action_index]
            for role in metadata.action_roles[action][1:]:
                role_index = metadata.role_index[role]
                action_role_key = '{}_{}'.format(action, role)
                best_score = -np.inf
                for j in range(human_num, human_num+obj_num):
                    if adj_mat[i, j] > 0.5:
                        action_role_score = (node_labels[j, action_index] + node_roles[j, role_index])/2  # TODO: how to evaluate action-role score
                        if action_role_score > best_score:
                            best_score = action_role_score
                            obj_info = np.append(boxes[j, :], action_role_score)
                if best_score > 0:
                    result[action_role_key] = obj_info
            all_results.append(result)


def get_vcocoeval(paths, imageset):
    return vsrl_eval.VCOCOeval(os.path.join(paths.data_root, '..', 'v-coco/data/vcoco/vcoco_{}.json'.format(imageset)),
                               os.path.join(paths.data_root, '..', 'v-coco/data/instances_vcoco_all_2014.json'),
                               os.path.join(paths.data_root, '..', 'v-coco/data/splits/vcoco_{}.ids'.format(imageset)))


def vcoco_evaluation(args, vcocoeval, imageset, all_results):
    det_file = os.path.join(args.eval_root, '{}_detections.pkl'.format(imageset))
    pickle.dump(all_results, open(det_file, 'wb'))
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)


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
