"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle
import argparse
import warnings

import torch.utils.data
import numpy as np
import vsrl_utils as vu

from . import vcoco_config


class VCOCO(torch.utils.data.Dataset):
    def __init__(self, root, imageset, node_feature_appd=False):
        self.root = root
        self.coco = vu.load_coco(root)
        vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), root)
        self.image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
        self.unique_image_ids = list(set(self.image_ids))
        self.node_feature_appd = node_feature_appd

        # action_role = dict()
        # for i, x in enumerate(vcoco_all):
        #     action_role[x['action_name']] = x['role_name']
        # print 'action_role', action_role

    def __getitem__(self, index):
        img_name = self.coco.loadImgs(ids=[self.unique_image_ids[index]])[0]['file_name']
        try:
            data = pickle.load(open(os.path.join(self.root, '..', 'processed', 'resnet', '{}.p'.format(img_name)), 'rb'))
            edge_features = np.load(os.path.join(self.root, '..', 'processed', 'resnet', '{}_edge_features.npy').format(img_name))
            node_features = np.load(os.path.join(self.root, '..', 'processed', 'resnet', '{}_node_features.npy').format(img_name))
    
            # append bbox and class to node features
            if self.node_feature_appd:
                node_features_appd = np.load(os.path.join(self.root, '..', 'processed', 'resnet', '{}_node_features_appd.npy').format(img_name))
                node_features = np.concatenate([node_features, node_features_appd], axis=-1)
        except IOError:
            # warnings.warn('data missing for {}'.format(img_name))
            return self.__getitem__(3)

        img_id = data['img_id']
        adj_mat = data['adj_mat']
        node_labels = data['node_labels']
        node_roles = data['node_roles']
        obj_boxes = data['obj_boxes']
        part_boxes = data['part_boxes']
        human_boxes = data['human_boxes']
        human_num = data['human_num']
        part_num = data['part_num']
        obj_num = data['obj_num']
        obj_classes = data['obj_classes']
        part_classes = data['part_classes']
        part_human_id = data['part_human_id']
        part_adj_mat = None # data['part_adj_mat']

        if part_num + obj_num != len(edge_features):
            print(img_name)
            exit()

        return edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, part_num, obj_num, obj_classes, part_classes, part_adj_mat, img_name

    def __len__(self):
        return len(self.unique_image_ids)


def main(args):
    start_time = time.time()

    subset = ['train', 'val', 'test']
    training_set = VCOCO(args.data_root, subset[0])
    print('{} instances.'.format(len(training_set)))
    edge_features, obj_features, part_features, part_human_id, adj_mat, obj_labels, obj_roles, human_labels, human_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, obj_num, obj_classes, part_classes = training_set[0]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def parse_arguments():
    paths = vcoco_config.Paths()
    parser = argparse.ArgumentParser(description='V-COCO dataset')
    parser.add_argument('--data-root', default=paths.data_root, help='dataset path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
