"""
Created on Feb 24, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import random
import pickle
import argparse
import warnings

import cv2
import torch.utils.data
import numpy as np
import vsrl_utils as vu
import scipy.misc

import vcoco_config
import feature_model
import metadata

import torchvision
import torchvision.transforms

import matplotlib.pyplot as plt

def get_model():
    feature_network = feature_model.Resnet152(num_classes=len(metadata.action_classes))
    feature_network.cuda()
    # checkpoint_dir = os.path.join(os.path.dirname(__file__), '../../tmp', 'checkpoints', 'vcoco', 'finetune_resnet_noisy'.format(feature_mode))
    # best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
    best_model_file = os.path.join(os.path.dirname(__file__), '../../../../data/model_resnet_noisy/finetune_resnet_noisy/model_best.pth')
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

    return feature_network, transform

def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def img_to_torch(img, t):
    """
    input: H x W x C img iterables with range 0-255
    output: C x H x W img tensor with range 0-1, normalized
    """

    # img = np.array(img) / 255.
    # img = (img - mean) / std
    # if len(img.shape) == 3:
    #     img = img.transpose([2,0,1])
    # elif len(img.shape) == 4:
    #     img = img.transpose([0,3,1,2])
    # elif len(img.shape) == 5:
    #     img = img.transpose([0,1,4,2,3])
    # img = torch.autograd.Variable(torch.Tensor(img)).cuda()
    # return img

    img = np.array(img)
    if len(img.shape) == 4:
        new_img = []
        for _i in img:
            new_img.append(t(_i))
        return torch.autograd.Variable(torch.stack(new_img, dim=0)).cuda()
    if len(img.shape) == 5:
        new_img = []
        for _j in img:
            new_img.append([])
            for _i in _j:
                new_img[-1].append(t(_i))
            new_img[-1] = torch.stack(new_img[-1], dim=0)
        return torch.autograd.Variable(torch.stack(new_img, dim=0)).cuda()


class NoisyVCOCO(torch.utils.data.Dataset):
    def __init__(self, root, imageset, node_feature_appd=False):
        self.root = root
        self.coco = vu.load_coco(root)
        vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), root)
        self.image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
        self.unique_image_ids = list(set(self.image_ids))
        self.node_feature_appd = node_feature_appd

        self.model, self.transform = get_model()

    def __getitem__(self, index):
        input_h, input_w = 224, 224

        # for i in range(len(self.unique_image_ids)):
        #     img_id = self.coco.loadImgs(ids=[self.unique_image_ids[i]])[0]['file_name']
        #     if '000000165' in img_id or '000000368' in img_id or '000000436' in img_id or '000000531' in img_id:
        #         print(i, img_id)
        # exit()

        img_name = self.coco.loadImgs(ids=[self.unique_image_ids[index]])[0]['file_name']
        img_type = img_name.split('_')[1]
        try:
            img = scipy.misc.imread(os.path.join(os.path.dirname(__file__), '../../../../', '/data/COCO_train2014_000000000368.jpg'), mode='RGB')
            # data = pickle.load(open(os.path.join(self.root, '..', 'processed', 'resnet', '{}.p'.format(img_name)), 'rb'))
            # _edge_features = np.load(os.path.join(self.root, '..', 'processed', 'resnet', '{}_edge_features.npy').format(img_name))
            # _node_features = np.load(os.path.join(self.root, '..', 'processed', 'resnet', '{}_node_features.npy').format(img_name))
            data = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../../../', '/data/feature_resnet_noisy', '{}.p'.format(img_name)), 'rb'))
            _edge_features = np.load(os.path.join(os.path.dirname(__file__), '../../../../', '/data/feature_resnet_noisy', '{}_edge_features.npy').format(img_name))
            _node_features = np.load(os.path.join(os.path.dirname(__file__), '../../../../', '/data/feature_resnet_noisy', '{}_node_features.npy').format(img_name))
        except IOError:
            raise
            warnings.warn('data missing for {}'.format(img_name))
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

        part_images = []
        obj_images = []
        edge_images = []
        node_features = np.zeros([part_num + obj_num, 2000])
        edge_features = np.zeros([part_num + obj_num, part_num + obj_num, 1000])

        for part_box in part_boxes.astype(int):
            part_images.append(cv2.resize(img[part_box[1]:part_box[3] + 1, part_box[0]:part_box[2] + 1, :], (input_h, input_w), interpolation=cv2.INTER_LINEAR))
        for obj_box in obj_boxes.astype(int):
            obj_images.append(cv2.resize(img[obj_box[1]:obj_box[3] + 1, obj_box[0]:obj_box[2] + 1, :], (input_h, input_w), interpolation=cv2.INTER_LINEAR))
        for part_box in part_boxes:
            edge_images.append([])
            for obj_box in obj_boxes:
                edge_box = combine_box(obj_box, part_box).astype(int)
                edge_images[-1].append(cv2.resize(img[edge_box[1]:edge_box[3] + 1, edge_box[0]:edge_box[2] + 1, :], (input_h, input_w), interpolation=cv2.INTER_LINEAR))
        
        part_images = img_to_torch(part_images, self.transform)
        feat, pred = self.model(part_images)
        node_features[:part_num, :1000] = feat.detach().cpu().numpy()

        obj_images = img_to_torch(obj_images, self.transform)
        feat, pred = self.model(obj_images)
        node_features[part_num:, 1000:] = feat.detach().cpu().numpy()

        edge_images = img_to_torch(edge_images, self.transform)
        for i_part in range(len(edge_images)):
            feat, pred = self.model(edge_images[i_part])
            edge_features[i_part, part_num:, :] = feat.detach().cpu().numpy()
            edge_features[part_num:, i_part, :] = edge_features[i_part, part_num:, :]

        np.save(os.path.join(os.path.dirname(__file__), '../../../../', '/data/feature_resnet_noisy', '{}_node_features2.npy').format(img_name), node_features)
        np.save(os.path.join(os.path.dirname(__file__), '../../../../', '/data/feature_resnet_noisy', '{}_edge_features2.npy').format(img_name), edge_features)

        print(np.linalg.norm(node_features - _node_features) / np.linalg.norm(_node_features))
        print(np.linalg.norm(edge_features - _edge_features) / np.linalg.norm(_node_features))

        # # append bbox and class to node features
        # if self.node_feature_appd:
        #     node_features_appd = np.load(os.path.join(self.root, '..', 'processed', 'resnet', '{}_node_features_appd.npy').format(img_name))
        #     node_features = np.concatenate([node_features, node_features_appd], axis=-1)

        return edge_features, \
                    node_features, \
                    part_human_id, \
                    adj_mat, \
                    node_labels, \
                    node_roles, \
                    obj_boxes, \
                    part_boxes, \
                    human_boxes, \
                    img_id, \
                    img_name, \
                    human_num, \
                    part_num, \
                    obj_num, \
                    obj_classes, \
                    part_classes, \
                    part_adj_mat, \
                    img_name

    def __len__(self):
        return len(self.unique_image_ids)


def main(args):
    start_time = time.time()

    subset = ['train', 'val', 'test']
    training_set = NoisyVCOCO(args.data_root, subset[0])
    print('{} instances.'.format(len(training_set)))
    edge_features, obj_features, part_features, part_human_id, adj_mat, obj_labels, obj_roles, human_labels, human_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, obj_num, obj_classes, part_classes = training_set[113]
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
