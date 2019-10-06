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

from . import vcoco_config
from . import metadata
from . import roi_pooling

import torchvision
import torchvision.transforms

import matplotlib.pyplot as plt
from skimage import io, transform, color

class Vgg16(torch.nn.Module):
    def __init__(self, last_layer=0, requires_grad=False):
        super(Vgg16, self).__init__()
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential()
        for x in range(len(pretrained_vgg.features)):
            self.features.add_module(str(x), pretrained_vgg.features[x])

        # if feature_mode == 'roi_vgg':
        #     self.classifier = torch.nn.Sequential()
        #     self.classifier.add_module(str(0), pretrained_vgg.classifier[0])

        # self.classifier.add_module(str(1), pretrained_vgg.classifier[1])
        # for x in range(len(pretrained_vgg.classifier)-last_layer):
        #     print pretrained_vgg.classifier[x]
        #     self.classifier.add_module(str(x), pretrained_vgg.classifier[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)

        # if feature_mode == 'roi_vgg':
        #     x = x.view(x.size(0), -1)
        #     x = self.classifier(x)
        return x


def get_model():
    feature_network = Vgg16(last_layer=1)
    feature_network.cuda()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    return feature_network, transform

def combine_box(box1, box2):
    return np.hstack((np.maximum(box1[:2], box2[:2]), np.minimum(box1[2:], box2[2:])))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def img_to_torch(img, t):
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

def transform_bbox(boxes, mat):
    bboxes2 = np.zeros([boxes.shape[0], 4, 2])
    bboxes2[:,:2,0] = boxes[:,[2]]
    bboxes2[:,2:,0] = boxes[:,[0]]
    bboxes2[:,::2,1] = boxes[:,[3]]
    bboxes2[:,1::2,1] = boxes[:,[1]]
    bboxes_transformed = np.zeros(bboxes2.shape)
    bboxes_transformed[:,:,0] = bboxes2[:,:,0] * mat[0,0] + bboxes2[:,:,1] * mat[0,1] + mat[0,2] # X = a0*x + a1*y + a2
    bboxes_transformed[:,:,1] = bboxes2[:,:,0] * mat[1,0] + bboxes2[:,:,1] * mat[1,1] + mat[1,2] # Y = b0*x + b1*y + b2
    boxes = np.zeros([boxes.shape[0], 4])
    boxes[:,2] = np.min(bboxes_transformed[:,:,0], axis=1)
    boxes[:,0] = np.max(bboxes_transformed[:,:,0], axis=1)
    boxes[:,3] = np.min(bboxes_transformed[:,:,1], axis=1)
    boxes[:,1] = np.max(bboxes_transformed[:,:,1], axis=1)
    return boxes

class NoisyVCOCO(torch.utils.data.Dataset):
    def __init__(self, root, imageset, node_feature_appd=False, chance=0.0):
        self.root = root
        self.coco = vu.load_coco(root)
        vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), root)
        self.image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
        self.unique_image_ids = list(set(self.image_ids))
        self.node_feature_appd = node_feature_appd
        self.chance = chance

        self.model, self.transform = get_model()

    def __getitem__(self, index):
        input_h, input_w = 224, 224
        feature_length = 4096
        feature_size = [3, 3]
        adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

        img_name = self.coco.loadImgs(ids=[self.unique_image_ids[index]])[0]['file_name']
        img_type = img_name.split('_')[1]
        try:
            # img_path = os.path.join(self.root, '../coco/coco/images', '{}2014'.format(img_type), img_name)
            # data = pickle.load(open(os.path.join(self.root, '..', 'processed', 'resnet', '{}.p'.format(img_name)), 'rb'))
            img_path = os.path.join('/media/tengyu/data/mscoco/%s/%s'%(img_type, img_name))
            data = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../../../data/feature_resnet_noisy', '{}.p'.format(img_name)), 'rb'))
            img = io.imread(img_path)
            if len(img.shape) == 2:
                img = np.tile(np.expand_dims(img, axis=-1), [1,1,3])
        except IOError:
            warnings.warn('data missing for {}'.format(img_name))
            return self.__getitem__(3)
        except ValueError:
            warnings.warn('img grayscale for {}'.format(img_name))
            return self.__getitem__(3)

        h, w, _ = img.shape
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

        # apply transformation
        if random.random() < self.chance:
            shear = random.random() * 30 - 15        # -15 deg to +15 deg
            shear = np.deg2rad(shear)
        else:
            shear = 0
        if random.random() < self.chance:
            rotate = random.random() * 30 - 15       # -15 deg to +15 deg
            rotate = np.deg2rad(rotate)
        else:
            rotate = 0
        if random.random() < self.chance:
            scale = [random.random() * 0.3 + 0.85, random.random() * 0.3 + 0.85]     # 0.85 to 1.15
        else:
            scale = [1.0, 1.0]

        translation = [img.shape[0] * 0.5, img.shape[1] * 0.5]

        if random.random() < self.chance:
            # flip
            img = img[:,::-1,:]
            for b in part_boxes:
                b[0] = w - b[0]
                b[2] = w - b[2]
            for b in obj_boxes:
                b[0] = w - b[0]
                b[2] = w - b[2]

        img = color.rgb2hsv(img)
        if random.random() < self.chance:
            # brightness
            img[:,:,2] += random.random() * 0.3 - 0.15  # -0.15 to + 0.15
            img = np.clip(img, 0, 1)
        if random.random() < self.chance:
            # hue
            img[:,:,0] += random.random() * 0.5 - 0.25  # -0.15 to + 0.15
            img = np.clip(img, 0, 1)
        img = color.hsv2rgb(img)

        transformation = transform.AffineTransform(shear=shear, scale=scale, rotation=rotate, translation=translation)
        mat = transformation.params

        img = (img * 255).astype(np.uint8)
        img = transform.warp(img, transformation.inverse, order=0, output_shape=(img.shape[0] * 2, img.shape[1] * 2))

        part_boxes = transform_bbox(part_boxes, mat)
        obj_boxes = transform_bbox(obj_boxes, mat)

        img_feature = self.model.features(img_to_torch(cv2.resize(img, (input_h, input_w), interpolation=cv2.INTER_LINEAR), self.transform))
        img_feature = torch.autograd.Variable(img_feature)
        scale_h = (feature_size[0] - 1)/float(img.shape[0])
        scale_w = (feature_size[1] - 1)/float(img.shape[1])

        obj_features = np.empty((obj_num, 512 * feature_size[0] * feature_size[1]))
        objs = np.copy(obj_boxes)
        objs[:, 0] *= scale_w
        objs[:, 2] *= scale_w
        objs[:, 1] *= scale_h
        objs[:, 3] *= scale_h
        
        for i_box in range(objs.shape[0]):
            obj = objs[i_box, :].astype(int)
            if (obj[1] - obj[3]) < 1 or (obj[0] - obj[2]) < 1:
                continue
            try:
                obj_feature = adaptive_max_pool(img_feature[..., obj[3]:(obj[1]), obj[2]:(obj[0])])
                obj_features[i_box, :] = obj_feature.data.cpu().numpy().flatten()
            except:
                print(obj)
                raise

        part_features = np.empty((part_num, 512 * feature_size[0] * feature_size[1]))
        parts = np.copy(part_boxes)
        parts[:, 0] *= scale_w
        parts[:, 2] *= scale_w
        parts[:, 1] *= scale_h
        parts[:, 3] *= scale_h

        for i_box in range(parts.shape[0]):
            part = parts[i_box, :].astype(int)
            if (part[1] - part[3]) < 1 or (part[0] - part[2]) < 1:
                continue
            try:
                part_feature = adaptive_max_pool(img_feature[..., part[3]:(part[1]), part[2]:(part[0])])
                part_features[i_box, :] = part_feature.data.cpu().numpy().flatten()
            except:
                print(part)
                raise
        
        edge_features = np.empty((part_num + obj_num, part_num + obj_num, 512 * feature_size[0] * feature_size[1]))

        for i_part, part_box in enumerate(part_boxes):
            edge_images.append([])
            for i_obj, obj_box in enumerate(obj_boxes):
                edge_box = combine_box(obj_box, part_box).astype(int)
                edge_box[0] *= scale_w
                edge_box[2] *= scale_w
                edge_box[1] *= scale_h
                edge_box[3] *= scale_h

                edge_box = edge_box.astype(int)
                if (edge_box[1] - edge_box[3]) < 1 or (edge_box[0] - edge_box[2]) < 1:
                    continue
                try:
                    edge_feature = adaptive_max_pool(img_feature[..., edge_box[3]:(edge_box[1]), edge_box[2]:(edge_box[0])])
                    edge_features[i_part, part_num + i_obj, :] = edge_feature.data.cpu().numpy().flatten()
                    edge_features[part_num + i_obj, i_part, :] = edge_feature.data.cpu().numpy().flatten()
                except:
                    print(edge_box)
                    raise

        node_features = np.concatenate([part_features, obj_features], axis=0)

        # append bbox and class to node features
        if self.node_feature_appd:
            part_eye = np.eye(14)
            obj_eye = np.eye(81)

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

            node_features = np.concatenate([node_features, node_features_appd], axis=-1)

            node_features[np.isnan(node_features)] = 0
            node_features[np.isinf(node_features)] = 0
            edge_features[np.isnan(edge_features)] = 0
            edge_features[np.isinf(edge_features)] = 0

        np.save('a.npy', node_features)
        np.save('b.npy', edge_features)
        exit()

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
    subset = ['train', 'val', 'test']
    training_set = NoisyVCOCO(args.data_root, subset[0])
    print('{} instances.'.format(len(training_set)))
    start_time = time.time()
    edge_features, obj_features, part_features, part_human_id, adj_mat, obj_labels, obj_roles, human_labels, human_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, obj_num, obj_classes, part_classes = training_set[113]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    edge_features, obj_features, part_features, part_human_id, adj_mat, obj_labels, obj_roles, human_labels, human_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, obj_num, obj_classes, part_classes = training_set[113]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    edge_features, obj_features, part_features, part_human_id, adj_mat, obj_labels, obj_roles, human_labels, human_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, obj_num, obj_classes, part_classes = training_set[113]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    edge_features, obj_features, part_features, part_human_id, adj_mat, obj_labels, obj_roles, human_labels, human_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, obj_num, obj_classes, part_classes = training_set[113]
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
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
