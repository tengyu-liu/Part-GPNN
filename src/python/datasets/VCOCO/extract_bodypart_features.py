"""
Created on Oct 12, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import json
import pickle
import warnings
import datetime

import numpy as np
import scipy.misc
import cv2
import torch
import torch.autograd
import torchvision.models

from pycocotools.coco import COCO
# import vsrl_utils as vu
import vcoco_config
import roi_pooling
import feature_model
import metadata

import matplotlib.pyplot as plt

feature_mode = 'None'

part_ids = {'Right Shoulder': 2,
            'Left Shoulder': 5,
            'Knee Right': 10,
            'Knee Left': 13,
            'Ankle Right': 11,
            'Ankle Left': 14,
            'Elbow Left': 6,
            'Elbow Right': 3,
            'Hand Left': 7,
            'Hand Right': 4,
            'Head': 0,
            'Hip': 8,
            }

part_names = list(part_ids.keys()) 

vcoco_mapping = {'train': 'train', 'test': 'val', 'val': 'train'}

class Vgg16(torch.nn.Module):
    def __init__(self, last_layer=0, requires_grad=False):
        super(Vgg16, self).__init__()
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential()
        for x in range(len(pretrained_vgg.features)):
            self.features.add_module(str(x), pretrained_vgg.features[x])

        # if feature_mode == 'roi_vgg':
        #     self.classifier = torch.nn.Sequential()
        #     self.classifier.add_modul 
        # e(str(0), pretrained_vgg.classifier[0])

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


def get_model(paths):
    # vgg16 = Vgg16(last_layer=1).cuda()
    
    if feature_mode == 'None':
        return 

    feature_network = feature_model.Resnet152(num_classes=len(metadata.action_classes))
    feature_network.cuda()
    checkpoint_dir = os.path.join(paths.tmp_root, 'checkpoints', 'vcoco', 'finetune_{}_noisy'.format(feature_mode))
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

    return feature_network, transform

def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def get_info(paths, imageset):
    vcoco_feature_path = paths.data_root
    vcoco_path = os.path.join(vcoco_feature_path, '../v-coco')

    prefix = 'instances' if 'test' not in vcoco_mapping[imageset] else 'image_info'
    coco = COCO(os.path.join(vcoco_path, 'coco', 'annotations', prefix + '_' + vcoco_mapping[imageset] + '2014.json'))
    coco_ids = coco.getImgIds()

    image_ids = [int(x.strip()) for x in open('/home/tengyu/Data/mscoco/v-coco/data/splits/vcoco_%s.ids'%imageset).readlines()]

    feature_path = os.path.join(vcoco_feature_path, 'features_{}_noisy'.format(feature_mode))

    classes = ['__background__',  # always index 0
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

    return vcoco_path, feature_path, classes, image_ids, coco_ids


def extract_features(paths, imageset):
    input_h, input_w = 244, 244
    feature_size = (7, 7)
    adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

    vcoco_path, feature_path, classes, image_ids, coco_ids = get_info(paths, imageset)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    if feature_mode != 'None':
        vgg16, transform = get_model(paths)

    # det_res = pickle.load(open('/home/tengyu/Documents/mmdetection/outputs/coco.%s.faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pkl'%vcoco_mapping[imageset], 'rb'))
    siyuan_det_path = '/home/tengyu/Documents/PartGPNN/gpnn/tmp/vcoco/vcoco_features'

    print('Extracting %s'%imageset)

    total = len(image_ids)
    t0 = time.time()

    for i_image, img_id in enumerate(image_ids):
        # FIXME: det_res uses a different set of images than image_names
        det_idx = coco_ids.index(img_id)
        img_name = 'COCO_%s2014_%012d.jpg'%(vcoco_mapping[imageset], img_id)

        # if not os.path.exists(os.path.join(siyuan_det_path, img_name + '.p')):
        #     continue

        # if os.path.exists(os.path.join(feature_path, '{}_obj_classes'.format(img_name))):
        #     continue

        # if not os.path.exists(os.path.join('/home/tengyu/Documents/densepose/DensePoseData/infer_out/%s/%s.pkl'%(vcoco_mapping[imageset], img_name))):
        #     continue

        # Read image
        image_path = os.path.join(vcoco_path, 'coco/', '{}2014'.format(vcoco_mapping[imageset]), img_name)
        assert os.path.exists(image_path)
        original_img = scipy.misc.imread(image_path, mode='RGB')

        obj_boxes_all = np.empty((0,4))
        obj_classes_all = list()
        part_boxes_all = np.empty((0,4))
        part_classes_all = list()
        part_human_id = list()
        edge_boxes_all = np.empty((0,4))
        edge_human_id = list()

        plt.imshow(original_img)
        # Read object detections
        instance = pickle.load(open(os.path.join(siyuan_det_path, img_name + '.p'), 'rb'), encoding='latin1')
        obj_boxes_all = instance['boxes'][instance['human_num']:]
        obj_classes_all = instance['classes'][instance['human_num']:]

        # Read human detections
        # boxes, bodies = pickle.load(open(os.path.join('/home/tengyu/Documents/densepose/DensePoseData/infer_out/%s/%s.pkl'%(vcoco_mapping[imageset], img_name)), 'rb'), encoding='latin-1')
        openpose = json.load(open(os.path.join(os.path.dirname(__file__), '../../../../data/openpose/vcoco/%s/%s.json'%(vcoco_mapping[imageset], img_name))))

        for human_id, human in enumerate(openpose['people']):
            keypoints = np.array(human['pose_keypoints_2d']).reshape([-1,3])
            w, h, _ = np.max(keypoints, axis=0) - np.min(keypoints, axis=0)
            for part_id, part_name in enumerate(part_names):
                x, y, s = keypoints[part_ids[part_name]]
                _box = [[y - h * 0.1, x - w * 0.1, y + h * 0.1, x + w * 0.1]]
                part_boxes_all = np.vstack([part_boxes_all, _box])
                part_classes_all.append(part_id)
                part_human_id.append(human_id)
                plt.plot([y0,y0,y1,y1,y0], [x0,x1,x1,x0,x0])

                for obj_box in obj_boxes_all:
                    edge_box = combine_box(obj_box, part_boxes_all[-1,:])
                    edge_human_id.append(human_id)
                    edge_boxes_all = np.vstack([edge_boxes_all, [edge_box]])
        plt.show()

        # for human_id in range(len(boxes[1])):
        #     if boxes[1][human_id][4] < 0.7:
        #         continue
        #     for part_id, part_name in enumerate(part_names):
        #         x, y = np.where(np.isin(bodies[1][human_id], part_ids[part_name]))
        #         x = x + boxes[1][human_id][1]
        #         y = y + boxes[1][human_id][0]
        #         if len(x) > 0:
        #             x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        #             part_boxes_all = np.vstack([part_boxes_all, np.array([[y0,x0,y1,x1]])])
        #             part_classes_all.append(part_id)
        #             part_human_id.append(human_id)

        #             # plt.plot([y0,y0,y1,y1,y0], [x0,x1,x1,x0,x0])

        #             # Add edges
        #             for obj_box in obj_boxes_all:
        #                 edge_box = combine_box(obj_box, part_boxes_all[-1,:])
        #                 edge_human_id.append(human_id)
        #                 edge_boxes_all = np.vstack([edge_boxes_all, [edge_box]])
        
        # plt.show()
        
        # Get image feature by applying VGG to ROI (roi_vgg)
        if feature_mode == 'resnet':
            # roi_features = np.empty((det_boxes_all.shape[0], 512, 7, 7))
            obj_features = np.zeros((obj_boxes_all.shape[0], 1000))
            part_features = np.zeros((part_boxes_all.shape[0], 1000))
            edge_features = np.zeros((edge_boxes_all.shape[0], 1000))
            for i_box in range(obj_boxes_all.shape[0]):
                obj = obj_boxes_all[i_box, :].astype(int)
                obj_image = original_img[obj[1]:obj[3]+1, obj[0]:obj[2]+1, :]
                # print(classes[obj_classes_all[i_box]+1])
                # plt.imshow(obj_image)
                # plt.show()
                obj_image = transform(cv2.resize(obj_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
                obj_image = torch.autograd.Variable(obj_image.unsqueeze(0)).cuda()
                feat, pred = vgg16(obj_image)
                obj_features[i_box, ...] = feat.data.cpu().numpy()


            for i_box in range(part_boxes_all.shape[0]):
                part = part_boxes_all[i_box, :].astype(int)
                part_image = original_img[part[1]:part[3]+1, part[0]:part[2]+1, :]
                # print(part_names[part_classes_all[i_box]])
                # plt.imshow(part_image)
                # plt.show()
                part_image = transform(cv2.resize(part_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
                part_image = torch.autograd.Variable(part_image.unsqueeze(0)).cuda()
                feat, pred = vgg16(part_image)
                part_features[i_box, ...] = feat.data.cpu().numpy()

            for i_box in range(edge_boxes_all.shape[0]):
                edge = edge_boxes_all[i_box, :].astype(int)
                edge_image = original_img[edge[1]:edge[3]+1, edge[0]:edge[2]+1, :]
                # print classes[det_classes_all[i_box]]
                # plt.imshow(roi_image)
                # plt.show()
                edge_image = transform(cv2.resize(edge_image, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
                edge_image = torch.autograd.Variable(edge_image.unsqueeze(0)).cuda()
                feat, pred = vgg16(edge_image)
                edge_features[i_box, ...] = feat.data.cpu().numpy()

        # Get image feature by extracting ROI from VGG feature (vgg_roi)
        if feature_mode == 'vgg_roi':
            scale_h = feature_size[0]/float(original_img.shape[0])
            scale_w = feature_size[1]/float(original_img.shape[1])
            transform_img = transform(cv2.resize(original_img, (input_h, input_w), interpolation=cv2.INTER_LINEAR))
            transform_img = torch.autograd.Variable(transform_img.unsqueeze(0)).cuda()
            img_feature = vgg16(transform_img)

            obj_features = np.empty((obj_boxes_all.shape[0], 512, feature_size[0], feature_size[1]))
            objs = np.copy(obj_boxes_all)
            objs[:, 0] *= scale_w
            objs[:, 2] *= scale_w
            objs[:, 1] *= scale_h
            objs[:, 3] *= scale_h

            for i_box in range(objs.shape[0]):
                obj = objs[i_box, :].astype(int)
                obj_feature = adaptive_max_pool(img_feature[..., obj[1]:(obj[3] + 1), obj[0]:(obj[2] + 1)])
                obj_features[i_box, :, :, :] = obj_feature.data.cpu().numpy()
            
            part_features = np.empty((part_boxes_all.shape[0], 512, feature_size[0], feature_size[1]))
            parts = np.copy(part_boxes_all)
            parts[:, 0] *= scale_w
            parts[:, 2] *= scale_w
            parts[:, 1] *= scale_h
            parts[:, 3] *= scale_h

            for i_box in range(parts.shape[0]):
                part = parts[i_box, :].astype(int)
                part_feature = adaptive_max_pool(img_feature[..., part[1]:(part[3] + 1), part[0]:(part[2] + 1)])
                part_features[i_box, :, :, :] = part_feature.data.cpu().numpy()
            
            edge_features = np.empty((edge_boxes_all.shape[0], 512, feature_size[0], feature_size[1]))
            edges = np.copy(edge_boxes_all)
            edges[:, 0] *= scale_w
            edges[:, 2] *= scale_w
            edges[:, 1] *= scale_h
            edges[:, 3] *= scale_h

            for i_box in range(edges.shape[0]):
                edge = edges[i_box, :].astype(int)
                edge_feature = adaptive_max_pool(img_feature[..., edge[1]:(edge[3] + 1), edge[0]:(edge[2] + 1)])
                edge_features[i_box, :, :, :] = edge_feature.data.cpu().numpy()
            
        np.save(os.path.join(feature_path, '{}_obj_classes'.format(img_name)), obj_classes_all)
        np.save(os.path.join(feature_path, '{}_obj_boxes'.format(img_name)), obj_boxes_all)
        if feature_mode != 'None':
            np.save(os.path.join(feature_path, '{}_obj_features'.format(img_name)), obj_features)

        np.save(os.path.join(feature_path, '{}_part_classes'.format(img_name)), part_classes_all)
        np.save(os.path.join(feature_path, '{}_part_boxes'.format(img_name)), part_boxes_all)
        np.save(os.path.join(feature_path, '{}_part_human_id'.format(img_name)), part_human_id)
        if feature_mode != 'None':
            np.save(os.path.join(feature_path, '{}_part_features'.format(img_name)), part_features)

        np.save(os.path.join(feature_path, '{}_edge_human_id'.format(img_name)), edge_human_id)
        np.save(os.path.join(feature_path, '{}_edge_boxes'.format(img_name)), edge_boxes_all)
        if feature_mode != 'None':
            np.save(os.path.join(feature_path, '{}_edge_features'.format(img_name)), edge_features)

        t1 = time.time()
        elapsed = t1 - t0
        ETA = elapsed / (i_image + 1) * (total - i_image - 1)
        print('\r\t%d/%d, ETA: %s'%(i_image, total, str(datetime.timedelta(seconds=ETA))), end='')
        # break


def main():
    paths = vcoco_config.Paths()
    imagesets = ['train', 'val', 'test']
    for imageset in imagesets:
        extract_features(paths, imageset)
        # break


if __name__ == '__main__':
    main()
