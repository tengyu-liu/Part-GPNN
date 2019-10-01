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

# import vsrl_utils as vu
import hico_config
import roi_pooling
import roi_feature_model
import metadata

import matplotlib.pyplot as plt

feature_mode = 'resnet'

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
            'Head': [23, 24]
            }

part_names = list(part_ids.keys()) 

def get_model(paths, feature_type):
    if feature_type == 'vgg':
        feature_network = roi_feature_model.Vgg16(num_classes=len(metadata.action_classes))
    elif feature_type == 'resnet':
        feature_network = roi_feature_model.Resnet152(num_classes=len(metadata.action_classes))
    elif feature_type == 'densenet':
        feature_network = roi_feature_model.Densenet(num_classes=len(metadata.action_classes))
    else:
        raise ValueError('feature type not recognized')

    if feature_type.startswith('alexnet') or feature_type.startswith('vgg'):
        feature_network.features = torch.nn.DataParallel(feature_network.features)
        feature_network.cuda()
    else:
        feature_network = torch.nn.DataParallel(feature_network).cuda()

    checkpoint_dir = os.path.join(paths.tmp_root, 'checkpoints', 'hico', 'finetune_{}'.format(feature_type))
    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
    checkpoint = torch.load(best_model_file)
    feature_network.load_state_dict(checkpoint['state_dict'])

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    return feature_network, transform

def combine_box(box1, box2):
    return np.hstack((np.minimum(box1[:2], box2[:2]), np.maximum(box1[2:], box2[2:])))


def get_info(paths):
    hico_path = paths.data_root
    hico_voc_path = os.path.join(hico_path, 'Deformable-ConvNets/data/hico/VOC2007')
    feature_path = os.path.join(hico_path, 'processed', 'features_roi_vgg')

    # image_list_file = os.path.join(hico_voc_path, 'ImageSets/Main/test.txt')
    # det_res_path = os.path.join(hico_path, 'Deformable-ConvNets/output/rfcn_dcn/hico/hico_detect/2007_test',
    #                             'hico_detect_test_detections.pkl')

    image_list_file = os.path.join(hico_voc_path, 'ImageSets/Main/trainvaltest.txt')
    det_res_path = os.path.join(hico_path, 'Deformable-ConvNets/output/rfcn_dcn/hico/hico_detect/2007_trainvaltest',
                                'hico_detect_trainvaltest_detections.pkl')

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

    return hico_path, hico_voc_path, det_res_path, feature_path, classes, image_list_file


def extract_features(paths, imageset):
    input_h, input_w = 244, 244
    feature_size = (7, 7)
    adaptive_max_pool = roi_pooling.AdaptiveMaxPool2d(*feature_size)

    hico_path, hico_voc_path, det_res_path, feature_path, classes, image_list_file = get_info(paths)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    vgg16, transform = get_model(paths, feature_mode)

    det_res = pickle.load(open('/home/tengyu/Documents/mmdetection/outputs/hico-det.%s.pkl'%imageset, 'rb'))

    print('Extracting %s'%imageset)

    total = len(list(det_res.keys()))
    t0 = time.time()

    for i_image, img_fn in enumerate(det_res.keys()):

        if not os.path.exists(os.path.join('/home/tengyu/Documents/densepose/DensePoseData/infer_out/hico-det/%s/%s.pkl'%(imageset, img_fn))):
            continue
        
        if os.path.exists(os.path.join(feature_path, '{}_edge_features'.format(img_fn))):
            continue

        # Read image
        image_path = os.path.join(hico_path, 'images/', '{}2015'.format(imageset), img_fn)
        assert os.path.exists(image_path)
        original_img = cv2.imread(image_path)

        obj_boxes_all = np.empty((0,4))
        obj_classes_all = list()
        part_boxes_all = np.empty((0,4))
        part_classes_all = list()
        part_human_id = list()
        edge_boxes_all = np.empty((0,4))
        edge_human_id = list()

        # plt.imshow(original_img)

        # Read human detections
        for c in range(2, len(classes)):
            for detection in det_res[img_fn][c-1]:
                if detection[4] > 0.7:
                    y0,x0,y1,x1 = detection[0], detection[1], detection[2], detection[3]
                    # plt.plot([y0,y0,y1,y1,y0], [x0,x1,x1,x0,x0], c='red')
                    obj_boxes_all = np.vstack((obj_boxes_all, np.array(detection[:4])[np.newaxis, ...]))
                    obj_classes_all.append(c-1)
        if len(obj_classes_all) == 0:
            continue

        boxes, bodies = pickle.load(open(os.path.join('/home/tengyu/Documents/densepose/DensePoseData/infer_out/hico-det/%s/%s.pkl'%(imageset, img_fn)), 'rb'), encoding='latin-1')

        for human_id in range(len(boxes[1])):
            if boxes[1][human_id][4] < 0.7:
                continue
            for part_id, part_name in enumerate(part_names):
                x, y = np.where(np.isin(bodies[1][human_id], part_ids[part_name]))
                x = x + boxes[1][human_id][1]
                y = y + boxes[1][human_id][0]
                if len(x) > 0:
                    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
                    part_boxes_all = np.vstack([part_boxes_all, np.array([[y0,x0,y1,x1]])])
                    part_classes_all.append(part_id)
                    part_human_id.append(human_id)

                    # plt.plot([y0,y0,y1,y1,y0], [x0,x1,x1,x0,x0], c='blue')

                    # Add edges
                    for obj_box in obj_boxes_all:
                        edge_box = combine_box(obj_box, part_boxes_all[-1,:])
                        edge_human_id.append(human_id)
                        edge_boxes_all = np.vstack([edge_boxes_all, [edge_box]])
        
        # plt.show()

        # Get image feature by applying VGG to ROI (roi_vgg)
        if feature_mode == 'resnet':
            # roi_features = np.empty((det_boxes_all.shape[0], 512, 7, 7))
            obj_features = np.zeros((obj_boxes_all.shape[0], 200))
            part_features = np.zeros((part_boxes_all.shape[0], 200))
            edge_features = np.zeros((edge_boxes_all.shape[0], 200))
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
            
        np.save(os.path.join(feature_path, '{}_obj_classes'.format(img_fn)), obj_classes_all)
        np.save(os.path.join(feature_path, '{}_obj_boxes'.format(img_fn)), obj_boxes_all)
        np.save(os.path.join(feature_path, '{}_obj_features'.format(img_fn)), obj_features)

        np.save(os.path.join(feature_path, '{}_part_classes'.format(img_fn)), part_classes_all)
        np.save(os.path.join(feature_path, '{}_part_boxes'.format(img_fn)), part_boxes_all)
        np.save(os.path.join(feature_path, '{}_part_human_id'.format(img_fn)), part_human_id)
        np.save(os.path.join(feature_path, '{}_part_features'.format(img_fn)), part_features)

        np.save(os.path.join(feature_path, '{}_edge_human_id'.format(img_fn)), edge_human_id)
        np.save(os.path.join(feature_path, '{}_edge_boxes'.format(img_fn)), edge_boxes_all)
        np.save(os.path.join(feature_path, '{}_edge_features'.format(img_fn)), edge_features)

        t1 = time.time()
        elapsed = t1 - t0
        ETA = elapsed / (i_image + 1) * (total - i_image - 1)
        print('\r\t%d/%d, ETA: %s'%(i_image, total, str(datetime.timedelta(seconds=ETA))), end='')
        # break


def main():
    paths = hico_config.Paths()
    imagesets = ['train', 'test']
    for imageset in imagesets:
        extract_features(paths, imageset)
        # break


if __name__ == '__main__':
    main()
