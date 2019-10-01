import os
import pickle

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import vsrl_utils as vu

imageset = 'train'

base_dir = '/home/tengyu/Data/mscoco/v-coco/processed/resnet'
# base_dir = '/home/tengyu/Documents/PartGPNN/gpnn/tmp/vcoco/vcoco_features'
img_dir = '/home/tengyu/Data/mscoco/coco/'

colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'black', 'white']

coco = vu.load_coco('/home/tengyu/Data/mscoco/v-coco/data')
vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), '/home/tengyu/Data/mscoco/v-coco/data')
for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)

image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()

for fn in os.listdir(base_dir):
    if '.p' not in fn:
        continue
    base_fn = fn[:-2]
    if imageset not in base_fn:
        continue
    if int(base_fn[:-4].split('_')[-1]) not in image_ids:
        continue
    print(fn)
    # imageset = fn.split('_')[1]
    img = scipy.misc.imread(os.path.join(img_dir, imageset + '2014', base_fn))

    instance = pickle.load(open(os.path.join(base_dir, fn), 'rb'))

    # show original image
    plt.subplot(231)
    plt.imshow(img)
    # show human box
    plt.subplot(232)
    plt.imshow(img)
    for i in range(instance['human_num']):
        x0,y0,x1,y1 = instance['human_boxes'][i]
        plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], c=colors[i%len(colors)])
    # show part box
    plt.subplot(233)
    plt.imshow(img)
    for i in range(instance['part_num']):
        x0,y0,x1,y1 = instance['part_boxes'][i]
        human_num = instance['part_human_id'][i]
        plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], c=colors[human_num%len(colors)])
    # show object box
    plt.subplot(234)
    plt.imshow(img)
    for i in range(instance['obj_num']):
        x0,y0,x1,y1 = instance['obj_boxes'][i]
        if instance['obj_classes'][i] != 0:
            plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], c=colors[-i%len(colors)])
        print('obj', colors[-i%len(colors)])
        print(instance['obj_boxes'][i])
    # show edge box
    plt.subplot(235)
    plt.imshow(img)
    edge_i = 0
    for human_i in range(instance['human_num']):
        for obj_i in range(instance['obj_num']):
            _obj_i = obj_i + instance['human_num']
            if instance['adj_mat'][human_i, _obj_i] == 1:
                hx0,hy0,hx1,hy1 = instance['human_boxes'][human_i]
                ox0,oy0,ox1,oy1 = instance['obj_boxes'][obj_i]
                plt.plot([hx0,hx0,hx1,hx1,hx0], [hy0,hy1,hy1,hy0,hy0], c=colors[edge_i%len(colors)])
                plt.plot([ox0,ox0,ox1,ox1,ox0], [oy0,oy1,oy1,oy0,oy0], c=colors[edge_i%len(colors)])
                edge_i += 1
    # show gt bboxes
    plt.subplot(236)
    plt.imshow(img)
    hoi_i = 0
    for action_i in range(len(vcoco_all)):
        for img_i in range(len(vcoco_all[action_i]['image_id'])):
            if int(vcoco_all[action_i]['image_id'][img_i]) == int(base_fn[:-4].split('_')[-1]):
                if vcoco_all[action_i]['label'][img_i][0] == 1:
                    print(vcoco_all[action_i]['action_name'], colors[hoi_i%len(colors)])
                    role_bbox = vcoco_all[action_i]['role_bbox'][img_i].reshape([-1,4])
                    for box_i in range(len(role_bbox)):
                        x0,y0,x1,y1 = role_bbox[box_i]
                        if np.isnan(x0):
                            continue
                        print(role_bbox[box_i])
                        plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], c=colors[hoi_i%len(colors)])
                    hoi_i += 1

    print(instance['human_num'], instance['obj_num'])
    print(instance['adj_mat'])
    plt.show()

    # TODO: use object box from vcoco_features.p