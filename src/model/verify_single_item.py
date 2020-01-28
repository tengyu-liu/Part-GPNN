import datetime
import os
import pickle
import random
import sys
import time

import numpy as np
import skimage.io as sio
import matplotlib.pyplot as plt

from config import flags
from dataloader_parallel import DataLoader
import metadata
import metrics
import vsrl_eval

vcoco_root = '/home/tengyu/Data/mscoco/v-coco'

def get_vcocoeval(imageset):
    return vsrl_eval.VCOCOeval(os.path.join(vcoco_root, 'data/vcoco/vcoco_{}.json'.format(imageset)),
                               os.path.join(vcoco_root, 'data/instances_vcoco_all_2014.json'),
                               os.path.join(vcoco_root, 'data/splits/vcoco_{}.ids'.format(imageset)))

def vcoco_evaluation(vcocoeval, imageset, all_results, name, method):
    print('\n%s: '%method, end='')
    det_file = os.path.join(os.path.dirname(__file__), 'eval', name, '%s_detections[%s].pkl'%(imageset, method))
    pickle.dump(all_results, open(det_file, 'wb'))
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)
    print()

def draw_box(box, color='blue'):
    x0,y0,x1,y1 = box
    plt.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], c=color)

train_vcocoeval = get_vcocoeval('train')
vcocodb = train_vcocoeval._get_vcocodb()

random.seed(0)
np.random.seed(0)

train_loader = DataLoader('train', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs, part_weight=flags.part_weight)

train_loader.shuffle()
train_loader.prefetch()
item = 0
total_data_time = 0
total_tf_time = 0

all_results_sum = []
all_results_mean = []
all_results_max = []

res = train_loader.fetch()

node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, \
        part_human_ids, human_boxes, pairwise_label_mask, batch_node_num, fns, \
        obj_nums, part_nums, obj_boxes, obj_classes, img_ids = res

i_item = 2

gt_item = [x for x in vcocodb if x['id'] == img_ids[i_item]][0]

img = sio.imread('/home/tengyu/Data/mscoco/coco/train2014/COCO_train2014_%012d.jpg'%img_ids[i_item])
# _ = plt.imshow(img)
# _ = plt.show()

# compare human boxes
filename = 'COCO_train2014_%012d.jpg'%img_ids[i_item]
meta_dir = '../../data/vcoco_features'
image_meta = pickle.load(open(os.path.join(meta_dir, filename + '.p'), 'rb'), encoding='latin1')
boxes_all = image_meta['boxes'][:image_meta['human_num']]
print(boxes_all)

_ = plt.imshow(img)
for i, box in enumerate(gt_item['boxes']):
    box_class = metadata.coco_classes[gt_item['gt_classes'][i]]
    if box_class == 'person':
        draw_box(box, 'blue')

for i, box in enumerate(boxes_all):
    draw_box(box, 'red')

plt.show()



