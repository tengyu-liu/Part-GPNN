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

def get_box(_box, human_boxes_all, used_human):
    max_iou = 0
    best_box = None
    best_i = None
    for i, box in enumerate(human_boxes_all):
        if i in used_human:
            continue
        iou = compute_iou(_box, box)
        if iou > max_iou:
            max_iou = iou
            best_box = box
            best_i = i
    return best_i, best_box

def show_img(img_id):
    img = sio.imread('/home/tengyu/Data/mscoco/coco/train2014/COCO_train2014_%012d.jpg'%img_id)
    _ = plt.imshow(img)
    _ = plt.show()


train_vcocoeval = get_vcocoeval('train')
vcocodb = train_vcocoeval._get_vcocodb()

random.seed(0)
np.random.seed(0)

import vsrl_utils as vu
vcoco_root = '/home/tengyu/Data/mscoco/v-coco/data'
coco = vu.load_coco(vcoco_root)
vcoco_all = vu.load_vcoco('vcoco_{}'.format('train'), vcoco_root)
for x in vcoco_all:
    x = vu.attach_gt_boxes(x, coco)

x = vcoco_all[metadata.action_classes.index('eat')-1]
for img_id in x['image_id']:
    img_id = img_id[0]
    image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
    i_image = image_ids.index(img_id)
    if x['label'][i_image, 0] == 1:
        break

train_loader = DataLoader('train', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs, part_weight=flags.part_weight, debug=1298)

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

all_results_sum, all_results_max, all_results_mean = metrics.append_results(
    all_results_sum, all_results_max, all_results_mean, 
    human_boxes, part_human_ids, gt_action_labels, gt_action_roles, 
    obj_nums, obj_boxes, obj_classes, img_ids)

i_item = 0

gt_item = [x for x in vcocodb if x['id'] == img_ids[i_item]][0]

img = sio.imread('/home/tengyu/Data/mscoco/coco/train2014/COCO_train2014_%012d.jpg'%img_ids[i_item])
_ = plt.imshow(img)
_ = plt.show()

# compare human boxes
boxes_all = human_boxes[i_item]

_ = plt.imshow(img)
for i, box in enumerate(gt_item['boxes']):
    box_class = metadata.coco_classes[gt_item['gt_classes'][i]]
    if box_class == 'person':
        draw_box(box, 'blue')

for i, box in enumerate(boxes_all):
    draw_box(box, 'red')

plt.show()

# compare obj boxes
boxes_all = obj_boxes[i_item]

plt.subplot(121)
_ = plt.imshow(img)
for i, box in enumerate(gt_item['boxes']):
    box_class = metadata.coco_classes[gt_item['gt_classes'][i]]
    if box_class != 'person':
        draw_box(box, 'blue')

plt.subplot(122)
_ = plt.imshow(img)
for i, box in enumerate(boxes_all):
    draw_box(box, 'red')

plt.show()

# compare action labels
for i_human in set(part_human_ids[i_item]):
    label_sum = np.sum(gt_action_labels[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0).sum(0)
    plt.imshow(img)
    draw_box(human_boxes[i_item][i_human], 'red')
    for i_action in range(len(label_sum)):
        if label_sum[i_action] > 0:
            print(metadata.action_classes[i_action] + ', ', end='')
    print()
    i_gt_human, gt_human_box = get_box(human_boxes[i_item][i_human], gt_item['boxes'][gt_item['gt_classes']==1], [])
    i_gt_human = np.where(gt_item['gt_classes'] == 1)[0][i_gt_human]
    draw_box(gt_human_box, 'blue')
    gt_item_actions = gt_item['gt_actions'][i_gt_human]
    gt_item_roles = gt_item['gt_role_id'][i_gt_human]
    for i_action in range(len(gt_item_actions)):
        if gt_item_actions[i_action] > 0:
            print(metadata.action_classes[i_action + 1] + ', ', end='')
    print()
    plt.show()

# inspect action label generation
image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()
i_image = image_ids.index(img_ids[i_item])
action_label = 'sit'
x = vcoco_all[metadata.action_classes.index(action_label) - 1]
assert x['label'][i_image, 0] == 1
# Draw human boxes
role_bbox = x['role_bbox'][i_image, :] * 1.
role_bbox = role_bbox.reshape((-1, 4))
bbox = role_bbox[0, :]
human_index = get_node_index(bbox, human_boxes[i_item], np.arange(len(human_boxes[i_item])))   # node_index uses human box

assert human_index < len(human_boxes[i_item])
for i_role in range(1, len(x['role_name'])):
    bbox = role_bbox[i_role, :]
    print(bbox)


plt.imshow(img)
draw_box(bbox, 'blue')
draw_box(human_boxes[i_item][human_index], 'red')
plt.show()

# # compare action roles
# for i_human in set(part_human_ids[i_item]):
#     role_sum = np.sum(gt_action_roles[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0)

# compare role boxes
# GT
_ = plt.subplot(121)
_ = plt.imshow(img)
for i_box, i_action, i_role in zip(*np.where(gt_item['gt_role_id'] > 0)):
    j_box = gt_item['gt_role_id'][i_box, i_action, i_role]
    draw_box(gt_item['boxes'][i_box], 'blue')
    draw_box(gt_item['boxes'][j_box], 'red')
    print(metadata.action_classes[i_action + 1], metadata.roles[i_role + 1])

plt.show()

