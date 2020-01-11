import datetime
import os
import pickle
import random
import sys
import time

import numpy as np

from config import flags
from dataloader_parallel import DataLoader
import metadata
import metrics
import vsrl_eval

vcoco_root = '/home/tengyu/dataset/v-coco'

def get_vcocoeval(imageset):
    return vsrl_eval.VCOCOeval(os.path.join(vcoco_root, 'data/vcoco/vcoco_{}.json'.format(imageset)),
                               os.path.join(vcoco_root, 'data/instances_vcoco_all_2014.json'),
                               os.path.join(vcoco_root, 'data/splits/vcoco_{}.ids'.format(imageset)))

def vcoco_evaluation(vcocoeval, imageset, all_results, name, method):
    print('\n%s: '%method, end='')
    det_file = os.path.join(os.path.dirname(__file__), 'eval', name, '%s_detections[%s].pkl'%(imageset, method))
    # pickle.dump(all_results, open(det_file, 'wb'))
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)
    print()

train_vcocoeval = get_vcocoeval('train')
val_vcocoeval = get_vcocoeval('val')
test_vcocoeval = get_vcocoeval('test')

# random.seed(0)
# np.random.seed(0)

# obj_action_pair = pickle.load(open(os.path.join(os.path.dirname(__file__), 'data', 'obj_action_pairs.pkl'), 'rb'))

# train_loader = DataLoader('train', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs, part_weight=flags.part_weight)
# val_loader = DataLoader('val', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs, part_weight=flags.part_weight)
# test_loader = DataLoader('test', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs, part_weight=flags.part_weight)


# train_loader.shuffle()
# train_loader.prefetch()
# item = 0
# total_data_time = 0
# total_tf_time = 0

all_results_sum = []
# all_results_mean = []
# all_results_max = []

# while True:
#     t0 = time.time()

#     res = train_loader.fetch()
#     if res is None:
#         break
#     node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, \
#             part_human_ids, human_boxes, pairwise_label_mask, batch_node_num, fns, \
#             obj_nums, part_nums, obj_boxes, obj_classes, img_ids = res
    
#     all_results_sum, all_results_max, all_results_mean = metrics.append_results(
#         all_results_sum, all_results_max, all_results_mean, human_boxes, 
#         part_human_ids, gt_action_labels, gt_action_roles, obj_nums, obj_boxes, obj_classes, img_ids)

vcoco_evaluation(train_vcocoeval, 'train', all_results_sum, flags.name, 'SUM')
# vcoco_evaluation(train_vcocoeval, 'train', all_results_max, flags.name, 'MAX')
# vcoco_evaluation(train_vcocoeval, 'train', all_results_mean, flags.name, 'MEAN')

