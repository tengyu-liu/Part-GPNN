import copy
import os
import sys
import time
import threading
import random
from queue import Queue
import pickle
import numpy as np

import vsrl_utils as vu

action_classes = ['none', 'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']
roles = ['none', 'obj', 'instr']

class IOThread(threading.Thread):
    def __init__(self, filename_queue, item_queue):
        self.fq = filename_queue
        self.iq = item_queue
        super(IOThread, self).__init__()
    
    def run(self):
        while not self.fq.empty():
            filename = self.fq.get()
            try:
                self.iq.put((pickle.load(open(filename, 'rb')), filename))
            except:
                pass
            self.fq.task_done()
        self.iq.put((None, None))

class BatchThread(threading.Thread):
    def __init__(self, filenames, node_num, negative_suppression=False, n_jobs=16, part_weight='central'):

        self.batch_queue = Queue(maxsize=100)
        self.item_queue = Queue(maxsize=100)
        self.filename_queue = Queue()
        for fn in filenames:
            self.filename_queue.put(fn)
        self.node_num = node_num
        self.negative_suppression = negative_suppression
        self.part_weight = part_weight
        self.n_jobs = n_jobs

        for i in range(n_jobs):
            t = IOThread(self.filename_queue, self.item_queue)
            t.start()

        super(BatchThread, self).__init__()
    
    def run(self):
        self.node_features = []
        self.edge_features = []
        self.adj_mat = []
        self.gt_action_labels = []
        self.gt_action_roles = []
        self.gt_strength_level = []
        self.part_human_ids = []
        self.batch_node_num = 0
        self.data_fn = []
        self.pairwise_action_mask = []
        self.img_ids = []
        self.human_boxes = []
        self.obj_nums = []
        self.part_nums = []
        self.obj_boxes = []
        self.obj_classes = []

        empty_count = 0

        while True:
            item, filename = self.item_queue.get()
            if item is None:
                empty_count += 1
                if empty_count < self.n_jobs:
                    continue
            if item is not None and item['node_num'] > 400: continue
            if empty_count == self.n_jobs or max(self.batch_node_num, item['node_num']) * (len(self.node_features) + 1) > self.node_num:
                node_features = np.zeros([len(self.node_features), self.batch_node_num, 1108])
                edge_features = np.zeros([len(self.edge_features), self.batch_node_num, self.batch_node_num, 1216])
                adj_mat = np.zeros([len(self.adj_mat), self.batch_node_num, self.batch_node_num])
                gt_strength_level = np.zeros([len(self.gt_strength_level), self.batch_node_num, self.batch_node_num])
                gt_action_labels = np.zeros([len(self.gt_action_labels), self.batch_node_num, self.batch_node_num, len(action_classes)])
                gt_action_roles = np.zeros([len(self.gt_action_roles), self.batch_node_num, self.batch_node_num, len(action_classes), len(roles)])
                pairwise_action_mask = np.zeros(gt_action_labels.shape)

                for i_file in range(len(self.node_features)):
                    node_num = len(self.node_features[i_file])
                    node_features[i_file, :node_num, :] = self.node_features[i_file]
                    edge_features[i_file, :node_num, :node_num, :] = self.edge_features[i_file]
                    adj_mat[i_file, :node_num, :node_num] = self.adj_mat[i_file]
                    if self.part_weight == 'central':
                        gt_strength_level[i_file, :node_num, :node_num] = self.gt_strength_level[i_file]
                    elif self.part_weight == 'edge':
                        gt_strength_level[i_file, :node_num, :node_num] = 1.1 - self.gt_strength_level[i_file]
                    elif self.part_weight == 'uniform':
                        gt_strength_level[i_file, :node_num, :node_num] = 1.0
                    else:
                        raise NotImplemented
                    gt_action_labels[i_file, :node_num, :node_num, 1:] = self.gt_action_labels[i_file]
                    gt_action_roles[i_file, :node_num, :node_num, 1:, 1:] = self.gt_action_roles[i_file]
                    gt_action_labels[i_file, :node_num, :node_num, 0] = (np.sum(self.gt_action_labels[i_file][:, :, 1:]) == 0).astype(float)
                    gt_action_roles[i_file, :node_num, :node_num, 0] = (np.sum(self.gt_action_roles[i_file][:, :, 1:, 1:]) == 0).astype(float)
                    pairwise_action_mask[i_file, :node_num, :node_num, :] = self.pairwise_action_mask[i_file]

                batch = (node_features, 
                        edge_features, 
                        adj_mat, 
                        gt_action_labels, 
                        gt_action_roles, 
                        gt_strength_level, 
                        copy.deepcopy(self.part_human_ids), 
                        copy.deepcopy(self.human_boxes), 
                        pairwise_action_mask, 
                        self.batch_node_num, 
                        copy.deepcopy(self.data_fn), 
                        copy.deepcopy(self.obj_nums),
                        copy.deepcopy(self.part_nums),
                        copy.deepcopy(self.obj_boxes),
                        copy.deepcopy(self.obj_classes),
                        copy.deepcopy(self.img_ids))

                self.batch_queue.put(batch)
                self.item_queue.task_done()

                if item is None: 
                    self.batch_queue.put(None)
                    return # End of epoch

                self.node_features = []
                self.edge_features = []
                self.adj_mat = []
                self.gt_action_labels = []
                self.gt_action_roles = []
                self.gt_strength_level = []
                self.part_human_ids = []
                self.batch_node_num = 0
                self.data_fn = []
                self.pairwise_action_mask = []
                self.img_ids = []
                self.human_boxes = []
                self.obj_nums = []
                self.part_nums = []
                self.obj_boxes = []
                self.obj_classes = []
            
            self.node_features.append(item['node_features'])# [i_file, :node_num, :] = data['node_features']
            self.edge_features.append(item['edge_features'])# [i_file, :node_num, :node_num, :] = data['edge_features']
            self.adj_mat.append(item['adj_mat'])# [i_file, :node_num, :node_num] = data['adj_mat']
            self.gt_strength_level.append(item['strength_level'])# [i_file, :node_num, :node_num] = data['strength_level']
            self.gt_action_labels.append(item['action_labels'])# [i_file, :node_num, :node_num, 1:] = data['action_labels']
            self.gt_action_roles.append(item['action_roles'])# [i_file, :node_num, :node_num, 1:] = data['action_roles']
            self.part_human_ids.append(item['part_human_id'])
            self.batch_node_num = max(self.batch_node_num, item['node_features'].shape[0])
            self.data_fn.append(filename)
            # self.pairwise_action_mask.append(item['pairwise_action_mask'])
            self.pairwise_action_mask.append(1)
            self.img_ids.append(item['img_id'])
            human_box = np.array(item['part_boxes'])[np.array(item['part_classes']) == 18]
            self.human_boxes.append(human_box)
            self.obj_nums.append(item['obj_num'])
            self.part_nums.append(item['part_num'])
            self.obj_boxes.append(item['obj_boxes'])
            self.obj_classes.append(item['obj_classes'])


class DataLoader:
    def __init__(self, imageset, node_num, datadir=os.path.join(os.path.dirname(__file__), '../../data/feature_resnet_tengyu2'), negative_suppression=False, n_jobs=16, part_weight='central', debug=None):
        self.imageset = imageset
        self.datadir = datadir
        self.node_num = node_num
        self.negative_suppression = negative_suppression
        self.part_weight = part_weight
        self.n_jobs = n_jobs

        self.thread = None

        self.coco = vu.load_coco('/home/tengyu/dataset/v-coco/data')
        vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), '/home/tengyu/dataset/v-coco/data')
        # self.coco = vu.load_coco('/home/tengyu/Data/mscoco/v-coco/data')
        # vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), '/home/tengyu/Data/mscoco/v-coco/data')

        self.filenames = list(set([os.path.join(self.datadir, x['file_name'] + '.data') for x in self.coco.loadImgs(ids=vcoco_all[0]['image_id'][:, 0].astype(int).tolist()) if os.path.exists(os.path.join(self.datadir, x['file_name'] + '.data'))]))[:100]
        if debug is not None:
            self.filenames = [x for x in self.filenames if '%012d'%debug in x]

        pass

    def __len__(self):
        return len(self.filenames)

    def shuffle(self):
        random.shuffle(self.filenames)
    
    def prefetch(self):
        if self.thread is not None:
            self.thread.join()
        self.thread = BatchThread(self.filenames, self.node_num, negative_suppression=self.negative_suppression, n_jobs=self.n_jobs, part_weight=self.part_weight)
        self.thread.start()
        
    def fetch(self):
        res = self.thread.batch_queue.get()
        self.thread.batch_queue.task_done()
        return res

if __name__ == "__main__":
    import time
    n_jobs = 16
    dl = DataLoader('val', 400, negative_suppression=True, n_jobs=n_jobs)
    dl.shuffle()
    dl.prefetch()
    item_count = 0
    total_time = 0
    for i in range(10):
        t0 = time.time()
        res = dl.fetch()
        t1 = time.time()
        item_count += res[0].shape[0]
        total_time += t1 - t0
        print('\rVCOCO %d IO Thread'%n_jobs, total_time, item_count, total_time / item_count, end='', flush=True)
    print('\rVCOCO %d IO Thread'%n_jobs, total_time, item_count, total_time / item_count)
    print('Finished')
