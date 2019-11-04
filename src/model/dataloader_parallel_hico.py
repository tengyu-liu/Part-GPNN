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

action_classes = ['adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy', 'carry', 'catch',
                   'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut_with', 'direct', 'drag', 'dribble',
                   'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly',
                   'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug', 'hunt', 'inspect',
                   'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light', 'load',
                  'lose', 'make', 'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay',
                  'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 'release',
                   'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 'sit_at',
                   'sit_on', 'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 'stick', 'stir',
                   'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast',
                   'train', 'turn', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip']

class IOThread(threading.Thread):
    def __init__(self, filename_queue, item_queue):
        self.fq = filename_queue
        self.iq = item_queue
        super(IOThread, self).__init__()
    
    def run(self):
        while not self.fq.empty():
            filename = self.fq.get()
            self.iq.put(pickle.load(open(filename, 'rb')))
            self.fq.task_done()
        self.iq.put(None)

class BatchThread(threading.Thread):
    def __init__(self, filenames, node_num, negative_suppression=False, n_job=16):

        self.batch_queue = Queue()
        self.item_queue = Queue()
        self.filename_queue = Queue()
        for fn in filenames:
            self.filename_queue.put(fn)
        self.node_num = node_num
        self.negative_suppression = negative_suppression

        for i in range(n_job):
            t = IOThread(self.filename_queue, self.item_queue)
            t.start()

        super(DataThread, self).__init__()
    
    def run(self):
        cur_node_num = 0

        self.node_features = []
        self.edge_features = []
        self.adj_mat = []
        self.gt_action_labels = []
        self.gt_strength_level = []
        self.part_human_ids = []
        self.batch_node_num = -1
        self.data_fn = []
        self.pairwise_action_mask = []
        self.part_list = []
        self.part_classes = []

        while True:
            item = self.item_queue.get()
            if item['node_num'] + cur_node_num > self.node_num or item is None:
                node_features = np.zeros([len(self.node_features), self.batch_node_num, 1108])
                edge_features = np.zeros([len(self.edge_features), self.batch_node_num, self.batch_node_num, 1216])
                adj_mat = np.zeros([len(self.adj_mat), self.batch_node_num, self.batch_node_num])
                gt_strength_level = np.zeros([len(self.gt_strength_level), self.batch_node_num, self.batch_node_num])
                gt_action_labels = np.zeros([len(self.gt_action_labels), self.batch_node_num, self.batch_node_num, len(action_classes)])
                pairwise_action_mask = np.zeros(gt_action_labels.shape)

                for i_file in range(len(self.node_features)):
                    node_num = len(self.node_features[i_file])
                    node_features[i_file, :node_num, :] = self.node_features[i_file]
                    edge_features[i_file, :node_num, :node_num, :] = self.edge_features[i_file]
                    adj_mat[i_file, :node_num, :node_num] = self.adj_mat[i_file]
                    gt_strength_level[i_file, :node_num, :node_num] = self.gt_strength_level[i_file]
                    gt_action_labels[i_file, :node_num, :node_num, 1:] = self.gt_action_labels[i_file]
                    gt_action_labels[i_file, :node_num, :node_num, 0] = (np.sum(self.gt_action_labels[i_file][:, :, 1:]) == 0).astype(float)
                    pairwise_action_mask[i_file, :node_num, :node_num, :] = self.pairwise_action_mask[i_file]

                batch = (node_features, 
                        edge_features, 
                        adj_mat, 
                        gt_action_labels, 
                        gt_strength_level, 
                        copy.deepcopy(self.part_human_ids), 
                        pairwise_action_mask, 
                        copy.deepcopy(self.part_list),
                        copy.deepcopy(self.part_classes),
                        self.batch_node_num, copy.deepcopy(self.data_fn))

                self.batch_queue.put(batch)
                self.item_queue.task_done()

                if item is None: 
                    self.batch_queue.put(None)
                    self.item_queue.all_tasks_done()
                    return # End of epoch

                self.node_features = []
                self.edge_features = []
                self.adj_mat = []
                self.gt_action_labels = []
                self.gt_strength_level = []
                self.part_human_ids = []
                self.batch_node_num = -1
                self.data_fn = []
                self.pairwise_action_mask = []
                self.part_list = []
                self.part_classes = []
            
            self.node_features.append(item['node_features'])# [i_file, :node_num, :] = data['node_features']
            self.edge_features.append(item['edge_features'])# [i_file, :node_num, :node_num, :] = data['edge_features']
            self.adj_mat.append(item['adj_mat'])# [i_file, :node_num, :node_num] = data['adj_mat']
            self.gt_strength_level.append(item['strength_level'])# [i_file, :node_num, :node_num] = data['strength_level']
            self.gt_action_labels.append(item['action_labels'])# [i_file, :node_num, :node_num, 1:] = data['action_labels']
            self.part_human_ids.append(item['part_human_id'])
            self.batch_node_num = max(self.batch_node_num, item['node_features'].shape[0])
            self.data_fn.append(filename)
            self.pairwise_action_mask.append(data['pairwise_action_mask'])
            self.part_list.append(data['part_list'])
            self.part_classes.append(data['part_classes'])


class DataLoader:
    def __init__(self, imageset, node_num, datadir=os.path.join(os.path.dirname(__file__), '../../data/hico/feature'), negative_suppression=False, n_jobs=16):
        self.imageset = imageset
        self.datadir = datadir
        self.node_num = node_num
        self.negative_suppression = negative_suppression
        self.n_jobs = n_jobs

        self.thread = None

        self.filenames = [os.path.join(self.datadir, filename) for filename in os.listdir(datadir) if imageset in filename]

        pass

    def __len__(self):
        return len(self.filenames)

    def shuffle(self):
        random.shuffle(self.filenames)
    
    def prefetch(self):
        if self.thread is not None:
            self.thread.join()
        self.thread = DataThread(self.filenames, self.node_num, negative_suppression=self.negative_suppression, n_jobs=self.n_jobs)
        self.thread.start()
        
    def fetch(self):
        res = self.thread.batch_queue.get()
        self.thread.batch_queue.task_done()
        return res

if __name__ == "__main__":
    import time

    n_jobs = 16
    dl = DataLoader('train', 400, negative_suppression=True, n_jobs=n_jobs)
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
    print('\rHICO-Det %d IO Thread'%n_jobs, total_time / item_count, end='', flush=True)
    print('\rHICO-Det %d IO Thread'%n_jobs, total_time / item_count)
    print('Finished')