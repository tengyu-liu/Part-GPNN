import copy
import os
import sys
import time
import threading
import random

import pickle
import numpy as np

action_classes = ['none', 'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']
roles = ['none', 'obj', 'instr']

class DataThread(threading.Thread):
    def __init__(self, filenames, node_num, with_name=False):
        # t0 = time.time()
        self.filenames = filenames
        self.node_num = node_num
        self.node_features = [] # np.zeros([len(filenames), self.node_num, 1108])
        self.edge_features = [] # np.zeros([len(filenames), self.node_num, self.node_num, 1216])
        self.adj_mat = [] # np.zeros([len(filenames), self.node_num, self.node_num])
        self.gt_strength_level = [] # np.zeros([len(filenames), self.node_num, self.node_num])
        self.gt_action_labels = [] # np.zeros([len(filenames), self.node_num, self.node_num, len(action_classes)])
        self.gt_action_roles = [] # np.zeros([len(filenames), self.node_num, self.node_num, len(roles)])
        self.part_human_ids = []
        self.batch_node_num = 0

        self.empty_count = threading.Semaphore(value=20)
        self.fill_count = threading.Semaphore(value=0)

        self.data_queue = []

        self.with_name = with_name

        self.data_fn = []

        super(DataThread, self).__init__()
        # print('Const: ', time.time() - t0)
    
    def run(self):
        # t0 = time.time()
        self.batch_node_num = -1
        node_num_cap = 400

        while len(self.filenames) > 0:
            filename = self.filenames.pop(0)
            data = pickle.load(open(filename, 'rb'))
            node_num = data['node_features'].shape[0]
            if node_num > self.node_num:
                continue
            if max(self.batch_node_num, node_num) * (len(self.node_features) + 1) > node_num_cap:
                node_features = np.zeros([len(self.node_features), self.batch_node_num, 1108])
                edge_features = np.zeros([len(self.edge_features), self.batch_node_num, self.batch_node_num, 1216])
                adj_mat = np.zeros([len(self.adj_mat), self.batch_node_num, self.batch_node_num])
                gt_strength_level = np.zeros([len(self.gt_strength_level), self.batch_node_num, self.batch_node_num])
                gt_action_labels = np.zeros([len(self.gt_action_labels), self.batch_node_num, self.batch_node_num, len(action_classes)])
                gt_action_roles = np.zeros([len(self.gt_action_roles), self.batch_node_num, self.batch_node_num, len(roles)])

                for i_file in range(len(self.node_features)):
                    node_num = len(self.node_features[i_file])
                    node_features[i_file, :node_num, :] = self.node_features[i_file]
                    edge_features[i_file, :node_num, :node_num, :] = self.edge_features[i_file]
                    adj_mat[i_file, :node_num, :node_num] = self.adj_mat[i_file]
                    gt_strength_level[i_file, :node_num, :node_num] = self.gt_strength_level[i_file]
                    gt_action_labels[i_file, :node_num, :node_num, 1:] = self.gt_action_labels[i_file]
                    gt_action_roles[i_file, :node_num, :node_num, 1:] = self.gt_action_roles[i_file]
                    gt_action_labels[i_file, :node_num, :node_num, 0] = (np.sum(self.gt_action_labels[i_file][:, :, 1:]) == 0).astype(float)
                    gt_action_roles[i_file, :node_num, :node_num, 0] = (np.sum(self.gt_action_roles[i_file][:, :, 1:]) == 0).astype(float)
                
                self.empty_count.acquire()
                self.data_queue.append((
                    node_features, 
                    edge_features, 
                    adj_mat, 
                    gt_action_labels, 
                    gt_action_roles, 
                    gt_strength_level, 
                    copy.deepcopy(self.part_human_ids), 
                    self.batch_node_num, copy.deepcopy(self.data_fn)))
                self.fill_count.release()

                self.node_features = []
                self.edge_features = []
                self.adj_mat = []
                self.gt_action_labels = []
                self.gt_action_roles = []
                self.gt_strength_level = []
                self.part_human_ids = []
                self.batch_node_num = -1
                self.data_fn = []

            self.node_features.append(data['node_features'])# [i_file, :node_num, :] = data['node_features']
            self.edge_features.append(data['edge_features'])# [i_file, :node_num, :node_num, :] = data['edge_features']
            self.adj_mat.append(data['adj_mat'])# [i_file, :node_num, :node_num] = data['adj_mat']
            self.gt_strength_level.append(data['strength_level'])# [i_file, :node_num, :node_num] = data['strength_level']
            self.gt_action_labels.append(data['action_labels'])# [i_file, :node_num, :node_num, 1:] = data['action_labels']
            self.gt_action_roles.append(data['action_roles'])# [i_file, :node_num, :node_num, 1:] = data['action_roles']
            self.part_human_ids.append(data['part_human_id'])
            self.batch_node_num = max(self.batch_node_num, data['node_features'].shape[0])
            self.data_fn.append(filename)

        self.empty_count.acquire()
        self.data_queue.append(None)
        self.fill_count.release()

class DataLoader:
    def __init__(self, imageset, batchsize, node_num, datadir=os.path.join(os.path.dirname(__file__), '../../data/feature_resnet_tengyu'), with_name=False):
        self.imageset = imageset
        self.batchsize = batchsize
        self.datadir = datadir
        self.__next = None
        self.node_num = node_num

        self.filenames = [os.path.join(self.datadir, x) for x in os.listdir(self.datadir) if self.imageset in x]

        self.filenames_backup = copy.deepcopy(self.filenames)
        self.thread = None
        self.with_name = with_name

        pass

    def __len__(self):
        return len(self.filenames_backup)

    def shuffle(self):
        self.filenames = copy.deepcopy(self.filenames_backup)
        random.shuffle(self.filenames)
    
    def prefetch(self):
        if self.thread is not None:
            self.thread.join()
        self.thread = DataThread(self.filenames, self.node_num, self.with_name)
        self.thread.start()
        
    def fetch(self):
        self.thread.fill_count.acquire()
        res = self.thread.data_queue.pop(0)
        self.thread.empty_count.release()
        return res

if __name__ == "__main__":
    import time

    dl_train = DataLoader('train', 1)
    print(dl_train.node_num)
    del dl_train
    dl_val = DataLoader('val', 1)
    print(dl_val.node_num)
    del dl_val
    dl_test = DataLoader('test', 1)
    print(dl_test.node_num)
    del dl_test
