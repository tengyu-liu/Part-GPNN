import os
import threading

import pickle
import numpy as np

action_classes = ['none', 'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']
roles = ['none', 'obj', 'instr']

class DataThread(threading.Thread):
    def __init__(self, filename, node_num):
        self.filename = filename
        self.node_num = node_num
        self.node_features = np.zeros([self.node_num, 1108])
        self.edge_features = np.zeros([self.node_num, self.node_num, 1216])
        self.adj_mat = np.zeros([self.node_num, self.node_num])
        self.gt_strength_level = np.zeros([self.node_num, self.node_num])
        self.gt_action_labels = np.zeros([self.node_num, self.node_num, len(action_classes) - 1])
        self.gt_action_roles = np.zeros([self.node_num, self.node_num, len(roles) - 1])
        super(DataThread, self).__init__()
    
    def run(self):
        data = pickle.load(open(self.filename, 'rb'))
        node_num = data['node_features'].shape[0]

        self.node_features[:node_num, :] = data['node_features']
        self.edge_features[:node_num, :node_num, :] = data['edge_features']
        self.adj_mat[:node_num, :node_num] = data['adj_mat']
        self.gt_strength_level[:node_num, :node_num] = data['strength_level']
        self.gt_action_labels[:node_num, :node_num, :] = data['action_labels']
        self.gt_action_roles[:node_num, :node_num, :] = data['action_roles']

class DataLoader:
    def __init__(self, imageset, batchsize, j=4, datadir=os.path.join(os.path.dirname(__file__), '../../data/feature_resnet_tengyu')):
        self.imageset = imageset
        self.batchsize = batchsize
        self.datadir = datadir
        self.filenames = [x for x in os.listdir(self.datadir) if self.imageset in x]
        self.__next = None
        self.shuffled_idx = np.arange(len(self.filenames))

        self.node_num = 0
        # Compute maximum node num
        for fn in self.filenames:
            self.node_num = max(self.node_num, pickle.load(open(os.path.join(self.datadir, fn), 'rb'))['node_features'].shape[0])

        self.threads = []
        pass

    def __len__(self):
        return len(self.filenames)

    def shuffle(self):
        self.shuffled_idx = np.random.permutation(len(self.filenames))
    
    def prefetch(self, batch_id):
        # # TODO: Load data in parallel for Nx speedup (single thread loading takes ~1.22 sec)
        # node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level = [], [], [], [], [], []
        # node_features = np.zeros([self.batchsize, self.node_num, 1108])
        # edge_features = np.zeros([self.batchsize, self.node_num, self.node_num, 1216])
        # adj_mat = np.zeros([self.batchsize, self.node_num, self.node_num])
        # gt_strength_level = np.zeros([self.batchsize, self.node_num, self.node_num])
        # gt_action_labels = np.zeros([self.batchsize, self.node_num, self.node_num, len(action_classes) - 1])
        # gt_action_roles = np.zeros([self.batchsize, self.node_num, self.node_num, len(roles) - 1])
        # 
        # for i_item, filename_idx in enumerate(self.shuffled_idx[self.batchsize * batch_id : self.batchsize * (batch_id + 1)]):
        #     data = pickle.load(open(os.path.join(self.datadir, self.filenames[filename_idx]), 'rb'))
        #     node_num = data['node_features'].shape[0]

        #     node_features[i_item, :node_num, :] = data['node_features']
        #     edge_features[i_item, :node_num, :node_num, :] = data['edge_features']
        #     adj_mat[i_item, :node_num, :node_num] = data['adj_mat']
        #     gt_strength_level[i_item, :node_num, :node_num] = data['strength_level']
        #     gt_action_labels[i_item, :node_num, :node_num, :] = data['action_labels']
        #     gt_action_roles[i_item, :node_num, :node_num, :] = data['action_roles']
        for idx in self.shuffled_idx[self.batchsize * batch_id : self.batchsize * (batch_id + 1)]:
            t = DataThread(os.path.join(self.datadir, self.filenames[idx]), self.node_num)
            t.start()
            self.threads.append(t)
        
    def fetch(self):
        node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level = [], [], [], [], [], []
        while len(self.threads) > 0:
            t = self.threads.pop()
            t.join()
            node_features.append(t.node_features)
            edge_features.append(t.edge_features)
            adj_mat.append(t.adj_mat)
            gt_action_labels.append(t.gt_action_labels)
            gt_action_roles.append(t.gt_action_roles)
            gt_strength_level.append(t.gt_strength_level)
        node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level = map(np.array, [node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level])
        return node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level

if __name__ == "__main__":
    import time

    dl = DataLoader('train', 32)
    for i in range(10):
        t0 = time.time()
        dl.prefetch(0)
        _, _, _, _, _, _ = dl.fetch()
        t1 = time.time()
        print(i, t1-t0)
    
