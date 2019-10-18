import os
import threading

import pickle
import numpy as np

action_classes = ['none', 'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']
roles = ['none', 'obj', 'instr']

class DataThread(threading.Thread):
    def __init__(self, filenames, node_num):
        t0 = time.time()
        self.filenames = filenames
        self.node_num = node_num
        self.node_features = np.zeros([len(filenames), self.node_num, 1108])
        self.edge_features = np.zeros([len(filenames), self.node_num, self.node_num, 1216])
        self.adj_mat = np.zeros([len(filenames), self.node_num, self.node_num])
        self.gt_strength_level = np.zeros([len(filenames), self.node_num, self.node_num])
        self.gt_action_labels = np.zeros([len(filenames), self.node_num, self.node_num, len(action_classes) - 1])
        self.gt_action_roles = np.zeros([len(filenames), self.node_num, self.node_num, len(roles) - 1])
        super(DataThread, self).__init__()
        print('Const: ', time.time() - t0)
    
    def run(self):
        t0 = time.time()
        for i_file, filename in enumerate(self.filenames):
            data = pickle.load(open(filename, 'rb'))
            node_num = data['node_features'].shape[0]

            self.node_features[i_file, :node_num, :] = data['node_features']
            self.edge_features[i_file, :node_num, :node_num, :] = data['edge_features']
            self.adj_mat[i_file, :node_num, :node_num] = data['adj_mat']
            self.gt_strength_level[i_file, :node_num, :node_num] = data['strength_level']
            self.gt_action_labels[i_file, :node_num, :node_num, :] = data['action_labels']
            self.gt_action_roles[i_file, :node_num, :node_num, :] = data['action_roles']
        print('Run: ', time.time() - t0)

class DataLoader:
    def __init__(self, imageset, batchsize, j=4, datadir=os.path.join(os.path.dirname(__file__), '../../data/feature_resnet_tengyu')):
        self.imageset = imageset
        self.batchsize = batchsize
        self.datadir = datadir
        self.filenames = [x for x in os.listdir(self.datadir) if self.imageset in x]
        self.__next = None
        self.shuffled_idx = np.arange(len(self.filenames))

        self.node_num = 0

        # TODO: Replace with real max node num
        for fn in self.filenames:
            self.node_num = max(self.node_num, pickle.load(open(os.path.join(self.datadir, fn), 'rb'))['node_features'].shape[0])

        self.thread = None
        pass

    def __len__(self):
        return len(self.filenames)

    def shuffle(self):
        self.shuffled_idx = np.random.permutation(len(self.filenames))
    
    def prefetch(self, batch_id):
        filenames = [os.path.join(self.datadir, self.filenames[idx]) for idx in self.shuffled_idx[self.batchsize * batch_id : self.batchsize * (batch_id + 1)]]
        self.thread = DataThread(filenames, self.node_num)
        self.thread.start()
        
    def fetch(self):
        self.thread.join()
        return self.thread.node_features, self.thread.edge_features, self.thread.adj_mat, self.thread.gt_action_labels, self.thread.gt_action_roles, self.thread.gt_strength_level

if __name__ == "__main__":
    import time

    dl = DataLoader('train', 32)
    for i in range(10):
        t0 = time.time()
        dl.prefetch(0)
        _, _, _, _, _, _ = dl.fetch()
        t1 = time.time()
        print('Total: ', t1-t0)
    
