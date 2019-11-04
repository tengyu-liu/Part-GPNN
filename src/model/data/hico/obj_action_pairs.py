import os 
import pickle
import numpy as np

basedir = os.path.join(os.path.dirname(__file__), '../../../../data/hico/feature')

obj_action_pair = np.zeros([81, 117])

total = len(os.listdir(basedir))
count = 0

for fn in os.listdir(basedir):
    item = pickle.load(open(os.path.join(basedir, fn), 'rb'))
    assert np.sum(item['action_labels'][...,117:]) == 0

    for i_node in range(item['part_num'], item['node_num']):
        i_obj = i_node - item['part_num']
        obj_cls = item['obj_classes'][i_obj]
        obj_action_pair[obj_cls] += np.reduce_sum(item['action_labels'][:, i_node, :117], axis=0)

    count += 1
    print('\r%d/%d %s'%(count, total, fn), end='', flush=True)

obj_action_pair = np.min(obj_action_pair, 1)

pickle.dump(obj_action_pair, open('obj_action_pairs.pkl', 'wb'))
