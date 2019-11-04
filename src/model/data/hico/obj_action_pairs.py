import os 
import pickle
import numpy as np

basedir = os.path.join(os.path.dirname(__file__), '../../../../data/hico/feature')

def process(fns, q):
    basedir = os.path.join(os.path.dirname(__file__), '../../../../data/hico/feature')
    
    total = len(fns)
    count = 0

    obj_action_pair = np.zeros([81, 117])

    for fn in fns:
        item = pickle.load(open(os.path.join(basedir, fn), 'rb'))

        assert np.sum(item['action_labels'][...,117:]) == 0

        for i_node in range(item['part_num'], item['node_num']):
            i_obj = i_node - item['part_num']
            obj_cls = item['obj_classes'][i_obj]
            obj_action_pair[obj_cls] += np.sum(item['action_labels'][:, i_node, :117], axis=0)

        count += 1
        print('\r', count, total, fn, end='', flush=True)

    q.put(obj_action_pair)

import threading
from queue import Queue

queue = Queue()

ts = []
for i in range(32):
    t = threading.Thread(target=process, args=(filenames[i], queue))
    t.start()
    ts.append(t)

for i in range(32):
    ts[i].join()

obj_action_pair = np.zeros([81, 117])

while not queue.empty():
    obj_action_pair += queue.get()
    queue.task_done()

obj_action_pair = np.min(obj_action_pair, 1)

pickle.dump(obj_action_pair, open('obj_action_pairs.pkl', 'wb'))
print(obj_action_pair)