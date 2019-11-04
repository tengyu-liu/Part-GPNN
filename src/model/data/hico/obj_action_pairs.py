import os 
import pickle
import numpy as np

basedir = os.path.join(os.path.dirname(__file__), '../../../../data/hico/feature')

def process(fn, q):
    basedir = os.path.join(os.path.dirname(__file__), '../../../../data/hico/feature')
    item = pickle.load(open(os.path.join(basedir, fn), 'rb'))
    obj_action_pair = np.zeros([81, 117])

    assert np.sum(item['action_labels'][...,117:]) == 0

    for i_node in range(item['part_num'], item['node_num']):
        i_obj = i_node - item['part_num']
        obj_cls = item['obj_classes'][i_obj]
        obj_action_pair[obj_cls] += np.sum(item['action_labels'][:, i_node, :117], axis=0)
    q.put(obj_action_pair)
    print(fn)

from multiprocessing as mp
from queue import Queue

queue = Queue()

p = mp.Pool(32)
p.map(process, [(x, queue) for x in os.listdir(basedir)])

obj_action_pair = np.zeros([81, 117])

while not queue.empty():
    obj_action_pair += queue.get()
    queue.task_done()

obj_action_pair = np.min(obj_action_pair, 1)

pickle.dump(obj_action_pair, open('obj_action_pairs.pkl', 'wb'))
print(obj_action_pair)