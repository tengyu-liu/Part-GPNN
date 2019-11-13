import re
import os
import pickle
import numpy as np

from metadata import hico_classes, action_classes

f = open('config/hico_hoi_list.txt')
lines = list(f.readlines())
f.close()

pair = np.zeros([81, 117])

for l in lines:
    if len(l.strip()) == 0:
        continue
    l = re.sub(" +", ' ', l).split(' ')
    obj_id = hico_classes.index(l[1].strip())
    action_id = action_classes.index(l[2].strip())
    pair[obj_id, action_id] = 1

print(pair)
pickle.dump(pair, open('obj_action_pairs.pkl', 'wb'))
