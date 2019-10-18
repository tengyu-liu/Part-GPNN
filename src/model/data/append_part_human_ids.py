import os
import pickle

import numpy as np

datadir = '/home/tengyu/Documents/github/Part-GPNN/data/feature_resnet_tengyu'
for fn in os.listdir(datadir):
    data = pickle.load(open(os.path.join(datadir, fn), 'rb'))
    if 'part_human_id' in data:
        continue
    
    adj_mat = data['adj_mat']
    