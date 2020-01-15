import os
import pickle
import numpy as np

base_dir = os.path.join(os.path.dirname(__file__), '../../../data/feature_resnet_tengyu2')

obj_action_pair = pickle.load(open(os.path.join(os.path.dirname(__file__), 'obj_action_pairs.pkl'), 'rb'))

total = len(os.listdir(base_dir))
count = 0

for fn in os.listdir(base_dir):
    try:
        data = pickle.load(open(os.path.join(base_dir, fn), 'rb'))
    except:
        print(fn)
        continue

    pairwise_action_mask = np.zeros([data['node_num'], data['node_num'], 27])
    for i_obj in range(data['part_num'], data['node_num']):
        pairwise_action_mask[:, i_obj, :] = obj_action_pair[[data['obj_classes'][i_obj - data['part_num']]]]
        pairwise_action_mask[i_obj, :, :] = obj_action_pair[[data['obj_classes'][i_obj - data['part_num']]]]
    for i_part in range(part_num):
        for j_part in range(part_num):
            if part_human_ids[i_part] == part_human_ids[j_part]:
                pairwise_action_mask[i_part, j_part, [2,5,16,21,24]] = 1.0

    data['pairwise_action_mask'] = pairwise_action_mask
    pickle.dump(data, open(os.path.join(base_dir, fn), 'wb'))

    count += 1
    print('\r%d/%d'%(count, total), end='', flush=True)
