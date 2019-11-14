"""
Prepare data for Part-GPNN model. 
Need: 
Node feature at different scales
Edge feature for valid edges
Adjacency matrix GT (parse graph GT)
Edge weight (corresponds to node level)
Edge label GT
"""

import os
import pickle

import vsrl_utils as vu

vcoco_root = '/mnt/hdd-12t/share/v-coco/data'
save_data_path = os.path.join(os.path.dirname(__file__), '../../../data/feature_resnet_tengyu')

for imageset in ['train', 'test', 'val']:
    coco = vu.load_coco(vcoco_root)
    vcoco_all = vu.load_vcoco('vcoco_{}'.format(imageset), vcoco_root)
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)
    image_ids = vcoco_all[0]['image_id'][:, 0].astype(int).tolist()

    total = len(image_ids)
    success = 0
    miss = 0

    for i_image, image_id in enumerate(image_ids):
        try:
            filename = coco.loadImgs(ids=[image_id])[0]['file_name']
            d = filename.split('_')[1][:-4]
            data = pickle.load(open(os.path.join(save_data_path, filename + '.data'), 'rb'))
            data['img_id'] = image_id
            pickle.dump(data, open(os.path.join(save_data_path, filename + '.data'), 'wb'))
            success += 1
        except:
            miss += 1
        print('\rSuccess: %d Miss: %d Total: %d/%d'%(success, miss, success+miss, total), end='', flush=True)