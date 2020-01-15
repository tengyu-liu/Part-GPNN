import vsrl_eval
import pickle
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import os
import random

vcoco_root = '/home/tengyu/Data/mscoco/v-coco'
mscoco_root = '/home/tengyu/Data/mscoco/coco/train2014'
def get_vcocoeval(imageset):
    return vsrl_eval.VCOCOeval(os.path.join(vcoco_root, 'data/vcoco/vcoco_{}.json'.format(imageset)),
                               os.path.join(vcoco_root, 'data/instances_vcoco_all_2014.json'),
                               os.path.join(vcoco_root, 'data/splits/vcoco_{}.ids'.format(imageset)))

def get_overlap(boxes, ref_box):
  ixmin = np.maximum(boxes[:, 0], ref_box[0])
  iymin = np.maximum(boxes[:, 1], ref_box[1])
  ixmax = np.minimum(boxes[:, 2], ref_box[2])
  iymax = np.minimum(boxes[:, 3], ref_box[3])
  iw = np.maximum(ixmax - ixmin + 1., 0.)
  ih = np.maximum(iymax - iymin + 1., 0.)
  inters = iw * ih
  # union
  uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
         (boxes[:, 2] - boxes[:, 0] + 1.) *
         (boxes[:, 3] - boxes[:, 1] + 1.) - inters)
  overlaps = inters / uni
  return overlaps

# def vcoco_evaluation(vcocoeval, imageset, all_results, name, method):
#     print('\n%s: '%method, end='')
#     det_file = os.path.join(os.path.dirname(__file__), 'eval', name, '%s_detections[%s].pkl'%(imageset, method))
#     pickle.dump(all_results, open(det_file, 'wb'))
#     vcocoeval._do_eval(det_file, ovr_thresh=0.5)
#     print()

train_vcocoeval = get_vcocoeval('train')
val_vcocoeval = get_vcocoeval('val')
test_vcocoeval = get_vcocoeval('test')

vcocodb = train_vcocoeval._get_vcocodb()

i = 4
gt_item = vcocodb[i]
image_id = gt_item['id']
my_item = pickle.load(open('../../data/feature_resnet_tengyu2/COCO_train2014_%012d.jpg.data'%image_id, 'rb'))

image = skimage.io.imread(os.path.join(mscoco_root, 'COCO_train2014_%012d.jpg'%image_id))

"Compare gt and my human boxes"
_ = plt.subplot(121)
_ = plt.imshow(image)
for i in range(len(gt_item['boxes'])):
    if gt_item['gt_classes'][i] == 1:
        x0,y0,x1,y1 = gt_item['boxes'][i]
        _ = plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], linewidth=3, c='red')

_ = plt.subplot(122)
_ = plt.imshow(image)
for i in range(len(my_item['part_boxes'])):
    if my_item['part_classes'][i] == 18 or True:
        x0,y0,x1,y1 = my_item['part_boxes'][i]
        _ = plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], c='red')

_ = plt.show()

# TODO: replace full body box by human detection bbox