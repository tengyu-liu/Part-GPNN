import os
import pickle

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

run_name = 'optimized-bs1'
result_dir = os.path.join(os.path.dirname(__file__), '../../tmp/evaluation/vcoco/', run_name)

img_dir = '/home/tengyu/Data/mscoco/coco/'
gt_dir = '/home/tengyu/Data/mscoco/v-coco/processed/resnet/'

def plot_box(box, c='red'):
    plt.plot([box[0], box[2], box[2], box[0], box[0]], [box[1], box[1], box[3], box[3], box[1]], c=c)

def sigmoid(x):
  return 1/(1+np.exp(-x))

action_classes = ['none', 'hold', 'stand', 'sit', 'ride', 'walk', 'look', 'hit', 'eat', 'jump', 'lay', 'talk_on_phone', 'carry',
                  'throw', 'catch', 'cut', 'run', 'work_on_computer', 'ski', 'surf', 'skateboard', 'smile', 'drink',
                  'kick', 'point', 'read', 'snowboard']

for fn in os.listdir(result_dir):
    try:
        image_name = fn[:-5]
        imageset = image_name.split('_')[1]

        img = scipy.misc.imread(os.path.join(img_dir, imageset, image_name))

        pred = pickle.load(open(os.path.join(result_dir, fn), 'rb'))
        gt = pickle.load(open(os.path.join(gt_dir, image_name + '.p'), 'rb'))

        pred_adj_mat = pred['adj_mat'].detach().cpu().numpy()
        node_labels = pred['node_labels'].detach().cpu().numpy()
        node_roles = pred['node_roles'].detach().cpu().numpy()
        part_num = gt['part_num']
        obj_num = gt['obj_num']
        part_boxes = gt['part_boxes']
        obj_boxes = gt['obj_boxes']

        # plt.imshow(img)

        pred_adj_mat = sigmoid(pred_adj_mat)

        # print(node_roles.shape, node_roles)

        largest = None
        largest_part = None
        largest_obj = None
        largest_label = None

        indices = np.array([[[i,j] for i in range(part_num)] for j in range(obj_num)]).reshape([-1,2])
        indices = sorted(indices, key=lambda x:pred_adj_mat[x[0], x[1] + part_num], reverse=True)

        count = 0
        for i, j in indices:
            if np.argmax(node_roles[j+part_num]) != 0:
                plt.imshow(img)
                plot_box(part_boxes[i], 'blue')
                plot_box(obj_boxes[j], 'red')
                plt.plot([(part_boxes[i][0] + part_boxes[i][2])/2, (obj_boxes[j][0] + obj_boxes[j][2])/2], [(part_boxes[i][1] + part_boxes[i][3])/2, (obj_boxes[j][1] + obj_boxes[j][3])/2], c='green')
                # plt.text(part_boxes[i][0] + 6, part_boxes[i][1] - 11, action_classes[np.argmax(node_labels[i])], color='white', bbox=dict(fc='blue',))
                print(action_classes[np.argmax(node_labels[i])])
                plt.show()
            count += 1
            if count == 3:
                break
    except:
        # raise
        continue