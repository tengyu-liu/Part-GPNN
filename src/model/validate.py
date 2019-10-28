import os
import random
import sys
import time
import datetime

import numpy as np
import tensorflow as tf

from config import flags
from dataloader import DataLoader
from metrics import compute_mAP
from model import Model

os.makedirs(os.path.join(os.path.dirname(__file__), 'pred'), exist_ok=True)
pred_dir = os.path.join(os.path.dirname(__file__), 'pred', flags.name)

val_loader = DataLoader('val', flags.batch_size, flags.node_num, with_name=True)

model = Model(flags)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

avg_prec_sum, avg_prec_max, avg_prec_mean, losses, batch_time, data_time = [], [], [], [], [], []
val_loader.prefetch()
item = 0
total_data_time = 0
total_tf_time = 0
while True:
    t0 = time.time()
    res = val_loader.fetch()
    if res is None:
        break
    node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, part_human_ids, batch_node_num, fn = res
    total_data_time += (time.time() - t0)
    item += len(node_features)
    
    tf_t0 = time.time()
    step, pred, loss = sess.run(fetches=[
        model.step, 
        model.edge_label_pred, 
        model.loss], feed_dict={
        model.node_features : node_features,
        model.edge_features : edge_features, 
        model.adj_mat       : adj_mat, 
        model.pairwise_label_gt : gt_action_labels, 
        model.gt_strength_level : gt_strength_level,
        model.batch_node_num : batch_node_num
    })
    tf_t1 = time.time()
    total_tf_time = (tf_t1 - tf_t0)

    for i_item in range(len(node_features)):
        _sum, _max, _mean = compute_mAP(pred[i_item], gt_action_labels[i_item], part_human_ids[i_item], batch_node_num)
        avg_prec_sum.append(_sum)
        avg_prec_max.append(_max)
        avg_prec_mean.append(_mean)

    losses.append(loss)
    batch_time.append(time.time() - t0)
    data_time.append(batch_time[-1] - (tf_t1 - tf_t0))

    print('[Validation] [%d/%d] Loss: %.4f(%.4f) mAP(SUM): %.4f(%.4f) mAP(MAX): %.4f(%.4f) mAP(MEAN): %.4f(%.4f) time: %.4f avg.data.time: (%.4f) avg.tf.time: (%.4f)'%(
        item, len(val_loader), loss, np.mean(losses), 
        np.mean(avg_prec_sum[-flags.batch_size:]), np.mean(avg_prec_sum), 
        np.mean(avg_prec_max[-flags.batch_size:]), np.mean(avg_prec_max), 
        np.mean(avg_prec_mean[-flags.batch_size:]), np.mean(avg_prec_mean), 
        batch_time[-1], total_data_time / item, total_tf_time / item
    ))

    for i in range(len(fn)):
        np.save(os.path.join(pred_dir, '%s.pred.npy'%fn[i]), pred[i])

avg_prec_sum, avg_prec_max, avg_prec_mean, losses = map(np.mean, [avg_prec_sum, avg_prec_max, avg_prec_mean, losses])
