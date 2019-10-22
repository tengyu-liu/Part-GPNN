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

train_loader = DataLoader('train', flags.batch_size, flags.node_num)
val_loader = DataLoader('val', flags.batch_size, flags.node_num)
test_loader = DataLoader('test', flags.batch_size, flags.node_num)
train_loader.shuffle()
train_loader.prefetch(0)

model = Model(flags)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'logs', flags.name), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'models', flags.name), exist_ok=True)
log_dir = os.path.join(os.path.dirname(__file__), 'logs', flags.name)
model_dir = os.path.join(os.path.dirname(__file__), 'models', flags.name)
fig_dir = os.path.join(os.path.dirname(__file__), 'figs', flags.name)

# logger and saver
train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
val_writer = tf.summary.FileWriter(os.path.join(log_dir, 'val'), sess.graph)
saver = tf.train.Saver(max_to_keep=0)
if flags.restore_epoch >= 0:
    saver.restore(sess, os.path.join(model_dir, '%04d.ckpt'%(flags.name, flags.restore_epoch)))

for epoch in range(flags.epochs):
    # Train
    avg_prec_sum, avg_prec_max, avg_prec_mean, losses, batch_time = [], [], [], [], []
    for batch_id in range(len(train_loader)):
        t0 = time.time()
        node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, part_human_ids = train_loader.fetch()
        if batch_id == len(train_loader) - 1:
            val_loader.prefetch(0)
        else:
            train_loader.prefetch(batch_id + 1)
        
        step, pred, loss, _ = sess.run(fetches=[
            model.step, 
            model.edge_label_pred, 
            model.loss, 
            model.train_op], feed_dict={
            model.node_features : node_features,
            model.edge_features : edge_features, 
            model.adj_mat       : adj_mat, 
            model.pairwise_label_gt : gt_action_labels, 
            model.gt_strength_level : gt_strength_level
        })

        for i_item in flags.batch_size:
            _sum, _max, _mean = compute_mAP(pred, gt_action_labels, part_human_ids)
            avg_prec_sum.append(_sum)
            avg_prec_max.append(_max)
            avg_prec_mean.append(_mean)

        losses.append(loss)
        batch_time.append(time.time() - t0)

        if batch_id % flags.log_interval or batch_id == len(train_loader) - 1:
            print('[Train %d] [%d/%d] Loss: %.4f(%.4f) mAP(SUM): %.4f(%.4f) mAP(MAX): %.4f(%.4f) mAP(MEAN): %.4f(%.4f) Time: %.4f(%.4f)'%(
                epoch, batch_id, len(train_loader), loss, np.mean(losses), 
                np.mean(avg_prec_sum[-flags.batch_size:]), np.mean(avg_prec_sum), 
                np.mean(avg_prec_max[-flags.batch_size:]), np.mean(avg_prec_max), 
                np.mean(avg_prec_mean[-flags.batch_size:]), np.mean(avg_prec_mean), 
                batch_time, np.mean(batch_time)
            ))

    avg_prec_sum, avg_prec_max, avg_prec_mean, losses = map(np.mean, [avg_prec_sum, avg_prec_max, avg_prec_mean, losses])

    summ = sess.run(fetches=model.summ, feed_dict={
        model.summ_loss_in: loss, 
        model.summ_map_sum_in : avg_prec_sum,
        model.summ_map_max_in : avg_prec_max,
        model.summ_map_mean_in : avg_prec_mean
    })
 
    train_writer.add_summary(summ, global_step=epoch)

    # Validate
    avg_prec_sum, avg_prec_max, avg_prec_mean, losses, batch_time = [], [], [], [], []
    for batch_id in range(len(val_loader)):
        node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, part_human_ids = val_loader.fetch()
        if batch_id == len(val_loader) - 1:
            if epoch == flags.epochs - 1:
                test_loader.prefetch(0)
            else:
                train_loader.shuffle()
                train_loader.prefetch(0)
        else:
            val_loader.prefetch(batch_id + 1)
        
        step, pred, loss = sess.run(fetches=[
            model.step, 
            model.edge_label_pred, 
            model.loss], feed_dict={
            model.node_features : node_features,
            model.edge_features : edge_features, 
            model.adj_mat       : adj_mat, 
            model.pairwise_label_gt : gt_action_labels, 
            model.gt_strength_level : gt_strength_level
        })

        for i_item in flags.batch_size:
            _sum, _max, _mean = compute_mAP(pred, gt_action_labels, part_human_ids)
            avg_prec_sum.append(_sum)
            avg_prec_max.append(_max)
            avg_prec_mean.append(_mean)

        losses.append(loss)
        batch_time.append(time.time() - t0)

        if batch_id % flags.log_interval or batch_id == len(val_loader) - 1:
            print('[Val %d] [%d/%d] Loss: %.4f(%.4f) mAP(SUM): %.4f(%.4f) mAP(MAX): %.4f(%.4f) mAP(MEAN): %.4f(%.4f) Time: %.4f(%.4f)'%(
                epoch, batch_id, len(val_loader), loss, np.mean(losses), 
                np.mean(avg_prec_sum[-flags.batch_size:]), np.mean(avg_prec_sum), 
                np.mean(avg_prec_max[-flags.batch_size:]), np.mean(avg_prec_max), 
                np.mean(avg_prec_mean[-flags.batch_size:]), np.mean(avg_prec_mean), 
                batch_time, np.mean(batch_time)
            ))

    avg_prec_sum, avg_prec_max, avg_prec_mean, losses = map(np.mean, [avg_prec_sum, avg_prec_max, avg_prec_mean, losses])

    summ = sess.run(fetches=model.summ, feed_dict={
        model.summ_loss_in: losses, 
        model.summ_map_sum_in : avg_prec_sum,
        model.summ_map_max_in : avg_prec_max,
        model.summ_map_mean_in : avg_prec_mean
    })
 
    val_writer.add_summary(summ, global_step=epoch)
    # Save model
    saver.save(sess, os.path.join(model_dir, '%04d.ckpt'%epoch))

    print('[%s] Epoch %d V.Loss: %f V.mAP: %f, %f, %f'%(datetime.datetime.now(), epoch, losses, avg_prec_sum, avg_prec_max, avg_prec_mean))

# Test
avg_prec_sum, avg_prec_max, avg_prec_mean, losses, batch_time = [], [], [], [], []
for batch_id in range(len(test_loader)):
    node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, part_human_ids = test_loader.fetch()
    if batch_id == len(test_loader) - 1:
        pass
    else:
        val_loader.prefetch(batch_id + 1)
    
    step, pred, loss = sess.run(fetches=[
        model.step, 
        model.edge_label_pred, 
        model.loss], feed_dict={
        model.node_features : node_features,
        model.edge_features : edge_features, 
        model.adj_mat       : adj_mat, 
        model.pairwise_label_gt : gt_action_labels, 
        model.gt_strength_level : gt_strength_level
    })

    for i_item in flags.batch_size:
        _sum, _max, _mean = compute_mAP(pred, gt_action_labels, part_human_ids)
        avg_prec_sum.append(_sum)
        avg_prec_max.append(_max)
        avg_prec_mean.append(_mean)

    losses.append(loss)
    batch_time.append(time.time() - t0)

    if batch_id % flags.log_interval or batch_id == len(test_loader) - 1:
        print('[TEST] [%d/%d] Loss: %.4f(%.4f) mAP(SUM): %.4f(%.4f) mAP(MAX): %.4f(%.4f) mAP(MEAN): %.4f(%.4f) Time: %.4f(%.4f)'%(
            batch_id, len(test_loader), loss, np.mean(losses), 
            np.mean(avg_prec_sum[-flags.batch_size:]), np.mean(avg_prec_sum), 
            np.mean(avg_prec_max[-flags.batch_size:]), np.mean(avg_prec_max), 
            np.mean(avg_prec_mean[-flags.batch_size:]), np.mean(avg_prec_mean), 
            batch_time, np.mean(batch_time)
        ))

avg_prec_sum, avg_prec_max, avg_prec_mean, losses = map(np.mean, [avg_prec_sum, avg_prec_max, avg_prec_mean, losses])
print('Experiment [%s] Result: ')
print('\t Best eval mAP: %f'%best_eval_mAP)
print('\t Final test mAP: %f'%max(avg_prec_sum, avg_prec_max, avg_prec_mean))
