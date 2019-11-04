import datetime
import os
import pickle
import random
import sys
import time

import numpy as np
import tensorflow as tf

from config import flags
from dataloader_parallel import DataLoader
from metrics import compute_mAP
from model import Model

random.seed(0)
np.random.seed(0)
tf.random.set_random_seed(0)

obj_action_pair = pickle.load(open(os.path.join(os.path.dirname(__file__), 'data', 'obj_action_pairs.pkl'), 'rb'))

train_loader = DataLoader('train', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs)
val_loader = DataLoader('val', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs)
test_loader = DataLoader('test', flags.node_num, negative_suppression=flags.negative_suppression, n_jobs=flags.n_jobs)

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
    avg_prec_sum, avg_prec_max, avg_prec_mean, losses, batch_time, data_time = [], [], [], [], [], []
    train_loader.shuffle()
    train_loader.prefetch()
    item = 0
    total_data_time = 0
    total_tf_time = 0

    while True:
        t0 = time.time()

        res = train_loader.fetch()
        if res is None:
            break
        node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, part_human_ids, pairwise_label_mask, batch_node_num, fns = res
        total_data_time += (time.time() - t0)
        item += len(node_features)
        
        tf_t0 = time.time()
        step, pred, loss, _ = sess.run(fetches=[
            model.step, 
            model.edge_label_pred, 
            model.loss, 
            model.train_op], feed_dict={
            model.node_features : node_features,
            model.edge_features : edge_features, 
            model.adj_mat       : adj_mat, 
            model.pairwise_label_gt : gt_action_labels, 
            model.gt_strength_level : gt_strength_level,
            model.batch_node_num : batch_node_num,
            model.pairwise_label_mask : pairwise_label_mask, 
            model.training: True
        })
        tf_t1 = time.time()
        total_tf_time += (tf_t1 - tf_t0)

        for i_item in range(len(node_features)):
            _sum, _max, _mean = compute_mAP(pred[i_item], gt_action_labels[i_item], part_human_ids[i_item], batch_node_num)
            avg_prec_sum.append(_sum)
            avg_prec_max.append(_max)
            avg_prec_mean.append(_mean)

        losses.append(loss)
        batch_time.append(time.time() - t0)
        data_time.append(batch_time[-1] - (tf_t1 - tf_t0))

        print('\r[Train %d] [%d/%d] Loss: %.4f(%.4f) mAP(SUM): %.4f(%.4f) mAP(MAX): %.4f(%.4f) mAP(MEAN): %.4f(%.4f) time: %.4f avg.data.time: (%.4f) avg.tf.time: (%.4f)'%(
            epoch, item, len(train_loader), loss, np.mean(losses), 
            np.mean(avg_prec_sum[-flags.batch_size:]), np.mean(avg_prec_sum), 
            np.mean(avg_prec_max[-flags.batch_size:]), np.mean(avg_prec_max), 
            np.mean(avg_prec_mean[-flags.batch_size:]), np.mean(avg_prec_mean), 
            batch_time[-1], total_data_time / item, total_tf_time / item
        ), end='', flush=True)

    avg_prec_sum, avg_prec_max, avg_prec_mean, losses = map(np.mean, [avg_prec_sum, avg_prec_max, avg_prec_mean, losses])

    summ = sess.run(fetches=model.summ, feed_dict={
        model.summ_loss_in: loss, 
        model.summ_map_sum_in : avg_prec_sum,
        model.summ_map_max_in : avg_prec_max,
        model.summ_map_mean_in : avg_prec_mean
    })
 
    train_writer.add_summary(summ, global_step=epoch)

    print('\r======== [Train %d] Loss: %.4f mAP(SUM) %.4f mAP(MAX): %.4f mAP(MEAN): %.4f ========'% (
        epoch, np.mean(losses), np.mean(avg_prec_sum), np.mean(avg_prec_max), np.mean(avg_prec_mean)
    ))

    if not flags.debug:
        # Validate
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
            node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, part_human_ids, pairwise_label_mask, batch_node_num, fns = res
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
                model.batch_node_num : batch_node_num,
                model.pairwise_label_mask : pairwise_label_mask,
                model.training: False
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

            print('\r[Val %d] [%d/%d] Loss: %.4f(%.4f) mAP(SUM): %.4f(%.4f) mAP(MAX): %.4f(%.4f) mAP(MEAN): %.4f(%.4f) time: %.4f avg.data.time: (%.4f) avg.tf.time: (%.4f)'%(
                epoch, item, len(val_loader), loss, np.mean(losses), 
                np.mean(avg_prec_sum[-flags.batch_size:]), np.mean(avg_prec_sum), 
                np.mean(avg_prec_max[-flags.batch_size:]), np.mean(avg_prec_max), 
                np.mean(avg_prec_mean[-flags.batch_size:]), np.mean(avg_prec_mean), 
                batch_time[-1], total_data_time / item, total_tf_time / item
            ), end='', flush=True)

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

        print('\r======== [Validation %d] Loss: %.4f mAP(SUM) %.4f mAP(MAX): %.4f mAP(MEAN): %.4f ========'% (
            epoch, losses, avg_prec_sum, avg_prec_max, avg_prec_mean
        ))


    if epoch % 5 == 4:
        # Test
        avg_prec_sum, avg_prec_max, avg_prec_mean, losses, batch_time, data_time = [], [], [], [], [], []
        test_loader.prefetch()
        item = 0
        total_data_time = 0
        total_tf_time = 0
        while True:
            t0 = time.time()
            res = test_loader.fetch()
            if res is None:
                break
            node_features, edge_features, adj_mat, gt_action_labels, gt_action_roles, gt_strength_level, part_human_ids, pairwise_label_mask, batch_node_num, fns = res
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
                model.batch_node_num : batch_node_num,
                model.pairwise_label_mask : pairwise_label_mask,
                model.training: False
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

            print('\r[TEST] [%d/%d] Loss: %.4f(%.4f) mAP(SUM): %.4f(%.4f) mAP(MAX): %.4f(%.4f) mAP(MEAN): %.4f(%.4f) time: %.4f avg.data.time: (%.4f) avg.tf.time: (%.4f)'%(
                item, len(test_loader), loss, np.mean(losses), 
                np.mean(avg_prec_sum[-flags.batch_size:]), np.mean(avg_prec_sum), 
                np.mean(avg_prec_max[-flags.batch_size:]), np.mean(avg_prec_max), 
                np.mean(avg_prec_mean[-flags.batch_size:]), np.mean(avg_prec_mean), 
                batch_time[-1], total_data_time / item, total_tf_time / item
            ), end='')

        avg_prec_sum, avg_prec_max, avg_prec_mean, losses = map(np.mean, [avg_prec_sum, avg_prec_max, avg_prec_mean, losses])
        print('\r======== [Test %d] Loss: %.4f mAP(SUM) %.4f mAP(MAX): %.4f mAP(MEAN): %.4f ========'% (
            epoch, losses, avg_prec_sum, avg_prec_max, avg_prec_mean
        ))

        f = open('validate.txt', 'a')
        f.write('%s/%d: %f, %f, %f, %f\n'%(flags.name, epoch, avg_prec_sum, avg_prec_max, avg_prec_mean, losses))
        f.close()
