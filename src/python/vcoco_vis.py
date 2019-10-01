"""
Created on Feb 25, 2018

@author: Siyuan Qi

Description of the file.

"""

import os
import argparse
import time
import datetime
import pickle

import numpy as np
import torch
import torch.autograd
import sklearn.metrics
import vsrl_eval

import datasets
import units
import models

import config
import logutil
import utils

action_class_num = len(datasets.vcoco_metadata.action_classes)
roles_num = len(datasets.vcoco_metadata.roles)


def evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=False):
    pred_node_labels_prob = torch.nn.Sigmoid()(pred_node_labels)
    np_pred_node_labels = pred_node_labels_prob.data.cpu().numpy()
    np_node_labels = node_labels.data.cpu().numpy()

    new_y_true = np.empty((2 * len(det_indices), action_class_num))
    new_y_score = np.empty((2 * len(det_indices), action_class_num))
    for y_i, (batch_i, i, j, s) in enumerate(det_indices):
        new_y_true[2*y_i, :] = np_node_labels[batch_i, i, :]
        new_y_true[2*y_i+1, :] = np_node_labels[batch_i, j, :]
        new_y_score[2*y_i, :] = np_pred_node_labels[batch_i, i, :]
        new_y_score[2*y_i+1, :] = np_pred_node_labels[batch_i, j, :]

    y_true = np.vstack((y_true, new_y_true))
    y_score = np.vstack((y_score, new_y_score))
    return y_true, y_score

def append_result(all_results, node_labels, node_roles, image_id, obj_boxes, part_boxes, human_boxes, human_num, part_num, obj_num, obj_classes, part_classes, adj_mat, part_human_id):
    metadata = datasets.vcoco_metadata
    for i in range(part_num):
        # if node_labels[i, metadata.action_index['none']] > 0.5:
        #     continue
        instance_result = dict()
        instance_result['image_id'] = image_id
        instance_result['part_box'] = part_boxes[i, :]
        instance_result['part_name'] = part_classes[i]
        instance_result['human_box'] = human_boxes[part_human_id[i]]
        for action_index, action in enumerate(metadata.action_classes):
            if action == 'none':  # or node_labels[i, action_index] < 0.5
                continue
            result = instance_result.copy()
            result['{}_agent'.format(action)] = node_labels[i, action_index]
            if node_labels[i, action_index] < 0.5:
                all_results.append(result)
                continue
            for role in metadata.action_roles[action][1:]:
                role_index = metadata.role_index[role]
                action_role_key = '{}_{}'.format(action, role)
                best_score = -np.inf
                best_j = -1
                for j in range(obj_num):
                    action_role_score = (node_labels[j + part_num, action_index] + node_roles[j + part_num, role_index] + adj_mat[i, j + part_num])/3
                    if action_role_score > best_score:
                        best_score = action_role_score
                        obj_info = np.append(obj_boxes[j, :], action_role_score)
                        best_j = j
                if best_score > 0.0:
                    # obj_info[4] = 1.0
                    result[action_role_key] = obj_info
                    result['{}_class'.format(role)] = obj_classes[best_j]
            all_results.append(result)


def append_results(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, part_human_id, img_ids, obj_boxes, part_boxes, human_boxes, human_nums, part_nums, obj_nums, obj_classes, part_classes, all_results):
    # Normalize label outputs
    pred_node_labels_prob = torch.nn.Sigmoid()(pred_node_labels)
    np_pred_node_labels = pred_node_labels_prob.data.cpu().numpy()

    # Normalize roles outputs
    pred_node_roles_prob = torch.nn.Softmax(dim=-1)(pred_node_roles)
    np_pred_node_roles = pred_node_roles_prob.data.cpu().numpy()

    # Normalize adjacency matrix
    pred_adj_mat_prob = torch.nn.Sigmoid()(pred_adj_mat)
    np_pred_adj_mat = pred_adj_mat_prob.data.cpu().numpy()

    # # TODO: debug metrics
    # np_node_labels = node_labels.data.cpu().numpy()
    # np_node_roles = node_roles.data.cpu().numpy()
    # np_adj_mat = adj_mat.data.cpu().numpy()
    # np_pred_node_labels = np_node_labels
    # np_pred_node_roles = np_node_roles
    # np_pred_adj_mat = np_adj_mat

    for batch_i in range(node_labels.size()[0]):
        append_result(all_results, np_pred_node_labels[batch_i, ...], np_pred_node_roles[batch_i, ...], img_ids[batch_i], obj_boxes[batch_i], part_boxes[batch_i], human_boxes[batch_i], human_nums[batch_i], part_nums[batch_i], obj_nums[batch_i], obj_classes[batch_i], part_classes[batch_i], np_pred_adj_mat[batch_i, ...], part_human_id[batch_i])

def vcoco_evaluation(args, vcocoeval, imageset, all_results):
    det_file = os.path.join(args.eval_root, '{}_detections.pkl'.format(imageset))
    pickle.dump(all_results, open(det_file, 'wb'))
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)

def compute_mean_avg_prec(y_true, y_score):
    # avg_prec = sklearn.metrics.average_precision_score(y_true[:, 1:], y_score[:, 1:], average=None)
    # print type(avg_prec), avg_prec
    try:
        avg_prec = sklearn.metrics.average_precision_score(y_true[:, 1:], y_score[:, 1:], average='micro')
        return avg_prec
        # mean_avg_prec = np.nansum(avg_prec) / len(avg_prec)
        # mean_avg_prec = np.nanmean(avg_prec)
    except ValueError:
        mean_avg_prec = 0

    return mean_avg_prec

def get_vcocoeval(args, imageset):
    return vsrl_eval.VCOCOeval(os.path.join(args.data_root, '..', 'v-coco/data/vcoco/vcoco_{}.json'.format(imageset)),
                               os.path.join(args.data_root, '..', 'v-coco/data/instances_vcoco_all_2014.json'),
                               os.path.join(args.data_root, '..', 'v-coco/data/splits/vcoco_{}.ids'.format(imageset)))


def main(args):
    # np.random.seed(0)
    # torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))
    
    # Load data
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = utils.get_vcoco_data(args)

    # Evaluation setup
    if not os.path.exists(args.eval_root):
        os.makedirs(args.eval_root)
    train_vcocoeval = get_vcocoeval(args, 'train')
    val_vcocoeval = get_vcocoeval(args, 'val')
    test_vcocoeval = get_vcocoeval(args, 'test')

    # Get data size and define model
    edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, part_num, obj_num, obj_classes, part_classes, part_adj_mat, _ = training_set[0]
    edge_feature_size, node_feature_size = edge_features.shape[2], node_features.shape[1]
    # message_size = int(edge_feature_size/2)*2
    # message_size = edge_feature_size*2
    # message_size = 1024
    message_size = edge_feature_size

    model_args = {
        'model_path': args.resume, 
        'edge_feature_size': edge_feature_size, 
        'node_feature_size': node_feature_size,
        'message_size': message_size, 
        'link_hidden_size': 256, 
        'link_hidden_layers': args.link_layer, 
        'link_relu': False, 
        'update_hidden_layers': args.update_layer, 
        'update_dropout': 0.0, 
        'update_bias': True, 
        'propagate_layers': args.prop_layer, 
        'hoi_classes': action_class_num, 
        'roles_num': roles_num, 
        'resize_feature_to_message_size': False, 
        'feature_type': args.feature_type}

    model = models.GPNN_VCOCO(model_args)
    del edge_features, node_features, adj_mat
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss(size_average=True)
    multi_label_loss = torch.nn.MultiLabelSoftMarginLoss(size_average=True)
    if args.cuda:
        model = model.cuda()
        mse_loss = mse_loss.cuda()
        multi_label_loss = multi_label_loss.cuda()

    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

        visualize(args, train_loader, model, mse_loss, multi_label_loss, val_vcocoeval, logger)
        visualize(args, valid_loader, model, mse_loss, multi_label_loss, val_vcocoeval, logger)
        visualize(args, test_loader, model, mse_loss, multi_label_loss, val_vcocoeval, logger)

    # For testing
    # loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    # if loaded_checkpoint:
    #     args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint
    # validate(args, test_loader, model, mse_loss, multi_label_loss, test_vcocoeval, test=True)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def visualize(args, val_loader, model, mse_loss, multi_label_loss, vcocoeval, logger=None, test=False):
    if args.visualize:
        result_folder = os.path.join(args.tmp_root, 'results/VCOCO/detections/', 'top'+str(args.vis_top_k))
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    batch_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()

    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))
    all_results = list()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, part_num, obj_num, obj_classes, part_classes, part_adj_mat, img_names) in enumerate(val_loader):
        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        part_adj_mat = utils.to_variable(part_adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)
        node_roles = utils.to_variable(node_roles, args.cuda)

        pred_adj_mat, pred_node_labels, pred_node_roles = model(edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, human_num, part_num, obj_num, part_classes, args)
        for j in range(len(img_names)):
            pickle.dump({'adj_mat': pred_adj_mat[j], 'node_labels': pred_node_labels[j], 'node_roles': pred_node_roles[j]}, open(os.path.join(args.eval_root, img_names[j] + '.pred'), 'wb'))
        if i % 10 == 0:
            print('\r%d/%d'%(i, len(val_loader)), end='')

def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = config.Paths()

    feature_type = 'resnet'

    # Path settings
    parser = argparse.ArgumentParser(description='VCOCO dataset')
    parser.add_argument('--project-root', default=paths.project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    parser.add_argument('--data-root', default=paths.vcoco_data_root, help='data path')
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'vcoco/parsing_{}'.format(feature_type)), help='log files path')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/vcoco/parsing_{}'.format(feature_type)), help='path to latest checkpoint')
    parser.add_argument('--eval-root', default=os.path.join(paths.tmp_root, 'evaluation/vcoco/{}'.format(feature_type)), help='path to save evaluation file')
    parser.add_argument('--feature-type', default=feature_type, help='feature_type')
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize final results')
    parser.add_argument('--vis-top-k', type=int, default=1, metavar='N', help='Top k results to visualize')

    # Model parameters
    parser.add_argument('--prop-layer', type=int, default=3, metavar='N',
                        help='Number of propogation layers (default: 3)')
    parser.add_argument('--update-layer', type=int, default=1, metavar='N',
                        help='Number of update hidden layers (default: 1)')
    parser.add_argument('--link-layer', type=int, default=3, metavar='N',
                        help='Number of link hidden layers (default: 3)')


    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--link-weight', type=float, default=2, metavar='N',
                        help='Loss weight of existing edges')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-4, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.8, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.8)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # i/o
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

    # Debug
    parser.add_argument('--debug', action='store_true', default=False, help='Runs only 10 batch and 1 epoch')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
