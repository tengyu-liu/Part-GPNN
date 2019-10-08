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


def evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=False, obj_action_pairs=None, node_classes=None):
    pred_node_labels_prob = torch.nn.Sigmoid()(pred_node_labels)
    np_pred_node_labels = pred_node_labels_prob.data.cpu().numpy()
    np_node_labels = node_labels.data.cpu().numpy()

    if node_classes is not None:
        human_num, obj_classes = node_classes
        new_y_true = np.empty((2 * len(det_indices), action_class_num))
        new_y_score = np.empty((2 * len(det_indices), action_class_num))

        for y_i, (batch_i, i, j, s) in enumerate(det_indices):
            new_y_true[2*y_i, :] = np_node_labels[batch_i, i, :]
            new_y_true[2*y_i+1, :] = np_node_labels[batch_i, j, :]
            new_y_score[2*y_i, :] = np_pred_node_labels[batch_i, i, :]
            label_mask = obj_action_pairs[obj_classes[batch_i][j - human_num[batch_i]]]
            new_y_score[2*y_i+1, :] = np_pred_node_labels[batch_i, j, :] * label_mask
    else:
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


def weighted_loss(output, target):
    weight_mask = torch.autograd.Variable(torch.ones(target.size()))
    if hasattr(args, 'cuda') and args.cuda:
        weight_mask = weight_mask.cuda()
    link_weight = args.link_weight if hasattr(args, 'link_weight') else 1.0
    weight_mask += target * link_weight
    if hasattr(args, 'cuda') and args.cuda:
        return torch.nn.MultiLabelSoftMarginLoss(weight=weight_mask).cuda()(output, target)
    else:
        return torch.nn.MultiLabelSoftMarginLoss(weight=weight_mask)(output, target)


def loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, human_nums, part_nums, obj_nums, part_human_id, part_adj_mat, mse_loss, multi_label_loss, obj_action_pairs=None, obj_classes=None):

    pred_adj_mat = pred_adj_mat.cpu()
    pred_node_labels = pred_node_labels.cpu()
    pred_node_roles = pred_node_roles.cpu()

    det_indices = list()
    # lift predictions to human level
    pred_adj_mat_lifted = torch.autograd.Variable(torch.zeros(adj_mat.size()))
    pred_node_labels_lifted_sum = torch.autograd.Variable(torch.zeros(node_labels.size()))
    pred_node_labels_lifted_max = torch.autograd.Variable(torch.zeros(node_labels.size()))
    pred_node_roles_lifted = torch.autograd.Variable(torch.zeros(node_roles.size()))
    
    if args.cuda:
        pred_adj_mat_lifted = pred_adj_mat_lifted#.cuda()
        pred_node_labels_lifted_sum = pred_node_labels_lifted_sum#.cuda()
        pred_node_labels_lifted_max = pred_node_labels_lifted_max#.cuda()
        pred_node_roles_lifted = pred_node_roles_lifted#.cuda()

    # pred_adj_mat_prob = torch.nn.Sigmoid()(pred_adj_mat)

    for batch_i in range(pred_adj_mat.size()[0]):
        
        phi = np.array(part_human_id[batch_i])
        for i_upper_node in range(human_nums[batch_i] + obj_nums[batch_i]):
            if i_upper_node < human_nums[batch_i]:
                i_lower_indices = torch.tensor(np.argwhere(phi == i_upper_node)).squeeze()
            else:
                i_lower_indices = torch.tensor(i_upper_node - human_nums[batch_i] + part_nums[batch_i]).squeeze()

            v1 = torch.index_select(pred_adj_mat[batch_i], dim=0, index=i_lower_indices)

            for j_upper_node in range(human_nums[batch_i] + obj_nums[batch_i]):
                if j_upper_node < human_nums[batch_i]:
                    j_lower_indices = torch.tensor(np.argwhere(phi == j_upper_node)).squeeze()
                else:
                    j_lower_indices = torch.tensor(j_upper_node - human_nums[batch_i] + part_nums[batch_i]).squeeze()

                v2 = torch.index_select(v1, dim=1, index=j_lower_indices)
                # pred_adj_mat_lifted[batch_i, i_upper_node, j_upper_node] = torch.max(v2)
                pred_adj_mat_lifted[batch_i, i_upper_node, j_upper_node] = torch.sum(v2)
            
            v1l = torch.index_select(pred_node_labels[batch_i], dim=0, index=i_lower_indices)
            v1r = torch.index_select(pred_node_roles[batch_i], dim=0, index=i_lower_indices)

            # pred_node_labels_lifted[batch_i, i_upper_node, :] = torch.max(v1l, dim=0).values
            # pred_node_roles_lifted[batch_i, i_upper_node, :] = torch.max(v1r, dim=0).values
            if obj_action_pairs is not None and i_upper_node >= human_nums[batch_i]:
                pred_node_labels_lifted_sum[batch_i, i_upper_node, :] = torch.sum(v1l, dim=0) * obj_action_pairs[obj_classes[batch_i][i_upper_node - human_nums[batch_i]]]
                pred_node_labels_lifted_max[batch_i, i_upper_node, :] = torch.max(v1l, dim=0).values * obj_action_pairs[obj_classes[batch_i][i_upper_node - human_nums[batch_i]]]
            else:
                pred_node_labels_lifted_sum[batch_i, i_upper_node, :] = torch.sum(v1l, dim=0)
                pred_node_labels_lifted_max[batch_i, i_upper_node, :] = torch.max(v1l, dim=0).values
    
            pred_node_roles_lifted[batch_i, i_upper_node, :] = torch.sum(v1r, dim=0)

        # for i_lower_node in range(pred_adj_mat.size()[1]):
        #     if i_lower_node < part_nums[batch_i]:
        #         i_upper_node = part_human_id[batch_i][i_lower_node]
        #     elif j_lower_node < part_nums[batch_i] + obj_nums[batch_i]:
        #         i_upper_node = i_lower_node - part_nums[batch_i] + human_nums[batch_i]
        #     for j_lower_node in range(pred_adj_mat.size()[2]):
        #         if j_lower_node < part_nums[batch_i]:
        #             j_upper_node = part_human_id[batch_i][j_lower_node]
        #         elif j_lower_node < part_nums[batch_i] + obj_nums[batch_i]:
        #             j_upper_node = j_lower_node - part_nums[batch_i] + human_nums[batch_i]
        #         pred_adj_mat_lifted[batch_i,i_upper_node,j_upper_node] += pred_adj_mat[batch_i,i_lower_node,j_lower_node]

        #     pred_node_labels_lifted[batch_i,i_upper_node,:] += pred_node_labels[batch_i,i_lower_node,:]
        #     pred_node_roles_lifted[batch_i,i_upper_node,:] += pred_node_roles[batch_i,i_lower_node,:]

    np_pred_adj_mat = torch.nn.Sigmoid()(pred_adj_mat_lifted).data.cpu().numpy()
    # np_adj_mat = adj_mat.data.cpu().numpy()
    det_indices = list()

    pred_adj_mat = pred_adj_mat.cuda()
    pred_node_labels_lifted_sum = pred_node_labels_lifted_sum.cuda()
    pred_adj_mat_lifted = pred_adj_mat_lifted.cuda()
    pred_node_roles_lifted = pred_node_roles_lifted.cuda()
    pred_node_labels_lifted_max = pred_node_labels_lifted_max.cuda()

    batch_size = pred_adj_mat.size()[0]
    for batch_i in range(batch_size):
        for i in range(human_nums[batch_i]):
            for j in range(human_nums[batch_i], human_nums[batch_i]+obj_nums[batch_i]):
                # TODO: debug metrics
                # if np_pred_adj_mat[batch_i, i, j] > 0.5:
                # if np_adj_mat[batch_i, i, j] == 1:
                det_indices.append((batch_i, i, j, np_pred_adj_mat[batch_i, i, j]))

    # Loss for labels and adjacency matrices
    # loss = multi_label_loss(pred_node_labels.view(-1, hoi_class_num), node_labels.view(-1, hoi_class_num)) + multi_label_loss(pred_adj_mat.view(batch_size, -1), adj_mat.view(batch_size, -1))
    # loss = weighted_loss(pred_node_labels.view(-1, action_class_num), node_labels.view(-1, action_class_num))
    # loss = weighted_loss(pred_node_labels.view(-1, action_class_num), node_labels.view(-1, action_class_num)) + weighted_loss(pred_adj_mat, adj_mat)

    # pred_adj_mat_lifted = pred_adj_mat_lifted.cuda()
    # pred_node_labels_lifted = pred_node_labels_lifted.cuda()
    # pred_node_roles_lifted = pred_node_roles_lifted.cuda()

    _, roles_indices = torch.max(node_roles, 2)

    loss = 0
    batch_size = pred_node_labels.size()[0]
    for batch_i in range(batch_size):
        node_num = human_nums[batch_i] + obj_nums[batch_i]
        part_num = part_nums[batch_i]
        loss_1 = weighted_loss(pred_node_labels_lifted_sum[batch_i, :node_num, :].view(-1, action_class_num), node_labels[batch_i, :node_num, :].view(-1, action_class_num))
        loss_2 = weighted_loss(pred_adj_mat_lifted[batch_i, :node_num, :node_num], adj_mat[batch_i, :node_num, :node_num])
        # loss_3 = weighted_loss(pred_adj_mat[batch_i, :part_num, :part_num], part_adj_mat[batch_i, :part_num, :part_num])
        loss += loss_1 + loss_2

        # Ablative analysis
        # loss = weighted_loss(pred_node_labels[batch_i, :node_num, :].view(-1, action_class_num),
        #                  node_labels[batch_i, :node_num, :].view(-1, action_class_num))

    # # MSE loss for roles prediction
    # loss += mse_loss(pred_node_roles, node_roles)

    # Cross entropy loss for roles prediction
    ce_loss = torch.nn.CrossEntropyLoss()
    if args.cuda:
        ce_loss = ce_loss.cuda()
    loss_4 = ce_loss(pred_node_roles_lifted.view(-1, roles_num), roles_indices.view(-1))
    loss += loss_4

    return pred_node_labels_lifted_max, det_indices, loss


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
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))
    
    # Load data
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader = utils.get_vcoco_data(args)

    # get training hard-negatives
    if os.path.exists('obj_action_pairs_cont.pkl'):
        obj_action_pairs = pickle.load(open('obj_action_pairs_cont.pkl', 'rb'))
    else:
        obj_action_pairs = np.ones([81,27])
        for edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, part_num, obj_num, obj_classes, part_classes, part_adj_mat, _ in train_loader:
            for i in range(len(adj_mat)):
                for obj_id in range(obj_num[i]):
                    for action_id in range(len(node_labels[i][human_num[i] + obj_id])):
                        obj_action_pairs[obj_classes[i][obj_id], action_id] += 1

        obj_action_pairs = obj_action_pairs / np.sum(obj_action_pairs, axis=-1, keepdims=True)
        pickle.dump(obj_action_pairs, open('obj_action_pairs_cont.pkl', 'wb'))

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
        'feature_type': args.feature_type,
        'po_type' : args.po_type}

    if args.model_type == 'V1':
        model = models.GPNN_VCOCO(model_args)
    elif args.model_type == 'V2':
        model = models.GPNN_VCOCO_v2(model_args)
    elif args.model_type == 'PG':
        model = models.GPNN_VCOCO_PG(model_args)

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

    epoch_errors = list()
    avg_epoch_error = np.inf
    best_epoch_error = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        logger.log_value('learning_rate', args.lr).step()

        # train for one epoch
        part_obj_prior = train(args, train_loader, model, mse_loss, multi_label_loss, optimizer, epoch, train_vcocoeval, logger, obj_action_pairs)

        if args.debug:
            break

        # test on validation set
        epoch_error = validate(args, valid_loader, model, mse_loss, multi_label_loss, val_vcocoeval, logger, obj_action_pairs=obj_action_pairs)

        epoch_errors.append(epoch_error)
        if len(epoch_errors) == 2:
            new_avg_epoch_error = np.mean(np.array(epoch_errors))
            if avg_epoch_error - new_avg_epoch_error < 0.005:
                pass
            avg_epoch_error = new_avg_epoch_error
            epoch_errors = list()

        if epoch % 2 == 1:
            print('Learning rate decrease')
            args.lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        is_best = epoch_error < best_epoch_error
        best_epoch_error = min(epoch_error, best_epoch_error)
        datasets.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                        'best_epoch_error': best_epoch_error, 'avg_epoch_error': avg_epoch_error,
                                        'optimizer': optimizer.state_dict(), },
                                       is_best=is_best, directory=args.resume)
        print('best_epoch_error: {}, avg_epoch_error: {}'.format(best_epoch_error,  avg_epoch_error))

    # For testing
    # loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    # if loaded_checkpoint:
    #     args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint
    # validate(args, test_loader, model, mse_loss, multi_label_loss, test_vcocoeval, test=True)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))
    print(args.resume)


def train(args, train_loader, model, mse_loss, multi_label_loss, optimizer, epoch, vcocoeval, logger, obj_action_pairs):
    batch_time = logutil.AverageMeter()
    data_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()

    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))
    all_results = list()

    # switch to train mode
    model.train()

    end_time = time.time()

    th_obj_action_pairs = torch.tensor(obj_action_pairs).float()#.cuda().float()

    for i, (edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, part_num, obj_num, obj_classes, part_classes, part_adj_mat, _) in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)

        adj_mat = utils.to_variable(adj_mat, args.cuda)
        # part_adj_mat = utils.to_variable(part_adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)
        node_roles = utils.to_variable(node_roles, args.cuda)

        pred_adj_mat, pred_node_labels, pred_node_roles = model(edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, human_num, part_num, obj_num, part_classes, obj_classes, args)
        if args.NRT:
            pred_node_label_lifted, det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, human_num, part_num, obj_num, part_human_id, part_adj_mat, mse_loss, multi_label_loss, obj_action_pairs=th_obj_action_pairs, obj_classes=obj_classes)
        else:
            pred_node_label_lifted, det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, human_num, part_num, obj_num, part_human_id, part_adj_mat, mse_loss, multi_label_loss)
        append_results(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, part_human_id, img_id,
                           obj_boxes, part_boxes, human_boxes, human_num, part_num, obj_num, obj_classes, part_classes, all_results)

        # TODO: node_classes
        if args.NRE:
            node_classes = (human_num, obj_classes)
        else:
            node_classes = None

        # Log and back propagate
        if len(det_indices) > 0:
            y_true, y_score = evaluation(det_indices, pred_node_label_lifted, node_labels, y_true, y_score, obj_action_pairs=obj_action_pairs, node_classes=node_classes)

        if not isinstance(loss, int):
            losses.update(loss.item(), edge_features.size()[0])
            loss.backward()
            optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.log_interval == 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Mean Avg Precision {mean_avg_prec:.4f} ({mean_avg_prec:.4f})\t'
                  'Detected HOIs {y_shape}'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, mean_avg_prec=mean_avg_prec, y_shape=y_true.shape))
    
        if args.debug and i == 30:
            break

    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
    # vcoco_evaluation(args, vcocoeval, 'train', all_results)

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)
        logger.log_value('train_epoch_map', mean_avg_prec)

    print('Epoch: [{0}] Avg Mean Precision {map:.4f}; Average Loss {loss.avg:.4f}; Avg Time x Batch {b_time.avg:.4f}'
          .format(epoch, map=mean_avg_prec, loss=losses, b_time=batch_time))


def validate(args, val_loader, model, mse_loss, multi_label_loss, vcocoeval, logger=None, test=False, obj_action_pairs=None):
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

    th_obj_action_pairs = torch.tensor(obj_action_pairs).float()#.cuda().float()

    end = time.time()
    for i, (edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, obj_boxes, part_boxes, human_boxes, img_id, img_name, human_num, part_num, obj_num, obj_classes, part_classes, part_adj_mat, _) in enumerate(val_loader):
        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        # part_adj_mat = utils.to_variable(part_adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)
        node_roles = utils.to_variable(node_roles, args.cuda)

        pred_adj_mat, pred_node_labels, pred_node_roles = model(edge_features, node_features, part_human_id, adj_mat, node_labels, node_roles, human_num, part_num, obj_num, part_classes, obj_classes, args)
        if args.NRT:
            pred_node_label_lifted, det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, human_num, part_num, obj_num, part_human_id, part_adj_mat, mse_loss, multi_label_loss, obj_action_pairs=th_obj_action_pairs, obj_classes=obj_classes)
        else:
            pred_node_label_lifted, det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, human_num, part_num, obj_num, part_human_id, part_adj_mat, mse_loss, multi_label_loss)
        append_results(pred_adj_mat, adj_mat, pred_node_labels, node_labels, pred_node_roles, node_roles, part_human_id, img_id,
                           obj_boxes, part_boxes, human_boxes, human_num, part_num, obj_num, obj_classes, part_classes, all_results)

        # TODO: node_classes
        if args.NRE:
            node_classes = (human_num, obj_classes)
        else:
            node_classes = None

        # Log
        if len(det_indices) > 0:
            losses.update(loss.item(), len(det_indices))
            y_true, y_score = evaluation(det_indices, pred_node_label_lifted, node_labels, y_true, y_score, test=test, obj_action_pairs=obj_action_pairs, node_classes=node_classes)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Mean Avg Precision {mean_avg_prec:.4f} ({mean_avg_prec:.4f})\t'
                  'Detected HOIs {y_shape}'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, mean_avg_prec=mean_avg_prec, y_shape=y_true.shape))

        if args.debug and i == 9:
            break

    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
    if test:
        vcoco_evaluation(args, vcocoeval, 'test', all_results)
        if args.visualize:
            utils.visualize_vcoco_result(args, result_folder, all_results)
    else:
        pass
        # vcoco_evaluation(args, vcocoeval, 'val', all_results)

    print(' * Average Mean Precision {mean_avg_prec:.4f}; Average Loss {loss.avg:.4f}'
          .format(mean_avg_prec=mean_avg_prec, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_map', mean_avg_prec)

    return 1.0 - mean_avg_prec


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
    parser.add_argument('--model-type', type=str, default='V1', 
                        help='GPNN model type')
    parser.add_argument('--po-type', type=str, default='mult', 
                        help='Part-Object prior type')


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

    # Feature
    parser.add_argument('--extra-feature', action='store_true', default=False, help='Include 101-d extra node feature')

    # Negative Rejection
    parser.add_argument('--NRT', action='store_true', default=False, help='Reject negative object-action pairs in training')
    parser.add_argument('--NRE', action='store_true', default=False, help='Reject negative object-action pairs in evaluation')

    # AdjMat Loss
    parser.add_argument('--ignore-adj', action='store_true', default=False, help='ignores adjacency matrix in loss function')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
