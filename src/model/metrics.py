import numpy as np
import sklearn.metrics

def compute_mAP(pred, gt, part_human_ids, node_num):
    # pred: N x N x M
    # gt  : N x N x M
    # phi : P

    gt = gt[:node_num, :node_num, :]

    human_ids = set(part_human_ids)
    human_num = len(human_ids)

    part_num = len(part_human_ids)
    obj_num = len(pred) - part_num

    lifted_pred_sum = np.zeros([human_num, obj_num, pred.shape[-1]])
    lifted_pred_max = np.zeros([human_num, obj_num, pred.shape[-1]])
    lifted_pred_mean = np.zeros([human_num, obj_num, pred.shape[-1]])
    lifted_gt = np.zeros([human_num, obj_num, pred.shape[-1]])
    
    # Step 1. Get lifted prediction
    for i_lifted, i_human in enumerate(human_ids):
        lifted_gt[i_lifted] = gt[ np.where(np.equal(part_human_ids, i_human))[0][0], part_num:, : ]
        print(np.sum(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0).shape)
        print(np.sum(pred[ part_num:, np.where(np.equal(part_human_ids, i_human))[0], :], axis=0).shape)
        lifted_pred_sum[i_lifted] = np.sum(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0) + np.sum(pred[ part_num:, np.where(np.equal(part_human_ids, i_human))[0], :], axis=0)
        lifted_pred_max[i_lifted] = np.maximum(np.max(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0), np.max(pred[ part_num:, np.where(np.equal(part_human_ids, i_human))[0], :], axis=0))
        lifted_pred_mean[i_lifted] = (np.mean(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0) + np.mean(pred[ part_num:, np.where(np.equal(part_human_ids, i_human))[0], :], axis=0)) / 2

    # Step 2. Compute mAP
    y_true = lifted_gt.reshape([-1, pred.shape[-1]])
    y_pred_sum = lifted_pred_sum.reshape([-1, pred.shape[-1]])
    y_pred_max = lifted_pred_max.reshape([-1, pred.shape[-1]])
    y_pred_mean = lifted_pred_mean.reshape([-1, pred.shape[-1]])

    avg_prec_sum = sklearn.metrics.average_precision_score(y_true, y_pred_sum, average='micro')
    avg_prec_max = sklearn.metrics.average_precision_score(y_true, y_pred_max, average='micro')
    avg_prec_mean = sklearn.metrics.average_precision_score(y_true, y_pred_mean, average='micro')

    if np.isnan(avg_prec_max):
        print(y_true.shape, y_true, avg_prec_sum, avg_prec_max, avg_prec_mean)
        exit()

    return avg_prec_sum, avg_prec_max, avg_prec_mean

part_ids = {'Torso': [1, 2],
            'Right Hand': [3],
            'Left Hand': [4],
            'Left Foot': [5],
            'Right Foot': [6],
            'Upper Leg Right': [7, 9],
            'Upper Leg Left': [8, 10],
            'Lower Leg Right': [11, 13],
            'Lower Leg Left': [12, 14],
            'Upper Arm Left': [15, 17],
            'Upper Arm Right': [16, 18],
            'Lower Arm Left': [19, 21],
            'Lower Arm Right': [20, 22], 
            'Head': [23, 24],
            'Upper Body': [1, 2, 3, 4, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 
            'Lower Body': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
            'Left Arm': [4, 15, 17, 19, 21], 
            'Right Arm': [3, 16, 18, 20, 22], 
            'Left Leg': [5, 8, 10, 12, 14], 
            'Right Leg': [6, 7, 9, 11, 13], 
            'Full Body': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
            }

part_names = list(part_ids.keys())

hake_to_densepose = [
    ['Right Foot'], 
    ['Upper Leg Right', 'Lower Leg Right', 'Right Leg'], 
    ['Upper Leg Left', 'Lower Leg Left', 'Left Leg'], 
    ['Left Foot'],
    ['Torso', 'Upper Leg Right', 'Upper Leg Left'],
    ['Head'],
    ['Right Hand'],
    ['Upper Arm Right', 'Lower Arm Right', 'Right Arm'],
    ['Upper Arm Left', 'Lower Arm Left', 'Left Arm'],
    ['Left Hand']
    ]

hake_to_densepose_idx = [[part_names.index(x) for x in y] for y in hake_to_densepose]

def compute_part_mAP(pred, gt, part_classes):
    # pred: N x N x M
    # gt: K

    part_num = len(part_classes)
    
    pred_sum = np.zeros(gt.shape)
    pred_max = np.zeros(gt.shape)
    pred_mean = np.zeros(gt.shape)

    for i_part, i_part_list in enumerate(hake_to_densepose_idx):
        pred_sum[i_part] = np.sum(pred[ np.in1d(part_classes, i_part_list) , part_num:, :]) + np.sum(pred[ part_num:, np.in1d(part_classes, i_part_list), :])
        pred_max[i_part] = max(np.max(pred[ np.in1d(part_classes, i_part_list) , part_num:, :]), np.max(pred[ part_num:, np.in1d(part_classes, i_part_list), :]))
        pred_mean[i_part] = (np.mean(pred[ np.in1d(part_classes, i_part_list) , part_num:, :]) + np.mean(pred[ part_num:, np.in1d(part_classes, i_part_list), :])) / 2
    
    avg_prec_sum = sklearn.metrics.average_precision_score([gt], [pred_sum], average='micro')
    avg_prec_max = sklearn.metrics.average_precision_score([gt], [pred_max], average='micro')
    avg_prec_mean = sklearn.metrics.average_precision_score([gt], [pred_mean], average='micro')

    return avg_prec_sum, avg_prec_max, avg_prec_mean
