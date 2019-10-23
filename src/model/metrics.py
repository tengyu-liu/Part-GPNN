import numpy as np
import sklearn.metrics

def compute_mAP(pred, gt, part_human_ids):
    # pred: N x N x M
    # gt  : N x N x M
    # phi : P
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
        lifted_pred_sum[i_lifted] = np.sum(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0)
        lifted_pred_max[i_lifted] = np.max(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0)
        lifted_pred_mean[i_lifted] = np.mean(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0)

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