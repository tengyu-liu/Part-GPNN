import numpy as np

def mAP(pred, gt, part_human_ids):
    human_ids = set(part_human_ids)
    human_num = len(human_ids)

    part_num = len(part_human_ids)
    obj_num = len(pred) - part_num

    lefted_pred_sum = np.zeros([human_num, obj_num, pred.shape[-1]])
    lefted_pred_max = np.zeros([human_num, obj_num, pred.shape[-1]])
    lefted_pred_mean = np.zeros([human_num, obj_num, pred.shape[-1]])
    lifted_gt = np.zeros([human_num, obj_num, pred.shape[-1]])
    
    # TODO: Compute mAP
    for i_lifted, i_human in enumerate(human_ids):


