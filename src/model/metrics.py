import metadata
import numpy as np
import sklearn.metrics

def compute_mAP(pred, gt, part_human_ids, node_num):
    # pred: N x N x M
    # gt  : N x N x M
    # phi : P

    gt = gt[:node_num, :node_num, :]
    pred = pred[:node_num, :node_num, :]

    human_ids = set(part_human_ids)
    human_num = len(human_ids)

    part_num = len(part_human_ids)
    obj_num = node_num - part_num

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

    if np.isnan(avg_prec_max) or np.isnan(avg_prec_sum) or np.isnan(avg_prec_mean):
        print(human_ids)
        print(y_true.shape, y_true, avg_prec_sum, avg_prec_max, avg_prec_mean)
        exit()

    return avg_prec_sum, avg_prec_max, avg_prec_mean

def append_results(all_results_sum, all_results_max, all_results_mean, human_boxes, part_human_ids, pred_label, pred_role, obj_nums, obj_boxes, obj_classes, img_ids):
    for i_item in range(len(pred_label)):
        curr_img_id = img_ids[i_item]
        curr_human_boxes = human_boxes[i_item]

        part_num = len(part_human_ids[i_item])
        human_num = len(human_boxes[i_item])
        obj_num = obj_nums[i_item]

        for i_human, human_box in enumerate(curr_human_boxes):
            instance = {
                'image_id' : curr_img_id,
                'person_box' : human_box,
            }

            result_sum = instance.copy()
            result_max = instance.copy()
            result_mean = instance.copy()

            if i_human not in part_human_ids[i_item]:
                continue
            #     for action_index, action in enumerate(metadata.action_classes):
            #         if action == 'none':
            #             continue
            #         result_sum['{}_agent'.format(action)] = -np.float('inf')
            #         result_max['{}_agent'.format(action)] = -np.float('inf')
            #         result_mean['{}_agent'.format(action)] = -np.float('inf')
            #         for role in metadata.action_roles[action][1:]:
            #             action_role_key = '{}_{}'.format(action, role)
            #             obj_info = np.array([0.,0.,0.,0.,0.])
            #             result_sum[action_role_key] = obj_info
            #             result_sum['{}_class'.format(role)] = 0
            #             result_max[action_role_key] = obj_info
            #             result_max['{}_class'.format(role)] = 0
            #             result_mean[action_role_key] = obj_info
            #             result_mean['{}_class'.format(role)] = 0
            #     all_results_sum.append(result_sum)
            #     all_results_max.append(result_max)
            #     all_results_mean.append(result_mean)
            #     continue

            pred_label_sum = np.sum(pred_label[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0)
            pred_label_max = np.max(pred_label[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0)
            pred_label_mean = np.mean(pred_label[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0)
    
            pred_role_sum = np.sum(pred_role[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0)
            pred_role_max = np.max(pred_role[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0)
            pred_role_mean = np.mean(pred_role[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], :, :], axis=0)

            for action_index, action in enumerate(metadata.action_classes):
                if action == 'none':
                    continue
                    
                result_sum['{}_agent'.format(action)] = np.max(pred_label_sum[:,action_index])
                result_max['{}_agent'.format(action)] = np.max(pred_label_max[:,action_index])
                result_mean['{}_agent'.format(action)] = np.max(pred_label_mean[:,action_index])

                for role in metadata.action_roles[action][1:]:
                    role_index = metadata.role_index[role]
                    action_role_key = '{}_{}'.format(action, role)
                    best_score = -np.inf
                    best_j = -1
                    for i_obj in range(obj_num):
                        action_role_score = pred_label_sum[i_obj + part_num, action_index] + pred_role_sum[i_obj + part_num, role_index]
                        if action_role_score > best_score:
                            best_score = action_role_score
                            obj_info = np.append(obj_boxes[i_item][i_obj, :], action_role_score)
                            best_j = i_obj
                    result_sum[action_role_key] = obj_info
                    result_sum['{}_class'.format(role)] = obj_classes[i_item][best_j]

                for role in metadata.action_roles[action][1:]:
                    role_index = metadata.role_index[role]
                    action_role_key = '{}_{}'.format(action, role)
                    best_score = -np.inf
                    best_j = -1
                    for i_obj in range(obj_num):
                        action_role_score = pred_label_max[i_obj + part_num, action_index] + pred_role_max[i_obj + part_num, role_index]
                        if action_role_score > best_score:
                            best_score = action_role_score
                            obj_info = np.append(obj_boxes[i_item][i_obj, :], action_role_score)
                            best_j = i_obj
                    result_max[action_role_key] = obj_info
                    result_max['{}_class'.format(role)] = obj_classes[i_item][best_j]

                for role in metadata.action_roles[action][1:]:
                    role_index = metadata.role_index[role]
                    action_role_key = '{}_{}'.format(action, role)
                    best_score = -np.inf
                    best_j = -1
                    for i_obj in range(obj_num):
                        action_role_score = pred_label_mean[i_obj + part_num, action_index] + pred_role_mean[i_obj + part_num, role_index]
                        if action_role_score > best_score:
                            best_score = action_role_score
                            obj_info = np.append(obj_boxes[i_item][i_obj, :], action_role_score)
                            best_j = i_obj
                    result_mean[action_role_key] = obj_info
                    result_mean['{}_class'.format(role)] = obj_classes[i_item][best_j]

            all_results_sum.append(result_sum)
            all_results_max.append(result_max)
            all_results_mean.append(result_mean)

    return all_results_sum, all_results_max, all_results_mean

#
#
#
