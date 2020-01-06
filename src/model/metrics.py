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
        try:
            lifted_pred_sum[i_lifted] = np.sum(pred[ np.where(np.equal(part_human_ids, i_human))[0], part_num:, :], axis=0)
        except:
            print(pred.shape, 'H', human_num, 'O', obj_num, 'P', part_num, 'N', node_num)
            raise
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
        print(y_true.shape, y_true, avg_prec_sum, avg_prec_max, avg_prec_mean)
        exit()

    return avg_prec_sum, avg_prec_max, avg_prec_mean

# part_ids = {'Right Shoulder': [2],
#             'Left Shoulder': [5],
#             'Knee Right': [10],
#             'Knee Left': [13],
#             'Ankle Right': [11],
#             'Ankle Left': [14],
#             'Elbow Left': [6],
#             'Elbow Right': [3],
#             'Hand Left': [7],
#             'Hand Right': [4],
#             'Head': [0],
#             'Hip': [8],
#             'Upper Body': [2,5,6,3,7,4,0,8],
#             'Lower Body': [10,13,11,14,8],
#             'Left Arm': [5,6,7],
#             'Right Arm': [2,3,4],
#             'Left Leg': [8,10,11],
#             'Right Leg': [8,13,14],
#             'Full Body': [2,5,10,13,11,14,6,3,7,4,0,8], 
#             }


# part_names = list(part_ids.keys())

# hake_to_densepose = [
#     ['Right Foot'], 
#     ['Upper Leg Right', 'Lower Leg Right', 'Right Leg'], 
#     ['Upper Leg Left', 'Lower Leg Left', 'Left Leg'], 
#     ['Left Foot'],
#     ['Torso', 'Upper Leg Right', 'Upper Leg Left'],
#     ['Head'],
#     ['Right Hand'],
#     ['Upper Arm Right', 'Lower Arm Right', 'Right Arm'],
#     ['Upper Arm Left', 'Lower Arm Left', 'Left Arm'],
#     ['Left Hand']
#     ]

# hake_to_densepose_idx = [[part_names.index(x) for x in y] for y in hake_to_densepose]

# def compute_part_mAP(pred, gt, part_classes):
#     # pred: N x N x M
#     # gt: K

#     part_num = len(part_classes)
    
#     pred_sum = np.zeros([len(gt)])
#     pred_max = np.zeros([len(gt)])
#     pred_mean = np.zeros([len(gt)])

#     pred = pred[:part_num, part_num:, :]

#     for i_part, i_part_list in enumerate(hake_to_densepose_idx):
#         idx = np.in1d(part_classes, i_part_list)
#         if idx.sum() > 0:
#             pred_sum[i_part] = np.sum(pred[ idx , ...])
#             pred_max[i_part] = np.max(pred[ idx , ...])
#             pred_mean[i_part] = np.mean(pred[ idx , ...])

#     avg_prec_sum = sklearn.metrics.average_precision_score(np.array([gt]), np.array([pred_sum]), average='micro')
#     avg_prec_max = sklearn.metrics.average_precision_score(np.array([gt]), np.array([pred_max]), average='micro')
#     avg_prec_mean = sklearn.metrics.average_precision_score(np.array([gt]), np.array([pred_mean]), average='micro')

#     return avg_prec_sum, avg_prec_max, avg_prec_mean

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
                for action_index, action in enumerate(metadata.action_classes):
                    if action == 'none':
                        continue
                    result_sum['{}_agent'.format(action)] = -np.float('inf')
                    result_max['{}_agent'.format(action)] = -np.float('inf')
                    result_mean['{}_agent'.format(action)] = -np.float('inf')
                    for role in metadata.action_roles[action][1:]:
                        action_role_key = '{}_{}'.format(action, role)
                        obj_info = np.array([0.,0.,0.,0.,0.])
                        result_sum[action_role_key] = obj_info
                        result_sum['{}_class'.format(role)] = 0
                        result_max[action_role_key] = obj_info
                        result_max['{}_class'.format(role)] = 0
                        result_mean[action_role_key] = obj_info
                        result_mean['{}_class'.format(role)] = 0
                all_results_sum.append(result_sum)
                all_results_max.append(result_max)
                all_results_mean.append(result_mean)
                continue

            pred_label_sum = np.sum(pred_label[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], part_num:, :], axis=0)
            pred_label_max = np.max(pred_label[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], part_num:, :], axis=0)
            pred_label_mean = np.mean(pred_label[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], part_num:, :], axis=0)
    
            pred_role_sum = np.sum(pred_role[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], part_num:, :], axis=0)
            pred_role_max = np.max(pred_role[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], part_num:, :], axis=0)
            pred_role_mean = np.mean(pred_role[i_item][ np.where(np.equal(part_human_ids[i_item], i_human))[0], part_num:, :], axis=0)

            for action_index, action in enumerate(metadata.action_classes):
                if action == 'none':
                    continue
                
                if len(pred_label_sum) == 0:
                    result_sum['{}_agent'.format(action)] = -np.float('inf')
                else:
                    result_sum['{}_agent'.format(action)] = np.sum(pred_label_sum[:,action_index])
                if len(pred_label_max) == 0:
                    result_max['{}_agent'.format(action)] = -np.float('inf')
                else:
                    result_max['{}_agent'.format(action)] = np.max(pred_label_max[:,action_index])
                if len(pred_label_mean) == 0:
                    result_mean['{}_agent'.format(action)] = -np.float('inf')
                else:
                    result_mean['{}_agent'.format(action)] = np.mean(pred_label_mean[:,action_index])

                for role in metadata.action_roles[action][1:]:
                    role_index = metadata.role_index[role]
                    action_role_key = '{}_{}'.format(action, role)
                    best_score = -np.inf
                    best_j = -1
                    for i_obj in range(obj_num):
                        action_role_score = pred_label_sum[i_obj, action_index] * pred_role_sum[i_obj, role_index]
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
                        action_role_score = pred_label_max[i_obj, action_index] * pred_role_max[i_obj, role_index]
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
                        action_role_score = pred_label_mean[i_obj, action_index] * pred_role_mean[i_obj, role_index]
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
