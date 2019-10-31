import os

name = "new-update-smaller-mat-adaptive-batch-lr1e-5-dropout-0.5"
epoch = sorted(list(set([int(x[:4]) for x in os.listdir(os.path.join(os.path.dirname(__file__), 'models', name)) if x[:4].isnumeric()])))[-1]

command = "CUDA_VISIBLE_DEVICES=2 python validate.py --node_num 400 --name %s --restore_epoch %d"%(name, epoch)

print(command)
os.system(command)
