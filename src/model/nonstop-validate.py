import os

name = "new-update-smaller-mat-adaptive-batch-lr1e-5-dropout-0.5"

for epoch in range(80, 200):
    command = "CUDA_VISIBLE_DEVICES=2 python validate.py --node_num 400 --name %s --restore_epoch %d"%(name, epoch)
    print(command)
    os.system(command)
