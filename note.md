# 10.06.2019
## Problem: 
* VGG isn't good enough for my problem. Final feature is too small. 
## Solution:
* Switch back to ResNet-152, compute each segment separately. Maybe get Feng Shi on solving the repeated computation part. Should speed up quite a bit. 

# 10.07.2019
## Experiment
### Change: Train .sum and evaluate with .max, see if we get better mAP
* bear:  CUDA_VISIBLE_DEVICES=0 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp0
* bear:  CUDA_VISIBLE_DEVICES=1 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --resume tmp/checkpoints/vcoco/exp1
* bear:  CUDA_VISIBLE_DEVICES=2 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRE --resume tmp/checkpoints/vcoco/exp2
* bear:  CUDA_VISIBLE_DEVICES=3 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --NRE --resume tmp/checkpoints/vcoco/exp3
* camel: CUDA_VISIBLE_DEVICES=0 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp0.2 --V2
* camel: CUDA_VISIBLE_DEVICES=1 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --resume tmp/checkpoints/vcoco/exp1.2 --V2
* camel: CUDA_VISIBLE_DEVICES=2 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRE --resume tmp/checkpoints/vcoco/exp2.2 --V2
* camel: CUDA_VISIBLE_DEVICES=3 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --NRE --resume tmp/checkpoints/vcoco/exp3.2 --V2

## New Implementation:
