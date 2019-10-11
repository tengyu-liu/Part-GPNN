# 10.06.2019
## Problem: 
* VGG isn't good enough for my problem. Final feature is too small. 
## Solution:
* Switch back to ResNet-152, compute each segment separately. Maybe get Feng Shi on solving the repeated computation part. Should speed up quite a bit. 

# 10.07.2019
## Idea 
Train .sum and evaluate with .max, see if we get better mAP
* bear:  CUDA_VISIBLE_DEVICES=0 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp0
* bear:  CUDA_VISIBLE_DEVICES=1 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --resume tmp/checkpoints/vcoco/exp1
* bear:  CUDA_VISIBLE_DEVICES=2 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRE --resume tmp/checkpoints/vcoco/exp2
* bear:  CUDA_VISIBLE_DEVICES=3 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --NRE --resume tmp/checkpoints/vcoco/exp3
### Result: No

# 10.09.2019
## Idea
1. Collect inferred part-action pairs and inspect for meaningful statistical patterns in part-action pair frequencies
2. Reject hard (soft) negatives based on part-action pairs in inference/training, hopefully it improves instance-level performance
3. Similar to the tangram model: EM on compositional grammar## Experiment
* camel: CUDA_VISIBLE_DEVICES=0 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp0.pg --model-type PG
* camel: CUDA_VISIBLE_DEVICES=1 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --resume tmp/checkpoints/vcoco/exp1.pg --model-type PG
* camel: CUDA_VISIBLE_DEVICES=2 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRE --resume tmp/checkpoints/vcoco/exp2.pg --model-type PG
* camel: CUDA_VISIBLE_DEVICES=3 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --NRT --NRE --resume tmp/checkpoints/vcoco/exp3.pg --model-type PG
### Result
pending

## Idea
Adjusting `prop-layer` parameter
* bear: CUDA_VISIBLE_DEVICES=0 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp4 --data-root /mnt/hdd-12t/share/v-coco/ --log-root ../../log/vcoco/exp4 --prop-layer 1
* bear: CUDA_VISIBLE_DEVICES=1 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp5 --data-root /mnt/hdd-12t/share/v-coco/ --log-root ../../log/vcoco/exp5 --prop-layer 2
* bear: CUDA_VISIBLE_DEVICES=2 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp6 --data-root /mnt/hdd-12t/share/v-coco/ --log-root ../../log/vcoco/exp6 --prop-layer 3
* bear: CUDA_VISIBLE_DEVICES=3 python vcoco.py --batch-size 1 --prefetch 4 --epochs 100 --extra-feature --resume tmp/checkpoints/vcoco/exp7 --data-root /mnt/hdd-12t/share/v-coco/ --log-root ../../log/vcoco/exp7 --prop-layer 4

## Idea
Suppress part-part edges between different humans