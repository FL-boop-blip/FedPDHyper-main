#!/bin/bash
# 定义要运行的方法列表


methods=('FedPDHyper')

for method in "${methods[@]}"; do
    python train.py --method "$method" \
        --dataset CIFAR10 \
        --comm-rounds 300 \
        --model ResNet18 \
        --batchsize 50 \
        --local-learning-rate 0.1 \
        --local-epochs 5 \
        --non-iid \
        --lamb 0.1 
done


