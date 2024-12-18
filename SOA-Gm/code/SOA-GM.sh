#!/bin/bash
python SOAGM.py \
--dataset CREMAD \
--model MMPareto \
--gpu_ids 4 \
--n_classes 6 \
--train \
--batch_size 64 \
--epochs 50 \
--learning_rate 0.002 \
--lr_decay_step 50 \
--optimizer sgd \
--random_seed 0 \
| tee log_print/SOAGM.log