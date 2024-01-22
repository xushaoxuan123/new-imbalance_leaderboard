#!/bin/bash
python just_loss.py \
--dataset KineticSound \
--model just_loss \
--gpu_ids 1 \
--learning_rate 1e-3 \
--n_classes 31 \
--train \
| tee log_print/just_loss.log