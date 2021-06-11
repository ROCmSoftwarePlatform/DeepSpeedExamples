#!/usr/bin/env bash
# DeepSpeed Profiling FlopsProfiler Examples

##################################################################################################################
# Training Example: non-zero profile-step enables and instructs the global step to FlopsProfiling in training loop
# flops_train_vision.py
#   Demo program to use FlopsProfiler on top of microbenchmarking script for torchvision training with ROCm.
#   To execute:
#      `python flops_train_vision.py --network <network name> [--profile-step <step_num>] [--batch-size <batch size> ] \
#	                                                      [--iterations <number of iterations>] [--fp16 <0 or 1> ] \
#                                                             [--dataparallel|--distributed_dataparallel] \
#                                                             [--device_ids <comma separated list (no spaces) of GPU indices (0-base) to run dataparallel/distributed_dataparallel>]`
#      Possible network names are: `alexnet`, `densenet121`, `inception_v3`, `resnet50`, `resnet101`, `SqueezeNet`, `vgg16` etc.
#
#      Defaults are profile step 0, training iterations 20, `fp16` off (i.e., 0), and a batch size of 128.
#      Set non-zero profile step to turn on FlopsProfiling.
#
#      `--distributed_dataparallel` will spawn multiple sub-processes and adjust world_size and rank accordingly. Py3.6 ONLY.

python flops_train_vision.py --network resnet50 --amp-opt-level=2 --batch-size=256 --iterations=20 --profile-step=10


######################################
# Inference Example use (another case)
# python flops_inference.py

