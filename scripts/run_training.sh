#!/bin/bash

# Configure NCCL for better compatibility
export CUDA_VISIBLE_DEVICES=2,3
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export no_proxy=localhost,127.0.0.1

# Launch training with accelerate
accelerate launch \
  --config_file ./accelerate_config.yaml \
  ./scripts/train_CDPA.py