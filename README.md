# Critique-Driven Reasoning Alignment (CDRA)

This repository contains the implementation for training a **Critique-Driven Reasoning Alignment (CDRA)** system, which combines a Pers-GenPRM with a custom Group Relative Policy Optimization (GRPO) algorithm to improve the reasoning capabilities of large language models.

## Overview

CDRA focuses on improving the quality of model reasoning by:
1. Training a reward model to evaluate reasoning steps
2. Using that reward model to guide policy optimization
3. Providing dense, step-by-step feedback during training

The key innovation is the focus on **process-level optimization** rather than just outcome-based rewards.

## Features

- 🔍 **Process-Level Evaluation**: Evaluates each step of the reasoning process
- 🎯 **Dense Feedback**: Provides token-level rewards for precise optimization
- 🚀 **Efficient Training**: Uses vLLM for high-throughput inference
- 📊 **Comprehensive Metrics**: Tracks multiple quality indicators during training

## Setup and Installation

### Requirements

- Python 3.10+
- CUDA 12.1+
- 4+ GPUs (recommended)

### Installation

```bash
# Clone the repository
git clone CDRA
cd CDRA

# Create and activate a conda environment
conda create -n cdra python=3.10
conda activate cdra

# Install PyTorch 2.6 (CUDA 12.1)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Configuration

Configure accelerate for distributed training:
```bash
accelerate config
```

Or use the provided configurations:
- `accelerate_config.yaml`: Single-node multi-GPU setup
- `accelerate_multigpu_config.yaml`: Multi-node setup
- `deepspeed_config.json`: Basic DeepSpeed configuration
- `deepspeed_config_optimized.json`: Memory-optimized configuration

## Training Pipeline

### 1. Train the Reward Model

First, train the Pers-GenPRM using supervised fine-tuning:

```bash
accelerate launch scripts/train_genprm.py \
    --model_name_or_path ./models/base_model \
    --dataset_name ./data/sft_dataset \
    --output_dir ./outputs/prm_sft
```

### 2. Policy Alignment Training

#### 2.1 Start vLLM Servers

Start two vLLM servers for policy and reward models:

```bash
# Terminal 1: Policy Model Server (GPUs 0,1)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model ./models/policy_model \
    --tensor-parallel-size 2 \
    --port 8000 \
    --trust-remote-code

# Terminal 2: Reward Model Server (GPUs 2,3)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model ./models/reward_model \
    --tensor-parallel-size 2 \
    --port 8001 \
    --trust-remote-code
```

#### 2.2 Run Training

Launch the CDRA training:

```bash
accelerate launch --config_file ./accelerate_config.yaml scripts/train_CDPA.py
```

## Evaluation

The project includes comprehensive evaluation scripts in the `eval/` directory:

```bash
cd eval
bash run_and_eval.sh
```

This will evaluate both zero-shot and chain-of-thought performance using:
- Error type analysis
- Preference following accuracy
- Quality metrics (innovation, depth, thoughtfulness)

## Project Structure

```
CDRA/
├── scripts/                           # Training scripts
│   ├── CDRA_trainer.py               # Custom trainer implementation
│   ├── train_CDPA.py             # Main training script
│   ├── train_genprm.py           # Main training script
│   ├── run_training.sh              # Training launcher
│   ├── start_vllm.sh               # vLLM server startup
│   └── stop_vllm.sh                # vLLM server shutdown
├── eval/                            # Evaluation code
│   ├── generation_task/            # Task-specific evaluation
│   ├── utils/                      # Evaluation utilities
│   └── run_and_eval.sh            # Evaluation launcher
├── grm/                            # Core model code
│   └── data/
│       └── processors.py           # Data processing utilities
├── configs/                        # Configuration files
│   ├── accelerate_config.yaml
│   ├── accelerate_multigpu_config.yaml
│   ├── deepspeed_config.json
│   └── deepspeed_config_optimized.json
└── README.md                       # This file
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the TRL library for the base GRPO implementation
- Thanks to the vLLM team for the efficient inference server