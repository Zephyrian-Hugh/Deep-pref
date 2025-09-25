#!/bin/bash

# Evaluate Chain-of-Thought (CoT) performance
echo "Running Chain-of-Thought (CoT) evaluation..."

# Error type evaluation for CoT
python generation_task/llm_based_evaluation_errortypes.py \
    --model qwen_grpo \
    --task cot \
    --topic test300 \
    --pref_form explicit

# Preference following accuracy for CoT
python generation_task/get_preference_following_accuracy_generation_task.py \
    --model qwen_grpo \
    --task cot \
    --topic test300 \
    --pref_form explicit

# Evaluate Zero-shot performance
echo "Running Zero-shot evaluation..."

# Error type evaluation for zero-shot
python generation_task/llm_based_evaluation_errortypes.py \
    --model qwen_grpo \
    --task zero-shot \
    --topic test300 \
    --pref_form explicit

# Preference following accuracy for zero-shot
python generation_task/get_preference_following_accuracy_generation_task.py \
    --model qwen_grpo \
    --task zero-shot \
    --topic test300 \
    --pref_form explicit