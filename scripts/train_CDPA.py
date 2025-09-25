import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig
from peft import LoraConfig, get_peft_model
import os
import sys
import logging

from trl.extras.vllm_client import VLLMClient
VLLMClient.HEALTH_CHECK_URL = "/health"


# Setup logging
log_file = "training_run.log"
# Clear old log file on each run
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
    ]
)

logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grm.data.processors import TPO_PROMPT
from scripts.CDPA_trainer import CDPATrainer

def format_prompt(example):
    """Format prompt using TPO_PROMPT template."""
    return {"prompt": TPO_PROMPT.format(preference=example["preference"], question=example["question"])}

def main():
    model_name_or_path = "./models/policy_model"
    train_file = "./data/train_CDPA.json"
    reward_model_name_or_path = "./models/reward_model"
    output_dir = "./outputs/CDPA_model"

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        logging_dir="./logs/CDPA",
        report_to="tensorboard",
        deepspeed="./deepspeed_config_optimized.json",
        learning_rate=1e-6,
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=8,  
        num_train_epochs=2,
        logging_steps=8,
        save_strategy="steps",
        save_steps=8,
        num_generations=4,  
        beta=0.05,
        max_prompt_length=2048,  
        max_completion_length=1024,  
        remove_unused_columns=False,
        bf16=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://127.0.0.1:8000",
        generation_kwargs={
            "temperature": 0.5,
            "top_p": 0.9,
        },
    )


    print("Loading policy model and tokenizer...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,  
        device_map=None,  
        trust_remote_code=True,
        low_cpu_mem_usage=True,  
    )
    policy_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    policy_model = get_peft_model(policy_model, lora_config)
    

    raw_dataset = load_dataset("json", data_files=train_file, split="train")
    column_names = raw_dataset.column_names
    dataset = raw_dataset.map(format_prompt, remove_columns=column_names)
    
    print("Configuring vLLM reward model...")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_or_path, trust_remote_code=True)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    

    class DummyRewardModel:
        def __init__(self):
            self.__name__ = "dummy_reward_vllm"
            pass
        def generate(self, *args, **kwargs):
            raise NotImplementedError("DummyRewardModel should not be used in vLLM mode")
    reward_model = DummyRewardModel()
    

    grpo_trainer = CDPATrainer(
        model=policy_model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=policy_tokenizer,
        reward_funcs=[reward_model],
        reward_processing_classes=[reward_tokenizer],
        use_reward_vllm=True,
        reward_vllm_server_base_url="http://127.0.0.1:8001",
    )
    
    logger.info("🎯 Starting CDPA training...")
    logger.info(f"📊 Training configuration:")
    logger.info(f"  - Dataset size: {len(dataset)}")
    logger.info(f"  - Batch size: {grpo_config.per_device_train_batch_size}")
    logger.info(f"  - Gradient accumulation steps: {grpo_config.gradient_accumulation_steps}")
    logger.info(f"  - Generations per prompt: {grpo_config.num_generations}")
    logger.info(f"  - Learning rate: {grpo_config.learning_rate}")
    logger.info(f"  - Training epochs: {grpo_config.num_train_epochs}")
    
    grpo_trainer.train()
    
    logger.info("🎉 CDPA training completed!")
    grpo_trainer.save_model(output_dir)
    logger.info(f"💾 Model saved to: {output_dir}")

if __name__ == "__main__":
    main()