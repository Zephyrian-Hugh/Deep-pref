from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import logging
import tensorboard
import sys
import os
import torch
from safetensors.torch import save_file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grm.data.processors import format_sft_chat_examples


class LoggingSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.logger = logging.getLogger('training_log')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            log_file = os.path.join(self.args.output_dir, 'training_predictions.log')
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - \n%(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        if self.args.local_rank in [-1, 0] and self.state.global_step % self.args.logging_steps == 0:
            try:
                if torch.isnan(loss):
                    input_ids = inputs.get("input_ids")[0] 
                    labels = inputs.get("labels")[0]
                    decoded_inputs = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                    log_message = (
                        f"---- NaN Loss Detected (Global Step: {self.state.global_step}) ----\n"
                        f"Loss: {loss.item()}\n\n"
                        f"**Potential Cause**: The response template was likely truncated from the input sequence.\n"
                        f"**Max Sequence Length**: {self.max_seq_length}\n\n"
                        f"**Raw Input IDs**:\n{input_ids.tolist()}\n\n"
                        f"**Decoded Raw Input**:\n'{decoded_inputs}'\n\n"
                        f"**Raw Labels (Note: -100 indicates masked tokens)**:\n{labels.tolist()}\n"
                        f"---------------------------------------------------\n"
                    )
                    self.logger.warning(log_message)
                else:
                    logits = outputs.get("logits")
                    labels = inputs.get("labels").clone()
                    prediction_ids = logits.argmax(dim=-1)

                    true_completion_ids = labels.clone()
                    true_completion_ids[true_completion_ids == -100] = self.tokenizer.pad_token_id
                    decoded_true_completion = self.tokenizer.batch_decode(true_completion_ids, skip_special_tokens=True)[0]

                    prediction_completion_ids = torch.full_like(prediction_ids, self.tokenizer.pad_token_id)
                    prediction_completion_ids[labels != -100] = prediction_ids[labels != -100]
                    decoded_prediction_completion = self.tokenizer.batch_decode(prediction_completion_ids, skip_special_tokens=True)[0]

                    log_message = (
                        f"---- Training Step Log (Global Step: {self.state.global_step}) ----\n"
                        f"Loss: {loss.item():.4f}\n\n"
                        f"**True Completion (Target for loss)**:\n{decoded_true_completion}\n\n"
                        f"**Model's Predicted Completion**:\n{decoded_prediction_completion}\n"
                        f"---------------------------------------------------\n"
                    )
                    self.logger.info(log_message)
            except Exception as e:
                self.logger.error(f"Error during logging: {e}")

        return (loss, outputs) if return_outputs else loss

def train_generative_grm_sft():
    model_name = "Qwen2.5-7B-Instruct"
    grm_model_output_dir = "./outputs/genprm"
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    

    def chat_formatting_func(example):

        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    
 
    raw_dataset = load_dataset("json", data_files={"train": "/prm_dataset/train_data.json", "eval": "prm_dataset/test_data.json"})
    

    train_dataset = raw_dataset["train"].map(
        format_sft_chat_examples, 
        batched=True, 
        remove_columns=raw_dataset["train"].column_names
    )
    eval_dataset = raw_dataset["eval"].map(
        format_sft_chat_examples, 
        batched=True, 
        remove_columns=raw_dataset["eval"].column_names
    )

    response_template = "<|im_start|>assistant\n" 
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=grm_model_output_dir,
        per_device_train_batch_size=4,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        deepspeed="deepspeed_config.json",
        logging_dir="./logs/genprm",
        report_to="tensorboard",
        bf16=True,                
        bf16_full_eval=True,      
    )

    trainer = LoggingSFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=data_collator,
        max_seq_length=8192,
        formatting_func=chat_formatting_func,
        dataset_text_field="messages",
    )


    trainer.train()
    trainer.save_model(grm_model_output_dir)
    

if __name__ == "__main__":
    train_generative_grm_sft()
