import copy
import re
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Union, Optional
import logging
from collections import defaultdict

import torch
from torch import nn
from transformers import PreTrainedTokenizer, GenerationConfig, Trainer
from transformers.utils import is_flash_attn_2_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator, profiling_context
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import (
    GRPOTrainer,
    gather,
    pad,
    truncate_with_protected_tokens,
    nanmin,
    nanmax,
)
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model, set_seed
logger = logging.getLogger(__name__)

# Instruction prompt for the reward model
INSTRUCTION_PROMPT = (
    "You are a master evaluator of reasoning, known for your critical and discerning judgment. Your task is to assign a precise score based on how well it infers an *implicit* user preference.\n\n"
    "You are famously tough but fair. You have a limited \"budget\" for high scores. A score of 0.9 or higher is reserved for truly exceptional, rare insights. You must use the full range of the scale to create meaningful differentiation between good, great, and brilliant ideas. Avoid clustering scores. **Be decisive and avoid defaulting to 'safe' scores in the 0.7-0.89 range unless they are truly warranted.**"
    "You will be given a user's question, their preferences, and a model's thinking step."
    "1.  **Analyze the Explicit-Implicit Leap:** The primary criterion is the quality of the inference. Does the \"New Thought\" make a creative, plausible, and non-obvious leap from what is stated to what is implied? "
    "2.  **Consider Alternatives (Forced Comparison):** Before scoring, mentally compare this \"New Thought\" to other plausible hypotheses. Is this idea truly more insightful or just one of several good possibilities? This mental check should inform your score. "
    "3.  **Assign a Precise Score:** Provide a score from -1.0 to 1.0 based on the following strict, continuous scale."
    "*   **1.0 (Revolutionary Insight):** A once-in-a-session, brilliant, and entirely non-obvious deduction that reframes the entire problem. This is a hypothesis so profound it feels like a genuine discovery."
    "*   **0.9 - 0.99 (Exceptional Inference):** A deeply insightful and highly creative leap that uncovers a core, hidden user motivation. This is reserved for the top 5% of good ideas."
    "*   **0.7 - 0.89 (Strong, Differentiated Inference):** A well-supported inference that adds significant, actionable nuance. It's a very good idea, but not a game-changer. Use this range to differentiate between solid (0.7) and very strong (0.89) insights."
    "*   **0.4 - 0.69 (Surface-Level Inference):** A safe, logical, but fairly obvious inference. It's helpful but lacks creativity. A 0.69 is a good but unsurprising idea. A 0.4 is a barely-useful observation."
    "*   **0.1 - 0.39 (No Real Inference):** The thought just rephrases the explicit preference or suggests a direct solution without uncovering any deeper meaning. It fails the core task."
    "*   **0.0 (Irrelevant):** The thought is not related to inferring preferences."
    "*   **-0.1 to -0.39 (Flawed Inference):** The thought makes a weak logical leap or a poorly supported assumption. It is unhelpful and slightly misguided."
    "*   **-0.4 to -0.69 (Misaligned Inference):** The thought demonstrates a misunderstanding of the user's explicit preference or proposes something that is clearly not aligned with their goals."
    "*   **-0.7 to -1.0 (Contradictory/Harmful):** The thought makes an illogical inference that directly contradicts the user's stated profile, or it could lead to a harmful or nonsensical recommendation."
    "4.Provide your step-by-step critique after `**Reasoning:**` and a score from -1.0 to 1.0 after `**SCORE:**`. "
    "5.Enclose the entire response in `<|im_start|>` and `<|im_end|>` tags.\n"
    "6.Example: <|im_start|>**Reasoning:**This is a good inference.**SCORE:** [your_score]<|im_end|>\n"
    "Here is the content to critique:"
)


class CDPATrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        # Extract reward vLLM config before calling parent constructor
        self.use_reward_vllm = kwargs.pop('use_reward_vllm', False)
        self.reward_vllm_server_base_url = kwargs.pop('reward_vllm_server_base_url', None)
        
        # Call parent constructor
        super().__init__(*args, **kwargs)

        # Initialize reward model vLLM server URL if enabled
        if self.use_reward_vllm:
            if self.reward_vllm_server_base_url is None:
                raise ValueError("reward_vllm_server_base_url must be provided when use_reward_vllm=True")
            logger.info(f"🚀 Configuring Reward Model vLLM Server: {self.reward_vllm_server_base_url}")

    def _find_subsequence(self, haystack: torch.Tensor, needle: torch.Tensor) -> int:
        """Finds the start index of a subsequence in a sequence."""
        haystack_len, needle_len = haystack.size(0), needle.size(0)
        if needle_len == 0:
            return 0
        if needle_len > haystack_len:
            return -1
        for i in range(haystack_len - needle_len + 1):
            if torch.equal(haystack[i : i + needle_len], needle):
                return i
        return -1

    def _split_completion_into_steps(self, completion: str) -> list[str]:
        """Helper function to split completions based on 'Step D:' markers."""
        step_pattern = r"Step (\d+):"
        step_matches = list(re.finditer(step_pattern, completion))
        
        if not step_matches:
            logger.info("No Step markers found, returning entire completion")
            return [completion.strip()]
        
        logger.info(f"Found {len(step_matches)} Step markers")
        
        # Find consecutive Step 1, 2, 3, 4, 5 sequence
        consecutive_steps = []
        expected_step = 1
        
        for match in step_matches:
            step_num = int(match.group(1))
            if step_num == expected_step:
                consecutive_steps.append(match)
                expected_step += 1
                if len(consecutive_steps) == 5:  # We only want first 5 consecutive steps
                    break
            elif step_num == 1 and len(consecutive_steps) == 0:
                # If we encounter a new Step 1, restart
                consecutive_steps = [match]
                expected_step = 2
        
        if not consecutive_steps:
            logger.info("No consecutive Step sequence found, using first 5 Step markers")
            consecutive_steps = step_matches[:5]
        
        logger.info(f"Using {len(consecutive_steps)} consecutive steps")
        
        steps = []
        for i, match in enumerate(consecutive_steps):
            start_pos = match.start()
            if i + 1 < len(consecutive_steps):
                end_pos = consecutive_steps[i + 1].start()
            else:
                # For the last step, find a reasonable endpoint
                end_pos = len(completion)
                # If there are other Step markers, end before them
                for remaining_match in step_matches[len(consecutive_steps):]:
                    if remaining_match.start() > start_pos:
                        end_pos = remaining_match.start()
                        break
            
            step_text = completion[start_pos:end_pos].strip()
            steps.append(step_text)
        
        # Add debug logs
        logger.info(f"Split completion into {len(steps)} steps")
        for i, step in enumerate(steps):
            logger.info(f"  Step {i+1}: {step[:100]}..." if len(step) > 100 else f"  Step {i+1}: {step}")
        
        return steps

    def _get_generative_rewards(
        self,
        prompts: List[str],
        completions_text: List[str],
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates rewards by generating critiques and scores for each step, then
        assigns them at the token level.
        """
        device = self.accelerator.device
        all_per_token_rewards = []

        reward_func = self.reward_funcs[0]
        reward_tokenizer = self.reward_processing_classes[0]

        # Ensure the special token used for stopping is in the tokenizer
        if "<|im_end|>" not in reward_tokenizer.vocab:
            reward_tokenizer.add_tokens(["<|im_end|>"], special_tokens=True)
        
        logger.info(f"Starting reward calculation for {len(completions_text)} completions...")

        for i in range(len(prompts)):
            prompt = prompts[i]
            completion_txt = completions_text[i]
            per_token_rewards = torch.zeros_like(completion_ids[i], dtype=torch.float, device=device)
            
            logger.info(f"Processing completion {i+1}/{len(prompts)}:")
            logger.info(f"  Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"  Prompt: {prompt}")
            logger.info(f"  Response: {completion_txt[:300]}..." if len(completion_txt) > 300 else f"  Response: {completion_txt}")
            
            steps = self._split_completion_into_steps(completion_txt)

            if not steps:
                logger.info(f"  No step markers found, skipping reward calculation")
                all_per_token_rewards.append(per_token_rewards)
                continue

            # Prepare inputs for the generative reward model for all steps in a completion
            reward_model_inputs = []
            context = ""
            for step in steps:
                user_content = str(prompt) + context + step
                full_user_content = f"{INSTRUCTION_PROMPT}\n\n{user_content}"
                messages = [{"role": "user", "content": full_user_content}]
                input_text = reward_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                reward_model_inputs.append(input_text)
                context += step + "\n" # Update context for next step

            logger.info(f"  Evaluating {len(steps)} steps with reward model...")
            
            # Use vLLM for reward model inference if enabled
            if self.use_reward_vllm:
                # Gather all reward model inputs across processes for main process generation
                all_reward_model_inputs = gather_object(reward_model_inputs)
                
                if self.accelerator.is_main_process:
                    logger.info(f"  Using vLLM for reward model inference, processing {len(all_reward_model_inputs)} inputs")
                    # Generate using reward model vLLM server with direct HTTP API
                    import requests
                    responses = []
                    
                    with profiling_context(self, "RewardvLLM.generate"):
                        for prompt in all_reward_model_inputs:
                            try:
                                response = requests.post(
                                    f"{self.reward_vllm_server_base_url}/v1/completions",
                                    json={
                                        "model": "reward_model",
                                        "prompt": prompt,
                                        "max_tokens": 1024,
                                        "temperature": 0.0,
                                        "stop": ["<|im_end|>"]
                                    },
                                    timeout=60
                                )
                                response.raise_for_status()
                                result = response.json()
                                text = result["choices"][0]["text"]
                                responses.append(text)
                            except Exception as e:
                                logger.warning(f"Reward model vLLM inference failed: {e}")
                                responses.append("")  # Fallback to empty response
                else:
                    responses = [None] * len(all_reward_model_inputs)
                
                # Broadcast results from main process to all processes
                responses = broadcast_object_list(responses, from_process=0)
                
                # Extract the portion corresponding to current process
                process_slice = slice(
                    self.accelerator.process_index * len(reward_model_inputs),
                    (self.accelerator.process_index + 1) * len(reward_model_inputs),
                )
                responses = responses[process_slice]
            else:
                # Fallback to traditional inference
                tokenized_inputs = reward_tokenizer(
                    reward_model_inputs, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                
                with torch.inference_mode():
                    outputs = reward_func.generate(
                        **tokenized_inputs,
                        max_new_tokens=512,
                        do_sample=False, # Use greedy decoding for consistency
                        eos_token_id=reward_tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        pad_token_id=reward_tokenizer.eos_token_id,
                    )
                    # Only decode the generated part (response), ignore input prompt
                    input_length = tokenized_inputs['input_ids'].shape[1]
                    response_ids = outputs[:, input_length:]
                    responses = reward_tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            step_scores = []
            logger.info(f"  Reward model scoring results (response part only):")
            for j, response in enumerate(responses):
                score_match = re.search(r"\*\*SCORE:\*\*\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        step_scores.append(score)
                        logger.info(f"    Step {j+1}: Score {score:.3f}")
                        logger.info(f"    Full evaluation: {response}")
                    except (ValueError, TypeError):
                        step_scores.append(0.0) # Default score if parsing fails
                        logger.info(f"    Step {j+1}: Score parsing failed, using default score 0.0")
                else:
                    step_scores.append(0.0) # Default score if marker is not found
                    logger.info(f"    Step {j+1}: No score marker found, using default score 0.0")
                    logger.info(f"    Original response: {response}")
            
            # Ensure step_scores matches steps length
            while len(step_scores) < len(steps):
                step_scores.append(0.0)
                logger.warning(f"    Adding default score 0.0 to match step count")
            
            avg_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
            logger.info(f"  Average score for this completion: {avg_score:.3f}")

            step_token_ids = [self.processing_class(step, add_special_tokens=False)["input_ids"] for step in steps]

            for j, step_id_list in enumerate(step_token_ids):
                # Add boundary check
                if j >= len(step_scores):
                    logger.warning(f"    Step index {j} exceeds score list range {len(step_scores)}, skipping")
                    continue
                    
                step_ids_tensor = torch.tensor(step_id_list, device=device)
                start_idx = self._find_subsequence(completion_ids[i], step_ids_tensor)
                if start_idx != -1:
                    end_idx = start_idx + len(step_ids_tensor)
                    # Assign the score of this step to all its tokens
                    per_token_rewards[start_idx:end_idx] = step_scores[j]

            all_per_token_rewards.append(per_token_rewards)

        return pad(all_per_token_rewards, padding_value=0.0)

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            # The chat template inserts a single image token into the prompt text. However, when this text is later
            # tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original template.
            if self.image_token is not None:
                prompts_text = [
                    re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text) for text in prompts_text
                ]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.args.vllm_guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.args.vllm_guided_decoding_regex:
                    from vllm.sampling_params import GuidedDecodingParams
                    guided_decoding = GuidedDecodingParams(regex=self.args.vllm_guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                }
                if guided_decoding is not None:
                     generation_kwargs["guided_decoding_backend"] = "outlines"
                     generation_kwargs["guided_decoding"] = guided_decoding
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                from vllm import SamplingParams
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        elif self.use_transformers_paged:
             raise NotImplementedError("Paged generation is not supported in this custom trainer.")
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                self.accelerator.unwrap_model(self.model).summon_full_params(self.model, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]
        
        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        
        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                pixel_values=prompt_inputs.get("pixel_values"),
                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                image_sizes=prompt_inputs.get("image_sizes"),
            )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None
        
        # Process supervision modification
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        logger.info(f"Decoded {len(completions_text)} completion texts")
        
        # Here we call our custom generative reward function to get token-level rewards
        token_rewards = self._get_generative_rewards(prompts_text, completions_text, completion_ids)
        
        # Gather rewards across all processes and normalize them per group
        gathered_token_rewards = gather(token_rewards)
        grouped_rewards = gathered_token_rewards.view(
            -1, self.num_generations, gathered_token_rewards.size(-1)
        )
        
        # Normalize rewards at the token level
        mean_rewards = grouped_rewards.mean(dim=1, keepdim=True)
        std_rewards = grouped_rewards.std(dim=1, keepdim=True)
        # Avoid division by zero
        advantages = (grouped_rewards - mean_rewards) / (std_rewards + 1e-8)
        advantages = advantages.view(-1, gathered_token_rewards.size(-1))
        
        # Logging metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = (
            1 - len(term_completion_lengths) / len(agg_completion_lengths) if len(agg_completion_lengths) > 0 else 0.0
        )
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        mean_reward = gathered_token_rewards.mean().item()
        std_reward = gathered_token_rewards.std().item()
        self._metrics[mode]["reward/token_mean"].append(mean_reward)
        self._metrics[mode]["reward/token_std"].append(std_reward)
        
        std_rewards_per_group = grouped_rewards.std(dim=1)
        is_std_zero = torch.isclose(std_rewards_per_group, torch.zeros_like(std_rewards_per_group))
        self._metrics[mode]["reward/frac_zero_std"].append(is_std_zero.float().mean().item())
        
        logger.info(f"Token-level reward stats: mean={mean_reward:.4f}, std={std_reward:.4f}")
        
        # Slice advantages for the current process
        process_slice = slice(
            self.accelerator.process_index * len(prompts), 
            (self.accelerator.process_index + 1) * len(prompts)
        )
        per_token_advantages = advantages[process_slice]

        # Log completion texts and rewards for debugging
        if self.log_completions:
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))
            
            # Since per-token rewards are too verbose, log mean reward per completion
            mean_rewards_per_completion = token_rewards.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
            self._logs["rewards"]["process_mean_reward"] = gather_object(mean_rewards_per_completion.tolist())

            # Log mean advantage per completion
            mean_advantages_per_completion = per_token_advantages.sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
            self._logs["advantages"] = gather_object(mean_advantages_per_completion.tolist())
            
            if has_images:
                self._logs["image"].extend(gather_object(images))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": per_token_advantages,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output["image_sizes"] = prompt_inputs["image_sizes"]
        return output

    def _compute_loss(
        self, model, inputs, return_outputs=False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if self.use_liger_loss:
            raise NotImplementedError("Process Supervision with Liger Loss is not supported.")

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1),
            torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1),
            inputs["completion_ids"].size(1),
            compute_entropy=True,  # Enable entropy computation for metrics
        )

        # Apply entropy mask if configured (useful for process supervision)
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, inputs["completion_mask"], 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        advantages = inputs["advantages"]  # This is now (batch_size, seq_len)
        
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps
        
        # For Process Supervision, we use token-level importance sampling
        # Note: Process supervision naturally works at token level, so we don't need
        # the complex importance_sampling_level logic from base GRPO
        log_importance_weights = log_ratio  # Token-level by default for process supervision
        
        # The multiplication is now element-wise, NO BROADCASTING needed for advantage.
        coef_1 = torch.exp(log_importance_weights)
        per_token_loss1 = coef_1 * advantages
        clipped_ratio = torch.clamp(torch.exp(log_ratio), 1 - self.epsilon_low, 1 + self.epsilon_high)
        coef_2 = clipped_ratio
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # Apply entropy mask if configured
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        completion_mask = inputs["completion_mask"]
        
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        
        # Log metrics
        mode = "train" if self.model.training else "eval"
        
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        
        def masked_batch_mean(x):
            # Adapt for Process Supervision's token-level advantages
            return (x * completion_mask).sum() / completion_token_count
        
        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())
        
        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())
        
        # Compute the clipped probability ratios
        # Note: We need to adjust this for token-level advantages
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped
        
        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())
        
        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Overrides the log method to ensure metrics from _metrics dictionary are recorded to TensorBoard.
        This implementation is adapted from the base GRPOTrainer and includes a check for the main process
        to prevent race conditions in multi-GPU setups.
        """
        mode = "train" if self.model.training else "eval"
        
        # Average the metrics
        if self._metrics[mode]:
            metrics_to_log = {key: sum(value) / len(value) for key, value in self._metrics[mode].items() if value}
            
            # In evaluation mode, prefix metrics with "eval_" to match Trainer's behavior
            if mode == "eval":
                metrics_to_log = {f"eval_{key}": val for key, val in metrics_to_log.items()}
                
            logs.update(metrics_to_log)

        # CRITICAL: Only log on the main process to prevent file corruption and redundant logging
        if self.is_world_process_zero():
            super().log(logs, start_time=start_time)
        
        # Clear the metrics for the next logging interval
        if self._metrics[mode]:
            self._metrics[mode].clear()