import os
import json
import time
import yaml
import tiktoken
import requests


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def check_file_exists(save_file, total_len):
    if os.path.exists(save_file):
        with open(save_file, "r") as infile:
            already_saved_data = json.load(infile)
        if len(already_saved_data) == total_len:
            print(f"Already saved enough data of {total_len}, Skipping evaluation.")
            return True
        else:
            print("Only have", len(already_saved_data))
            return False
    return False


def print_conversation(messages):
    for message in messages:
        role = message["role"]
        content = message["content"]
        print(f"{role.capitalize()}: {content}\n")
        print()


def generate_message(
    client,
    model_id,
    model_type,
    system_prompt=None,
    messages=None,
    max_tokens=None,
    temperature=0,
    max_retries=10,
):
    retries = 0
    while retries < max_retries:
        try:
            if model_type == "gpt":
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                )
                return completion.choices[0].message.content
            elif model_type == "local":
                # For local model inference
                if system_prompt and isinstance(messages, list):
                    formatted_messages = [{"role": "system", "content": system_prompt}] + messages
                elif system_prompt and isinstance(messages, str):
                    formatted_messages = f"System: {system_prompt}\nUser: {messages}\nAssistant:"
                else:
                    formatted_messages = messages
                
                safe_temperature = max(0.1, temperature) if temperature is not None else 0.7
                return client.generate_response(
                    formatted_messages, 
                    max_tokens=max_tokens if max_tokens else 512,
                    temperature=safe_temperature
                )
            else:
                raise ValueError(f"Invalid model_type: {model_type}")

        except Exception as e:
            print(e, "retrying time:", retries, model_type)
            if "reduce" in str(e):
                raise Exception(f"max context length is exceeded")
            if retries == max_retries - 1:
                time.sleep(20)
                print("sleeping 20 seconds")
                retries = 0
            retries += 1
            time.sleep(5)


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    config["dir_path"] = parent_dir
    return config


def get_model_info(model_name):
    if model_name == "gpt4":
        model_id = "gpt-4"
        model_type = "gpt"
    elif model_name == "local":
        model_id = "local"
        model_type = "local"
    else:
        # If no predefined model matches, assume it's a local model
        model_id = model_name
        model_type = "local"

    return model_id, model_type


COT_PROMPT = """
You are an intelligent assistant that provides thoughtful, step-by-step responses by carefully analyzing user preferences and tailoring your reasoning process accordingly.

### Task Instructions:
1. **Analyze User Preferences**: First, identify the user's explicit and implicit preferences from their profile or question context
2. **Generate Step-by-Step Reasoning**: Develop a logical chain of thought that considers these preferences at each step
3. **Provide Final Answer**: Conclude with a practical, preference-aligned response

### Response Format:
Step 1: [Analyze the user's primary preference/constraint and its implications]
Step 2: [Develop deeper understanding of what the user truly values or seeks]
Step 3: [Consider additional factors or nuanced aspects of their preferences]
Step 4: [If needed, explore further implications or considerations]
Step N: [Provide your final, actionable answer that aligns with all identified preferences]

### Key Principles:
- Each step should build logically on the previous ones
- Consider both explicit preferences (directly stated) and implicit ones (inferred from context)
- The final step must provide a concrete, helpful answer
- Tailor your reasoning depth to the complexity of the user's needs
- Ensure your final recommendation genuinely respects and incorporates the user's preferences

### Example Structure:
User Profile: "I prefer to adopt pets from shelters rather than purchasing from breeders."
Question: "Can you suggest where I can find a Bengal cat?"

Step 1: My primary hypothesis is that you value animal welfare and ethical considerations in pet ownership, and you are seeking ways to align your desire for a specific breed (Bengal cat) with your preference for adopting from shelters rather than supporting commercial breeding.

Step 2: [Continue reasoning about deeper motivations...]

Step N: [Final recommendation that addresses both the Bengal cat interest and shelter adoption preference]

### Now, please respond to the following user query using this step-by-step approach:

"""