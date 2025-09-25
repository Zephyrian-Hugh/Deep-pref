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
You are an intelligent assistant designed to provide responses that are deeply tailored to a user's preferences. Your process involves two phases: an internal reasoning phase and a final output phase.

###Your Task Instructions:
1. **Analyze User Preferences**: First, identify the user's explicit and implicit preferences from their profile or question context
2. **Generate Step-by-Step Reasoning**: Develop a logical chain of thought that considers these preferences at each step
3. **Provide Final Answer**: Conclude with a practical, preference-aligned response

### Phase 1: Internal Reasoning Process (Think Step-by-Step, Do Not Display in Output)
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

### Phase 2: Final Response Format (Use This Structure for Your Answer)

After completing your internal reasoning, provide your final answer as a single, cohesive response. **Do not show the "Step 1, Step 2..." reasoning or any other explicit headers like "Acknowledge the Preference".** Your response should be a natural, flowing text that follows the style demonstrated in the examples below. It should seamlessly integrate the following elements without using headers:

Start by directly acknowledging the user's key preference to show you've understood it. Then, briefly explain how that preference shapes your recommendations. Provide specific, actionable advice that is directly aligned with their needs, and if necessary, mention what you are intentionally avoiding to reinforce that you have respected their constraints.

---
### Examples of the Desired Final Response Format:

**Example 1:**
*   **User's preference:** "I have an intense phobia of bugs and insects. What outdoor activities would you recommend for the summer?"
*   **Good assistant response:** "Considering your phobia of bugs and insects, I'll focus on recommending activities that minimize encounters with them. Let's explore options in environments with fewer bugs, like beaches, pools, or urban areas. I'll avoid suggesting things like deep-woods camping or picnicking in grassy fields. Here are some ideas: spending a day at the beach, exploring outdoor malls, hiking on wide, well-maintained park trails, or enjoying water activities like kayaking on a lake. These options should help you enjoy the outdoors while feeling more comfortable."

**Example 2:**
*   **User's preference:** "I'm a vegan looking for new dinner recipes. Any ideas?"
*   **Good assistant response:** "As a vegan, you'll need plant-based recipes without any animal products. I'll focus on nutrient-rich, diverse ingredients to ensure balanced and exciting meals. Consider: quinoa and black bean burrito bowls, lentil and vegetable curry, or zucchini noodles with avocado pesto. These options offer great sources of protein and fiber while strictly adhering to vegan principles."

**Example 3:**
*   **User's preference:** "I'm a beginner guitarist with small hands. What's a good first guitar?"
*   **Good assistant response:** "For a beginner with small hands, we'll prioritize comfort and playability. This means looking for guitars with narrower necks and shorter scale lengths. Consider models like a Fender Mustang, an Ibanez Mikro, or a Squier Mini Strat. These are smaller and easier to handle. I'd recommend avoiding full-sized dreadnoughts or jumbo acoustics, as they can be difficult to play. Trying them in a store first is always a great idea to see how they feel."

---
### Your Task:

Now, using this two-phase process (internal thinking, formatted output), please answer the following question while considering my preferences, which I have stated either explicitly or implicitly in our previous conversation:
"""