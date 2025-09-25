import json
import os

INSTRUCTION = """
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

OUTPUT = {
    "instruction": INSTRUCTION,
    "input": "",
    "output": ""
}


def process_files():
    data_dir = '/data/hzy_data/data/test'
    output_file = '/data/hzy_data/data/sft_test_data_all.json'
    
    all_processed_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for item in data:
                    # best_response = max(item['response_list'], key=lambda x: x.get('score', float('-inf')))
                    for best_response in item['response_list']:
                        output = OUTPUT.copy()
                        output["input"] = f"user_profile: {item['user_profile']}\ninitial_question: {item['initial_question']}"
                        output["output"] = "\n".join(best_response['response'])
                        all_processed_data.append(output)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    process_files()