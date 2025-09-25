from datasets import Dataset

step_separator = "\n\n\n\n\n"


TPO_PROMPT = """
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

User Profile: {preference}
Question: {question}
"""

def format_sft_chat_examples(examples):

    all_formatted_messages = []
    
    # examples是一个字典，key是列名，value是该列在一个batch中的数据列表
    for i in range(len(examples["question"])):
        preference = examples["preference"][i]
        question = examples["question"][i]
        process = examples["process"][i]
        labels = examples["label"][i]

        initial_user_prompt = TPO_PROMPT.format(preference=preference, question=question)

        steps = process.split(step_separator)
        valid_steps = [s.strip() for s in steps if s.strip()]

        if len(valid_steps) != len(labels):
            continue

        context = ""
        for j, step in enumerate(valid_steps):
            user_content = initial_user_prompt + context + step + step_separator
            
            critique = labels[j]['critique']
            score = labels[j]['score']

            assistant_content = critique
            instruction_prompt = (
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
   
            full_user_content = f"{instruction_prompt}\n\n{user_content}"
            messages = [
                {"role": "user", "content": full_user_content},
                {"role": "assistant", "content": assistant_content}
            ]
            all_formatted_messages.append(messages)


            context += step + step_separator + "\n"


    return {"messages": all_formatted_messages}

