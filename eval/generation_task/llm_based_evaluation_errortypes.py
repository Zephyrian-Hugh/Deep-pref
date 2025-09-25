import os
import json
import time
import warnings
import argparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common_utils import load_config


def parse_explanation_and_answer(input_string):
    soup = BeautifulSoup(input_string, "html.parser")
    explanation_tag = soup.find("explanation")
    explanation = explanation_tag.text.strip() if explanation_tag else ""
    answer_tag = soup.find("answer")
    answer = answer_tag.text.strip() if answer_tag else ""
    return explanation, answer


def parse_preference_and_answer(input_string):
    soup = BeautifulSoup(input_string, "html.parser")
    preference_tag = soup.find("preference")
    preference = preference_tag.text.strip() if preference_tag else ""
    answer_tag = soup.find("answer")
    answer = answer_tag.text.strip() if answer_tag else ""
    return preference, answer


def generate_message(model_id, system_prompt, messages, max_tokens, max_retries=20):
    payload = {
        "model": model_id,
        "stream": False,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post("YOUR_API_ENDPOINT", headers=headers, json=payload)
            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
                retries += 1
                time.sleep(1)
                continue

            try:
                response_json = response.json()
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0].get("message", {}).get("content", "")
                    return {"content": [{"text": content}]}
                else:
                    print(f"Unexpected response format: {response_json}")
                    retries += 1
                    time.sleep(1)
                    continue
            except json.JSONDecodeError:
                print(f"Failed to decode JSON response: {response.text}")
                retries += 1
                time.sleep(1)
                continue

        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"Request error: {e}, retrying: {retries}")
            time.sleep(1)

    print("Failed to get response after max retries.")
    return {"content": [{"text": ""}]}


def main():
    exp_configs = load_config("../config.yaml")

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model", type=str, default="claude3s")
    parser.add_argument("--topic", type=str, default="test")
    parser.add_argument(
        "--task", type=str, default="zero-shot", choices=["zero-shot", "cot"]
    )
    parser.add_argument("--pref_form", type=str, default="explicit", choices=["explicit"])
    args = parser.parse_args()
    args.dir = exp_configs["dir_path"]

    try:
        # Load evaluator model
        model_id = "gpt-4o"  # Replace with your model ID
        max_tokens = 1024

        base_generation_file = f"{args.model}_{args.topic}.json"
        args.dir_name = (
            f"{args.dir}/benchmark_results/{args.pref_form}/generation_results/{args.task}/{args.topic}/"
        )
        generation_file = f"{args.dir_name}{base_generation_file}"
        topic_data_path = f"{args.dir}/benchmark_dataset/{args.pref_form}_preference/{args.topic}.json"
        save_file = f"{args.dir_name}error_{args.model}_{args.topic}.json"

        # Load topic data
        with open(topic_data_path, "r") as f:
            topic_data = json.load(f)
        num_topic_data = len(topic_data)

        # Load generation data
        with open(generation_file, "r") as f:
            generation_data = json.load(f)

        # Validate response count
        response_cnt = sum("response_to_q" in d for d in generation_data)
        if response_cnt != num_topic_data:
            warnings.warn(
                f"Warning: The number of generated responses ({response_cnt}) does not match the expected topic data count ({num_topic_data})."
            )
            print(f"Generated responses count: {response_cnt}, Expected: {num_topic_data}")

        # Load existing evaluation data or initialize it
        if os.path.isfile(save_file):
            with open(save_file, "r") as f:
                existing_eval_data = json.load(f)
        else:
            existing_eval_data = generation_data

        # Evaluation loop
        for task_id, task in enumerate(tqdm(existing_eval_data)):
            if "response_to_q" not in task:
                print(f"This task does not have a response yet (Task ID: {task_id})")
                continue

            print(f"Evaluating {task_id}/{num_topic_data}; from file {save_file}")

            if "evaluation_error_analysis" in task:
                analysis = task["evaluation_error_analysis"]
                if all(key in analysis for key in [
                    "acknow", "violate", "hallucinate", "helpful",
                    "innovative", "misleading", "deep_mining", "thoughtful"
                ]):
                    print("Already finished evaluating task id", task_id)
                    continue

            preference = task["preference"]
            question = task["question"]
            end_generation = task["response_to_q"]
            system_prompt = "You are a helpful assistant in evaluating an AI assistant's response. You should be fair and strict and follow the user's instruction"

            BASE_DIR = f"{args.dir}/error_type"
            file_dict = {
                "acknow": "check_acknowledge.txt",
                "violate": "check_violation.txt",
                "hallucinate": "check_hallucination.txt",
                "helpful": "check_helpful.txt",
                "innovative": "check_innovative.txt",
                "misleading": "check_misleading.txt",
                "deep_mining": "check_deep_mining.txt",
                "thoughtful": "check_thoughtful.txt",
            }

            eval_message_texts = []
            for metric_name, file_name in file_dict.items():
                file_path = os.path.join(BASE_DIR, file_name)
                with open(file_path, "r") as f:
                    eval_message_texts.append([metric_name, f.read()])

            if "evaluation_error_analysis" in task:
                error_check = task["evaluation_error_analysis"]
            else:
                error_check = {}

            for idx, (metric, eval_message_text) in enumerate(eval_message_texts):
                if metric in error_check:
                    continue
                if metric == "acknow":
                    eval_message_text = eval_message_text.replace("{end_generation}", end_generation)
                    eval_message_text = eval_message_text.replace("{question}", question)
                elif metric in ["violate", "helpful", "innovative", "misleading", "deep_mining", "thoughtful"]:
                    eval_message_text = eval_message_text.replace("{preference}", preference)
                    eval_message_text = eval_message_text.replace("{question}", question)
                    eval_message_text = eval_message_text.replace("{end_generation}", end_generation)
                elif metric == "hallucinate":
                    extracted_pref = error_check["acknow"]["extract_pref"]
                    eval_message_text = eval_message_text.replace("{preference}", preference)
                    eval_message_text = eval_message_text.replace("{assistant_restatement}", extracted_pref)

                eval_message = [{"role": "user", "content": eval_message_text}]
                eval_response = generate_message(model_id, system_prompt, eval_message, max_tokens)["content"][0]["text"]
                error_check[metric] = {}

                if metric != "acknow":
                    explanation, answer = parse_explanation_and_answer(eval_response)
                    error_check[metric]["explanation"] = explanation
                    error_check[metric]["answer"] = answer
                else:
                    extract_preference, answer = parse_preference_and_answer(eval_response)
                    error_check[metric]["answer"] = answer
                    error_check[metric]["extract_pref"] = extract_preference

            task["evaluation_error_analysis"] = error_check
            existing_eval_data[task_id] = task
            with open(save_file, "w") as outfile:
                json.dump(existing_eval_data, outfile, indent=4)

        print("Done evaluating:", save_file)
    except Exception as err:
        print(f"An error occurred: {err}")


if __name__ == "__main__":
    main()