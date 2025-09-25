import os
import json
import argparse
import yaml
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common_utils import load_config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluation Setup")
    parser.add_argument("--model", type=str, default="claude3s")
    parser.add_argument("--topic", type=str, default="test")
    parser.add_argument(
        "--task", type=str, default="zero-shot", choices=["zero-shot", "cot"]
    )
    parser.add_argument("--pref_form", type=str, default="explicit", choices=["explicit"])
    return parser.parse_args()


def setup_paths(args, exp_configs):
    base_file = f"{args.model}_{args.topic}.json"
    args.dir_name = (
        f"{exp_configs['dir_path']}/benchmark_results/{args.pref_form}/generation_results/{args.task}/{args.topic}/"
    )
    generation_file = f"{args.dir_name}{base_file}"
    topic_data_path = f"{exp_configs['dir_path']}/benchmark_dataset/{args.pref_form}_preference/{args.topic}.json"
    eval_file = f"{args.dir_name}error_{args.model}_{args.topic}.json"
    return generation_file, topic_data_path, eval_file


def load_evaluation_data(eval_file):
    if not os.path.isfile(eval_file):
        raise FileNotFoundError(f"File not found: {eval_file}")
    with open(eval_file, "r") as f:
        return json.load(f)


def analyze_errors(data):
    stats = {
        "acknowledgement": 0,
        "hallucination": 0,
        "violation": 0,
        "error_unhelpful": 0,
        "error_inconsistent": 0,
        "hallucination_of_preference_violation": 0,
        "preference_unaware_violation": 0,
        "preference_adherence_accuracy": 0,
        "innovative": 0,
        "misleading": 0,
        "deep_mining": 0,
        "thoughtful": 0,
        "quality_aware_accuracy": 0,
    }

    for idx, entry in tqdm(enumerate(data)):
        if "evaluation_error_analysis" not in entry:
            print("Error: this entry has not been evaluated yet!")
            continue  # Skip entries that haven't been evaluated
        error_types = entry["evaluation_error_analysis"]
        
        # Basic checks
        is_acknowledgement = "yes" in error_types.get("acknow", {}).get("answer", "").lower()
        is_hallucination = is_acknowledgement and "yes" in error_types.get("hallucinate", {}).get("answer", "").lower()
        is_violation = "yes" in error_types.get("violate", {}).get("answer", "").lower()
        is_unhelpful = "no" in error_types.get("helpful", {}).get("answer", "").lower()
        
        # Quality checks
        is_innovative = "yes" in error_types.get("innovative", {}).get("answer", "").lower()
        is_misleading = "yes" in error_types.get("misleading", {}).get("answer", "").lower()
        is_deep_mining = "yes" in error_types.get("deep_mining", {}).get("answer", "").lower()
        is_thoughtful = "yes" in error_types.get("thoughtful", {}).get("answer", "").lower()

        # Error type calculations
        is_inconsistent = is_acknowledgement and not is_hallucination and is_violation and not is_unhelpful
        is_hallucination_of_preference_violation = (
            is_acknowledgement and is_hallucination and is_violation and not is_unhelpful
        )
        is_preference_unaware_violation = not is_acknowledgement and is_violation and not is_unhelpful

        # Basic preference following accuracy
        preference_following_accuracy = not any(
            [is_inconsistent, is_hallucination_of_preference_violation, is_preference_unaware_violation, is_unhelpful]
        )

        # Quality-aware accuracy
        quality_aware_accuracy = (is_innovative or is_deep_mining or is_thoughtful)

        # Update stats
        stats["acknowledgement"] += is_acknowledgement
        stats["hallucination"] += is_hallucination
        stats["violation"] += is_violation
        stats["error_unhelpful"] += is_unhelpful
        stats["error_inconsistent"] += is_inconsistent
        stats["hallucination_of_preference_violation"] += is_hallucination_of_preference_violation
        stats["preference_unaware_violation"] += is_preference_unaware_violation
        stats["preference_adherence_accuracy"] += preference_following_accuracy
        stats["innovative"] += is_innovative
        stats["misleading"] += is_misleading
        stats["deep_mining"] += is_deep_mining
        stats["thoughtful"] += is_thoughtful
        stats["quality_aware_accuracy"] += quality_aware_accuracy

    return stats, len(data)


def print_evaluation_results(stats, total_data, args):
    print("\n--- Evaluation Setup ---")
    print(f"Model: {args.model}")
    print(f"Topic: {args.topic}")
    print(f"Task: {args.task}")
    print(f"Preference Form: {args.pref_form}")

    print(f"\n--- Results ---")
    print(f"Total Entries Evaluated: {total_data}")
    print(f"Error Type Distribution:")
    print(f"  Unhelpful Responses: {stats['error_unhelpful']}")
    print(f"  Inconsistent Responses: {stats['error_inconsistent']}")
    print(f"  Hallucination of Preference Violations: {stats['hallucination_of_preference_violation']}")
    print(f"  Preference Unaware Violations: {stats['preference_unaware_violation']}")
    print(f"  Misleading Responses: {stats['misleading']}")
    
    # Calculate and display accuracies
    basic_accuracy = (stats["preference_adherence_accuracy"] / total_data) * 100
    quality_accuracy = (stats["quality_aware_accuracy"] / total_data) * 100
    print(f"\nAccuracy Metrics:")
    print(f"  Preference Following Accuracy: {basic_accuracy:.2f}%")
    print(f"  Quality-Aware Accuracy: {quality_accuracy:.2f}%")
    
    print(f"\n--- Additional Quality Metrics ---")
    if total_data > 0:
        innovative_pct = (stats['innovative'] / total_data) * 100
        deep_mining_pct = (stats['deep_mining'] / total_data) * 100
        thoughtful_pct = (stats['thoughtful'] / total_data) * 100
        print(f"  Innovative Responses: {stats['innovative']} ({innovative_pct:.2f}%)")
        print(f"  Deep Mining Responses: {stats['deep_mining']} ({deep_mining_pct:.2f}%)")
        print(f"  Thoughtful Responses: {stats['thoughtful']} ({thoughtful_pct:.2f}%)")
    else:
        print("  No data to calculate additional quality metrics.")


def main():
    exp_configs = load_config("../config.yaml")
    args = parse_arguments()
    generation_file, topic_data_path, eval_file = setup_paths(args, exp_configs)
    data = load_evaluation_data(eval_file)
    stats, total_data = analyze_errors(data)
    print_evaluation_results(stats, total_data, args)


if __name__ == "__main__":
    main()