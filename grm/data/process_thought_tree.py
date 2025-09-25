import json
import argparse
import os

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in:
        try:
            data = json.load(f_in)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {input_path}")
            return

    with open(output_path, 'a', encoding='utf-8') as f_out:
        record_idx = 0
        for record in data:
            record_idx += 1
            question = record.get("initial_question")
            preference = record.get("user_profile")
            thought_tree = record.get("thought_tree")

            if not all([question, preference, thought_tree]):
                continue

            def find_paths_recursive(node_id, current_path):
                node = thought_tree.get(str(node_id))
                if not node:
                    return

                current_path.append(node)

                children = node.get("children", [])
                if not children:
                    process_str = ""
                    labels = []
                    final_score = 0.0

                    for i, step_node in enumerate(current_path):
                        process_str += f"Step {i + 1}: {step_node.get('thought', '')}\n\n\n\n\n"
                        labels.append({
                            "critique": step_node.get("reasoning", ""),
                            "score": step_node.get("score", 0.0)
                        })

                    if current_path:
                        final_score = current_path[-1].get("score", 0.0)

                    output_record = {
                        "idx": record_idx,
                        "question": question,
                        "preference": preference,
                        "process": process_str.strip(),
                        "label": labels,
                        "final_score": final_score
                    }
                    f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    return

                for child_id in children:
                    find_paths_recursive(child_id, list(current_path))

            root_node = thought_tree.get("0")
            if root_node and isinstance(root_node, dict):
                for child_id in root_node.get("children", []):
                    find_paths_recursive(child_id, [])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process thought tree data.')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSONL file.')
    args = parser.parse_args()

    # Clear output file before writing
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(args.input_dir, filename)
            process_file(input_path, args.output_file)
            print(f"Processed {filename}")

    print(f"Processing complete. Output written to {args.output_file}")