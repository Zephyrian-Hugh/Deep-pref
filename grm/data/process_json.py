import json
import os

def process_json_files():
    origin_data_dir = '/data/hzy_data/data/origin_data'
    output_dir = '/data/hzy_data/data/processed_data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(origin_data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(origin_data_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for item in data:
                    if 'response_list' in item and isinstance(item['response_list'], list):
                        for res in item['response_list']:
                            if 'response' in res and isinstance(res['response'], list) and len(res['response']) > 5:
                                res['response'] = res['response'][:5]
                                res['answer'] = res['response'][-1][8:]
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    process_json_files()