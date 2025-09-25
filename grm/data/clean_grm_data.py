import json
import os

input_file = '/data/hzy_data/data/grm/grm_train_data.json'
output_file = '/data/hzy_data/data/grm/grm_train_data_processed.json'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(line)
            continue

        # 1. 如果process中有包含"\n\n\n\n\nStep 6"，则删除"\n\n\n\n\nStep 6"在内的后面所有字符。
        if '\n\n\n\n\nStep 6' in data.get('process', ''):
            data['process'] = data['process'].split('\n\n\n\n\nStep 6')[0]

        # 2. 如果label中长度大于5，则只保留前5个。
        if 'label' in data and isinstance(data['label'], list) and len(data['label']) > 5:
            data['label'] = data['label'][:5]

        # 3. 如果label的最后一个critique内容是"Final Answer Node."，则添加"\n\n**SCORE:** "并跟上final_score的分数。
        if 'label' in data and isinstance(data['label'], list) and data['label']:
            last_label = data['label'][-1]
            if last_label.get('critique') == 'Final Answer Node.':
                last_label['critique'] += f'\n\n**SCORE:** {data.get("final_score", "")}'
        
        f_out.write(json.dumps(data) + '\n')