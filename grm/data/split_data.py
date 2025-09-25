import json
import random
import os
import math

# --- 全局配置 ---
# 这是保证所有文件打乱顺序一致的“秘钥”。
# 在您所有的相关脚本中都必须使用这同一个种子！
SHARED_RANDOM_SEED = 42

# 定义文件路径
# 您可以根据实际情况修改这些列表
# 注意：这里的路径是示例，请替换成您的真实路径
input_dir = '/data/hzy_data/data/cot_preference'
output_base_dir = '/data/hzy_data/data/cot_preference_split'
train_dir = os.path.join(output_base_dir, 'train')
test_dir = os.path.join(output_base_dir, 'test')


# --- 核心功能函数 ---

def shuffle_list_with_seed(data_list, seed):
    """使用指定的种子对列表进行可复现的打乱。"""
    if not isinstance(data_list, list):
        print(f"警告：输入数据不是列表，无法打乱。")
        return data_list
    
    # 创建一个副本以避免修改原始数据
    shuffled_list = data_list[:]
    # 使用共享种子初始化随机数生成器
    random.seed(seed)
    # 对副本进行打乱
    random.shuffle(shuffled_list)
    return shuffled_list

def split_and_save_files():
    """
    读取、打乱、分割并保存所有文件。
    """
    # 确保输出目录存在
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print(f"创建输出目录: {train_dir}")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"创建输出目录: {test_dir}")

    file_names = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for file_name in file_names:
        input_path = os.path.join(input_dir, file_name)
        train_output_path = os.path.join(train_dir, file_name)
        test_output_path = os.path.join(test_dir, file_name)

        try:
            # 1. 读取文件
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. 使用共享种子打乱文件的数据
            shuffled_data = shuffle_list_with_seed(data, SHARED_RANDOM_SEED)

            # 3. 分割数据
            split_index = math.floor(len(shuffled_data) * 0.9)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]

            # 4. 将分割后的数据保存到新文件
            with open(train_output_path, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=4)
            
            with open(test_output_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=4)

            print(f"成功处理并保存文件: {train_output_path} 和 {test_output_path}")

        except FileNotFoundError:
            print(f"错误：找不到输入文件 {input_path}")
        except Exception as e:
            print(f"处理文件 {input_path} 时发生错误: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    split_and_save_files()