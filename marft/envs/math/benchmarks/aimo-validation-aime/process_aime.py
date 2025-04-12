import json
from typing import List

def process_data(input_file: str, output_file: str):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        # 转换字段
        new_entry = {
            "problem": item["problem"],
            "final_answer": item["answer"],
            "id": item["id"],
            "solution": item["solution"],
            "url": item["url"],
        }
        
        processed.append(new_entry)
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_data("train_s.json", "train.json")