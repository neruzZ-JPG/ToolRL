# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import json
import numpy as np
import pandas as pd
import argparse
import json
import ast

def make_serializable(obj):
    """
    暴力清洗器：递归遍历对象，把所有 JSON 不支持的类型（bytes）强转为字符串
    """
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    else:
        return obj

def recursive_parse(content):
    """
    递归解析：尝试把字符串剥离成由 dict/list 组成的纯对象
    """
    # 1. 已经是对象，递归清洗内部
    if isinstance(content, dict):
        return {k: recursive_parse(v) for k, v in content.items()}
    if isinstance(content, list):
        return [recursive_parse(i) for i in content]
    
    # 2. 如果不是字符串，直接返回（经过 make_serializable 处理后这里很安全）
    if not isinstance(content, str):
        return content
    
    content = content.strip()
    
    # 3. 处理 agent_response 前缀
    if content.startswith('agent_response :'):
        try:
            prefix, payload = content.split(':', 1)
            parsed_payload = recursive_parse(payload)
            # 如果 payload 是对象，为了美观，序列化回干净的 JSON 串
            if isinstance(parsed_payload, (dict, list)):
                # 重新 dumps 时确保没有 bytes，防止报错
                safe_payload = make_serializable(parsed_payload)
                clean_json = json.dumps(safe_payload, ensure_ascii=False)
                return f"{prefix.strip()} : {clean_json}"
            return f"{prefix.strip()} : {parsed_payload}"
        except:
            pass

    # 4. 尝试解析 (JSON 或 Python Literal)
    parsed = None
    success = False

    try:
        parsed = json.loads(content)
        success = True
    except:
        try:
            # 只有像结构体的才尝试 ast，防止误伤普通文本
            if content.startswith(("{", "[", "b'", 'b"')):
                parsed = ast.literal_eval(content)
                success = True
        except:
            pass
    
    if success:
        # 【关键步骤】解析出来后，立刻把里面的 bytes 递归转成字符串
        # 这一步解决了 "bytes is not JSON serializable" 错误
        parsed = make_serializable(parsed)
        # 继续递归，处理多重嵌套
        return recursive_parse(parsed)

    # 5. 实在解析不了的字符串，按你的要求，做一个简单的替换优化
    # 去除多余的转义符，把 \" 变成 " (仅在视觉上，不改变逻辑)
    # 这一步比较激进，只对非 JSON 格式的纯文本做
    content = content.replace('\\n', '\n')
    
    # 如果是 b'...' 格式但没被解析成 bytes，手动去头去尾
    if content.startswith("b'") and content.endswith("'"):
        content = content[2:-1]
    
    return content

def clean_conversation_data(conversation_list):
    cleaned_list = []
    
    for item in conversation_list:
        new_item = {}
        for role, content in item.items():
            # 1. 递归解析得到纯对象
            obj_data = recursive_parse(content)
            
            # 2. 再次确保对象里没有 bytes (双重保险)
            obj_data = make_serializable(obj_data)
            
            # 3. 统一序列化为 JSON 字符串
            if isinstance(obj_data, (dict, list)):
                cleaned_content = json.dumps(obj_data, ensure_ascii=False, separators=(',', ':'))
            else:
                cleaned_content = str(obj_data)
                
            new_item[role] = cleaned_content
        cleaned_list.append(new_item)
    
    return cleaned_list

np.random.seed(31415)

def files_to_json(files_dir, output_json):
    data = []
    # 获取目录下所有文件
    for file_name in os.listdir(files_dir):
        file_path = os.path.join(files_dir, file_name)
        # 只处理文件，不处理目录
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                processed = clean_conversation_data(json.load(f))
                data.append(processed)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def process_chatops_dataset(dataset, type):
    data_source = 'chatops'

    raw_dir = f"./dataset/chatops_raw/{dataset}"
    out_dir = f'./dataset/chatops/{dataset}'

    files_to_json(raw_dir, raw_dir+".json")
    dataset = json.load(open(raw_dir+".json", "r"))
    
    # Shuffle dataset
    np.random.shuffle(dataset)

    # Split into train and test sets (2% test data)
    test_num = int(len(dataset) * 0.1)
    train_dataset = dataset[:-test_num]
    test_dataset = dataset[-test_num:]

    # Function to process each example
    def process_fn(example, idx, split, type):
        try:
            # 增加保护，防止空数据索引报错
            system_prompt = example[0]["System Message"].replace("toolname", "tool_name")
            output = example[-1]["Ai Message"].strip()
            output = output.replace("toolname", "tool_name")
        except (KeyError, IndexError):
            return None

        if type == "tool_calling":
            format_check_pass = True
            pd_json = None
            try:
                pd_json = json.loads(output)
            except:
                format_check_pass = False
                # print(f"json parse failed {idx}")
                return None
            
            # --- FIX START ---
            if not isinstance(pd_json, dict):
                format_check_pass = False
            else:
                # 只有确认为 dict 后才检查 key
                if "tool_name" not in pd_json:
                    format_check_pass = False
                elif "parameters" not in pd_json or not isinstance(pd_json["parameters"], list):
                    format_check_pass = False
            # --- FIX END ---

            if not format_check_pass:
                print(f"format check not pass {idx}")
                return None
                
        prompt = [
            {"role": "system", "content": system_prompt},
        ]
        for i in range(1, len(example)-1):
            message = example[i]
            # 这里也建议用 .get 防止报错
            if "Ai Message" in message:
                prompt.append({"role": "assistant", "content": message["Ai Message"].replace("toolname", "tool_name")})
            else:
                try:
                    prompt.append({"role": "user", "content": message["Human Message"].replace("toolname", "tool_name")})
                except:
                    print(message)

        data = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": output
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'input_str': json.dumps(prompt, ensure_ascii=False),
                "output": output,
                "type": type
            }
        }
        return data

    # Process dataset and filter None values
    print(f"Processing {type} train set...")
    train_dataset = [res for idx, d in enumerate(train_dataset) if (res := process_fn(d, idx, 'train', type)) is not None]
    
    print(f"Processing {type} test set...")
    test_dataset = [res for idx, d in enumerate(test_dataset) if (res := process_fn(d, idx, 'test', type)) is not None]

    # Convert to Pandas DataFrame
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    # Save as Parquet
    local_dir = out_dir
    os.makedirs(local_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved datasets to {local_dir}. Train size: {len(train_df)}, Test size: {len(test_df)}")

if __name__ == '__main__':
    process_chatops_dataset('observation', 'observation')
    process_chatops_dataset('tool_calling', 'tool_calling')
    union_dir = './dataset/chatops/union'
    os.makedirs(union_dir, exist_ok=True)
    union_df = pd.concat([
        pd.read_parquet(os.path.join('./dataset/chatops/observation', 'train.parquet')),
        pd.read_parquet(os.path.join('./dataset/chatops/tool_calling', 'train.parquet')),
    ])
    union_df.to_parquet(os.path.join(union_dir, 'train.parquet'))
    union_df = pd.concat([
        pd.read_parquet(os.path.join('./dataset/chatops/observation', 'test.parquet')),
        pd.read_parquet(os.path.join('./dataset/chatops/tool_calling', 'test.parquet')),
    ])
    union_df.to_parquet(os.path.join(union_dir, 'test.parquet'))
    
    

    