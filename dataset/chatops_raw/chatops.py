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


def process_chatops_dataset(target_dataset_name, process_type):
    """
    target_dataset_name: 输出文件夹的名字 (例如 'plan', 'observation', 'tool_calling')
    process_type: 处理逻辑类型 ('plan', 'observation', 'tool_calling')
    """
    data_source = 'chatops'

    # 【关键修改 1】源文件重定向
    # 如果我们要生成 'plan' 数据，其实原始数据在于 'observation' 文件夹里
    raw_source_name = "observation" if target_dataset_name == "plan" else target_dataset_name
    
    raw_dir = f"./dataset/chatops_raw/{raw_source_name}"
    out_dir = f'./dataset/chatops/{target_dataset_name}' # 这里是真正保存的目录

    # 确保源文件存在
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory {raw_dir} does not exist.")
        return

    # 生成临时 json (如果还没生成过)
    json_path = raw_dir + ".json"
    files_to_json(raw_dir, json_path)
    
    print(f"Loading raw data from {json_path}...")
    dataset = json.load(open(json_path, "r"))
    
    # Shuffle & Split
    np.random.shuffle(dataset)
    test_num = int(len(dataset) * 0.1)
    train_dataset = dataset[:-test_num]
    test_dataset = dataset[-test_num:]

    # 内部处理函数
    def process_fn(example, idx, split, current_type):
        try:
            # 获取 Prompt 和 Output
            system_prompt = example[0]["System Message"].replace("toolname", "tool_name")
            output = example[-1]["Ai Message"].strip()
            output = output.replace("toolname", "tool_name")
        except (KeyError, IndexError):
            return None

        final_output_type = current_data_type = current_type
        
        # --- 情况 A: 处理 Plan 类型 ---
        if current_type == "plan":
            # 必须是 JSON List 格式，否则丢弃
            if not (output.startswith("[") and output.endswith("]")):
                return None
            try:
                plan_json = json.loads(output)
                if not isinstance(plan_json, list):
                    return None
            except:
                return None
            # 通过筛选，保留数据
            final_output_type = "plan"


        # --- 情况 C: 处理 Tool Calling 类型 ---
        elif current_type == "tool_calling":
            pd_json = None
            try:
                pd_json = json.loads(output)
            except:
                return None
            
            if not isinstance(pd_json, dict):
                return None
            if "tool_name" not in pd_json:
                return None
            if "parameters" not in pd_json or not isinstance(pd_json["parameters"], list):
                return None
            
            final_output_type = "tool_calling"

        # 构建 Prompt list
        prompt = [{"role": "system", "content": system_prompt}]
        for i in range(1, len(example)-1):
            message = example[i]
            if "Ai Message" in message:
                prompt.append({"role": "assistant", "content": message["Ai Message"].replace("toolname", "tool_name")})
            elif "Human Message" in message:
                prompt.append({"role": "user", "content": message["Human Message"].replace("toolname", "tool_name")})

        # 构建最终数据结构
        data = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "chatops", 
            "reward_model": {
                "style": "rule",
                "ground_truth": output
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'input_str': json.dumps(prompt, ensure_ascii=False),
                "output": output,
                "type": final_output_type # 记录具体的类型 (plan, SUCCESS, tool_calling)
            }
        }
        return data

    # 执行处理
    print(f"Processing {target_dataset_name} ({process_type}) train set...")
    train_output = [res for idx, d in enumerate(train_dataset) if (res := process_fn(d, idx, 'train', process_type)) is not None]
    
    print(f"Processing {target_dataset_name} ({process_type}) test set...")
    test_output = [res for idx, d in enumerate(test_dataset) if (res := process_fn(d, idx, 'test', process_type)) is not None]

    # 保存 Parquet
    local_dir = out_dir
    os.makedirs(local_dir, exist_ok=True)

    pd.DataFrame(train_output).to_parquet(os.path.join(local_dir, 'train.parquet'))
    pd.DataFrame(test_output).to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"✅ Saved to {local_dir} | Train: {len(train_output)}, Test: {len(test_output)}")

if __name__ == '__main__':
    # 1. 生成纯净的 Observation 数据 (不含 Plan)
    # 读取 raw/observation -> 过滤 -> 保存到 dataset/chatops/observation
    process_chatops_dataset('observation', 'observation')

    # 2. 生成纯净的 Plan 数据
    # 读取 raw/observation -> 过滤 -> 保存到 dataset/chatops/plan
    process_chatops_dataset('plan', 'plan') 

    # 3. 生成 Tool Calling 数据
    # 读取 raw/tool_calling -> 过滤 -> 保存到 dataset/chatops/tool_calling
    process_chatops_dataset('tool_calling', 'tool_calling')
    
    # 将两个文件夹的数据合并
    union_dir = './dataset/chatops/union'
    os.makedirs(union_dir, exist_ok=True)
    
    # 合并 Train
    dfs_train = []
    dfs_test = [] # <--- 新增 Test 列表
    for dtype in ['observation', 'tool_calling']:
        # 读取 Train
        p_train = os.path.join(f'./dataset/chatops/{dtype}', 'train.parquet')
        if os.path.exists(p_train):
            dfs_train.append(pd.read_parquet(p_train))
        
        # 读取 Test
        p_test = os.path.join(f'./dataset/chatops/{dtype}', 'test.parquet') # <--- 新增
        if os.path.exists(p_test):
            dfs_test.append(pd.read_parquet(p_test))
    
    if dfs_train:
        union_df = pd.concat(dfs_train)
        union_df.to_parquet(os.path.join(union_dir, 'train.parquet'))
        print(f"✅ Union Train size: {len(union_df)}")

    # 【新增】合并并保存 Test Parquet
    if dfs_test:
        union_test_df = pd.concat(dfs_test)
        union_test_df.to_parquet(os.path.join(union_dir, 'test.parquet'))
        print(f"✅ Union Test size: {len(union_test_df)}")
        
        
    # 5. 生成 Union Double Plan 数据集 (Obs + Tool + Plan)
    union_double_plan_dir = './dataset/chatops/union_double_plan'
    os.makedirs(union_double_plan_dir, exist_ok=True)
    
    dfs_train = []
    dfs_test = [] # <--- 新增 Test 列表
    for dtype in ['observation', 'plan', 'tool_calling']:
        # 读取 Train
        p_train = os.path.join(f'./dataset/chatops/{dtype}', 'train.parquet')
        if os.path.exists(p_train):
            dfs_train.append(pd.read_parquet(p_train))
            if dtype == 'plan': # Double Plan Logic
                 dfs_train.append(pd.read_parquet(p_train)) # Copy 1
                 # dfs_train.append(pd.read_parquet(p_train)) # Copy 2 (Optional)

        # 读取 Test
        p_test = os.path.join(f'./dataset/chatops/{dtype}', 'test.parquet') # <--- 新增
        if os.path.exists(p_test):
            dfs_test.append(pd.read_parquet(p_test))
            # Test 集通常不需要 Double，保持原始分布即可，当然 Double 也没坏处
    
    if dfs_train:
        union_df = pd.concat(dfs_train)
        union_df.to_parquet(os.path.join(union_double_plan_dir, 'train.parquet'))
        print(f"✅ Union Double Plan Train size: {len(union_df)}")

    # 【新增】合并并保存 Test Parquet
    if dfs_test:
        union_test_df = pd.concat(dfs_test)
        union_test_df.to_parquet(os.path.join(union_double_plan_dir, 'test.parquet'))
        print(f"✅ Union Double Plan Test size: {len(union_test_df)}")
    
    

    