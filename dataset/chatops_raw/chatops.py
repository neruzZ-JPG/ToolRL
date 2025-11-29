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

np.random.seed(31415)

def files_to_json(files_dir, output_json):
    data = []
    # 获取目录下所有文件
    for file_name in os.listdir(files_dir):
        file_path = os.path.join(files_dir, file_name)
        # 只处理文件，不处理目录
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data.append(json.load(f))
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def process_chatops_dataset(dataset, type):
    data_source = 'chatops'

    raw_dir = f"./dataset/chatops_raw/{dataset}"
    out_dir = f'./dataset/chatops/{dataset}'

    files_to_json(raw_dir, raw_dir+".json")
    dataset = json.load(open(raw_dir+".json", "r"))
    # Load dataset

    # Shuffle dataset
    np.random.shuffle(dataset)

    # Split into train and test sets (2% test data)
    test_num = int(len(dataset) * 0.02)
    train_dataset = dataset[:-test_num]
    test_dataset = dataset[-test_num:]
    # Function to process each example
    def process_fn(example, idx, split, type):
        system_prompt = example[0]["System Message"]
        output = example[-1]["Ai Message"].strip()
        
        prompt = [
            {"role": "system", "content": system_prompt},
        ]
        for i in range(1, len(example)-1):
            message = example[i]
            if "Ai Message" in message.keys():
                prompt.append({"role": "assistant", "content": message["Ai Message"]})
            else:
                try:
                    prompt.append({"role": "user", "content": message["Human Message"]})
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

    # Process dataset using list comprehension
    train_dataset = [process_fn(d, idx, 'train', type) for idx, d in enumerate(train_dataset)]
    test_dataset = [process_fn(d, idx, 'test', type) for idx, d in enumerate(test_dataset)]

    # Convert to Pandas DataFrame
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    # Save as Parquet
    local_dir = out_dir
    os.makedirs(local_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_df.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved datasets to {local_dir}")

if __name__ == '__main__':
    process_chatops_dataset('observation', 'observation')
    process_chatops_dataset('tool_calling', 'tool_calling')
    union_df = pd.concat([
        pd.read_parquet(os.path.join('./dataset/chatops/observation', 'train.parquet')),
        pd.read_parquet(os.path.join('./dataset/chatops/tool_calling', 'train.parquet')),
    ])
    union_df.to_parquet(os.path.join('./dataset/chatops/union', 'train.parquet'))
    union_df = pd.concat([
        pd.read_parquet(os.path.join('./dataset/chatops/observation', 'test.parquet')),
        pd.read_parquet(os.path.join('./dataset/chatops/tool_calling', 'test.parquet')),
    ])
    union_df.to_parquet(os.path.join('./dataset/chatops/union', 'test.parquet'))

    
    

    