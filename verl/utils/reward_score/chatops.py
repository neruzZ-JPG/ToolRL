import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import re
def compute_tool_call_reward(gt, pd, max_possible_reward, min_possible_reward):
    """
    Compute the reward for tool call.
    Args:
        gt (str): Ground truth tool calling.
        pd (str): Predicted tool calling.
        max_possible_reward (float): Maximum possible reward.
        min_possible_reward (float): Minimum possible reward.
    Returns:
        float: Reward for tool call.
    """
    # step1 : format check
    format_check_pass = False
    format_score = (max_possible_reward - min_possible_reward) / 2
    try:
        pd_json = json.loads(pd)
    except:
        print(f"pd is not in json format: {pd}")
        return min_possible_reward
    if not isinstance(pd_json, dict):
        print(f"pd not a dict: {pd}")
        return min_possible_reward + format_score * 0.2
    if "tool_name" not in pd_json.keys():
        print(f"tool_name not in pd: {pd}")
        return min_possible_reward + format_score * 0.4
    if "parameters" not in pd_json.keys() or not isinstance(pd_json["parameters"], list):
        print(f"wrong parameter list : {pd}")
        return min_possible_reward + format_score * 0.6
    for para in pd_json["parameters"]:
        if not isinstance(para, dict):
            print(f"param not a dict: {para}")
            return min_possible_reward + format_score * 0.7
        if len(para.keys()) != 2:
            print(f"param len != 2: {para}")
            return min_possible_reward + format_score * 0.8
        if "parameter_name" not in para.keys() or "parameter_value" not in para.keys():
            print(f"wrong param format : {para}")
            return min_possible_reward + format_score * 0.9

    min_possible_reward = (min_possible_reward + max_possible_reward) / 2
    print(f"pd is :{pd} ,,, gt is {gt}")
    # step2 : check tool calling info
    gt_json = json.loads(gt)
    if pd_json["tool_name"] != gt_json["tool_name"]:
        return min_possible_reward
    total_param_num = len(gt_json["parameters"])
    params = {}
    for p in gt_json["parameters"]:
        params[p["parameter_name"]] = p["parameter_value"]
    param_cnt = 0
    for param in pd_json["parameters"]:
        if param["parameter_name"] in params.keys():
            if param["parameter_value"] == params[param["parameter_name"]]:
                param_cnt += 1
            else:
                param_cnt += 0.8
        else:
            param_cnt -= 0.2
    MINI = 0.00001
    return min_possible_reward + (max(MINI, param_cnt) / (total_param_num + MINI)) * (max_possible_reward - min_possible_reward)

SUCCESS_FLAG = "SUCCESS"
FAILURE_FLAG = "FAIL"

PLANNING_JUDGE_PROMPT = '''
You're a professional and experienced planner. Your task is to evaluate the quality of a given planning.
#####
In the context of the following prompt:
{input_str}
#####
The predicted planning is as follows:
{pd}
#####
Your evaluation criteria are as follows:
1. The planning should be aimed at accomplishing the user request
2. The planning should not go beyond the user request
3. The planning should not contain any unnecessary steps. If user request is finished, the planning should begin with "SUCCESS"; If the user request cannot be finished, the planning should begin with "FAIL".Else, it should contain several steps to be done.
4. The planning should be as short and concise as possible. And it should not contain any previously-executed steps.
##### 
Your evaluation score should be a float number between 0 and 1. 
Output the number and ONLY the number!
'''

def compute_planning_reward(input_str, gt, pd, max_possible_reward, min_possible_reward):
    """
    Compute the reward for planning.
    Args:
        gt (str): Ground truth planning.
        pd (str): Predicted planning.
        max_possible_reward (float): Maximum possible reward.
        min_possible_reward (float): Minimum possible reward.
    Returns:
        float: Reward for planning.
    """
    print(f"pd for planning: {pd}")
    # step1 : format check
    format_check_pass = False
    if pd.startswith(SUCCESS_FLAG) or pd.startswith(FAILURE_FLAG):
        format_check_pass = True
    if not format_check_pass:
        try:
            gt_json = json.loads(gt)
            if isinstance(gt_json, list):
                format_check_pass = True
            else:
                return min_possible_reward
        except:
            return min_possible_reward
    if not format_check_pass:
        return min_possible_reward
    min_possible_reward = (min_possible_reward + max_possible_reward) / 2
    # step2 :  llm judge?
    base_url = "https://vip.apiyi.com/v1"
    api_key = "sk-5PYQRpTeWXyM9ibd96B5737aFdCc47B1B89a3937F6447eEe"
    model_name = "gpt-4.1-nano"
    temperature = 1.0
    prompt_template = ChatPromptTemplate.from_template(PLANNING_JUDGE_PROMPT)
    llm = ChatOpenAI(
        base_url=base_url, 
        api_key=api_key, 
        model=model_name,
        temperature = temperature,
        timeout=15.0,
        max_retries=3,
    )
    messages = prompt_template.format_messages(input_str=input_str, pd=pd)
    try:
        response = llm.invoke(messages)
        score = float(response.content)
    except:
        print("llm juedge issue, return mid score")
        return (min_possible_reward + max_possible_reward) / 2
    print(f"llm judge score : {score}")
    if score < 0:
        return min_possible_reward
    if score > 1:
        return max_possible_reward
    return score * (max_possible_reward - min_possible_reward) + min_possible_reward

def remove_thinking_tags(text):
    """
    移除字符串中的<think>...</think>思考标签及其中间内容
    """
    # 使用正则表达式移除<think>...</think>标签及内容
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return cleaned_text.strip()

def compute_score(data_source,
                solution_str,
                ground_truth,
                extra_info):
    """
    Compute the reward for the solution.
    Args:
        solution_str (str): Solution string.
        ground_truth (str): Ground truth.
        input_str (str): the prompt for llm, used for score computation
    Returns:
        float: Reward for the solution.
    """
    exp_name = str(os.getenv("EXPERIMENT_NAME", ""))
    if "llama" in exp_name:
        predict_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")
    predict_str = remove_thinking_tags(predict_str)
    # print(predict_str)
    score = 0
    type = extra_info.get("type", None)
    if type == 'observation':
        input_str = extra_info.get("input_str", None)
        if input_str is None:
            raise ValueError("input_str is None")
        score = compute_planning_reward(input_str, ground_truth, predict_str, 5, -5)
    elif type == 'tool_calling':
        score = compute_tool_call_reward(ground_truth, predict_str, 5, -5)
    else:

        raise NotImplementedError
    print(f"final score: {score}")
    return score