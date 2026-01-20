import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import re
def compute_tool_call_reward(gt, pd, max_possible_reward, min_possible_reward, format_ratio):
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
    format_score = (max_possible_reward - min_possible_reward) * format_ratio
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

    min_possible_reward = min_possible_reward + format_score
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
You are a strict QA system for a planning agent. Your goal is to evaluate if the predicted planning (`pd`) strictly follows the complex branching logic and format constraints based on the user request (`input_str`) and execution history.

### 1. The Three Mutually Exclusive Modes
The `pd` MUST belong to EXACTLY ONE of the following modes.

* **Mode A: Planning (JSON Array of Independent Paths)**
    * **Condition:** The user request is NOT fully satisfied yet.
    * **Format:** A raw JSON array containing **at least 2 strings**. NO markdown formatting (no ```json).
    * **Constraint 1 (Independence):** Each item in the list must be a **distinct, independent solution path** or a **parallel subtask**. Plan B CANNOT depend on the result of Plan A.
    * **Constraint 2 (Context Awareness):** Plans must carry over known information (e.g., "Known name is Harry, find age") rather than starting from zero.
    * **Example:** `["Search for the user's age based on name 'Harry'", "Search for the user's signup date based on name 'Harry'"]`

* **Mode B: Success (Termination)**
    * **Condition:** The **ENTIRE** User Request is fulfilled.
    * **Format:** String starting with "SUCCESS, " followed by the final answer.
    * **Constraint:** Do NOT return SUCCESS if only a sub-task is done (e.g., found Name but not Age). This is a critical error.

* **Mode C: Failure (Termination)**
    * **Condition:** The request is proven **UNSOLVABLE** (e.g., API dead, no workaround).
    * **Format:** String starting with "FAIL, " followed by the reason.
    * **Constraint:** Do NOT return FAIL if alternative paths exist.

---

### 2. Scoring Rubric (Strict & Granular)

#### **Level 1: Critical Format Violation (Score: 0.0)**
* **The "Markdown" Error:** Output contains markdown code blocks (e.g., ```json ... ```). The requirement is RAW text/JSON.
* **The "Double-Talk" Error:** Output mixes text and JSON.
* **Invalid JSON:** Syntax error preventing parsing.

#### **Level 2: Major Logic Errors (Score: 0.1 - 0.3)**
* **Violation of Independence (Sequential Dependency):** [CRITICAL for Mode A]
    * *Error:* The plans look like steps in a sequence.
    * *Example:* `["Download file", "Read the downloaded file"]` -> **Score 0.1** (Plan 2 relies on Plan 1. They must be independent).
* **Quantity Violation:** [CRITICAL for Mode A]
    * *Error:* The JSON array contains fewer than 2 plans.
    * *Example:* `["Just one plan"]` -> **Score 0.2** (Instruction requires at least 2).
* **Premature Success (Partial Completion):**
    * *Error:* User asked for "Name and Age", Agent found Name and returned SUCCESS. -> **Score 0.1**
* **Premature Give-up:**
    * *Error:* Returning FAIL when retries or other tools are available. -> **Score 0.2**

#### **Level 3: Strategic Flaws (Score: 0.4 - 0.6)**
* **Context Amnesia:**
    * *Error:* The plans ignore information already retrieved.
    * *Context:* Agent already found "ID: 123".
    * *Bad Plan:* `["Find user ID"]` (Redundant/Wasteful).
    * *Good Plan:* `["Use ID 123 to find email"]`.
* **Logical Weakness:**
    * *Error:* The proposed plans are technically valid but unlikely to work (e.g., brute forcing a password).
* **Vague Plans:**
    * *Error:* `["Do something", "Try another way"]` (Too generic).

#### **Level 4: Efficiency & Quality Issues (Score: 0.7 - 0.9)**
* **Minor Overlap:** The two plans are slightly too similar (lack of diversity in strategy).
* **Wording Issues:** The formatting is correct, but the English description is slightly awkward.

#### **Level 5: Perfect Execution (Score: 1.0)**
* **Mode A:** Valid JSON array, **count >= 2**, plans are strictly independent, fully utilizes context.
* **Mode B:** Correctly identifies that ALL parts of the user request are done.
* **Mode C:** Correctly identifies a true dead end.

---

### 3. Evaluation Task
User Request:
{input_str}

Predicted Planning (`pd`):
{pd}

### 4. Output
1. Check Format first (Raw JSON? No Markdown?).
2. Check Logic (Independence? Quantity >= 2? Full completion?).
3. Output the float number score (0.0 to 1.0).

**Output ONLY the float number.**
'''

def compute_planning_reward(input_str, gt, pd, max_possible_reward, min_possible_reward, format_ratio):
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
    format_score = (max_possible_reward - min_possible_reward) * format_ratio
    format_check_pass = False
    if pd.startswith(SUCCESS_FLAG) or pd.startswith(FAILURE_FLAG):
        format_check_pass = True
    if not format_check_pass:
        try:
            gt_json = json.loads(pd)
            if isinstance(gt_json, list):
                format_check_pass = True
            else:
                return min_possible_reward
        except:
            return min_possible_reward
    if not format_check_pass:
        return min_possible_reward
    min_possible_reward = min_possible_reward + format_score
    # step2 : gt compare
    # only compare their first words
    gt_same = False
    bias = 0
    if gt.startswith(SUCCESS_FLAG) and pd.startswith(SUCCESS_FLAG):
        gt_same = True
    if gt.startswith(FAILURE_FLAG) and pd.startswith(FAILURE_FLAG):
        gt_same = True
    if gt.strip().startswith("[") and pd.strip().startswith("["):
        gt_same = True
    if not gt_same:
        bias -= 1
    # step3 :  llm judge?
    # base_url = "https://vip.apiyi.com/v1"
    # api_key = "sk-5PYQRpTeWXyM9ibd96B5737aFdCc47B1B89a3937F6447eEe"
    base_url = "https://hk.n1n.ai/v1"
    api_key = "sk-EdZZPRcLsVDAAoyZLHloQC27ejjZKKqnemVvR6tnQUZ9pw5C"
    model_name = "gpt-5-nano"
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
        return (min_possible_reward + max_possible_reward) / 2 + bias
    print(f"llm judge score : {score}")
    if score < 0:
        return min_possible_reward
    if score > 1:
        return max_possible_reward
    return (score * (max_possible_reward - min_possible_reward) + min_possible_reward) + bias


def compute_planning_reward_for_plan(input_str, gt, pd, max_possible_reward, min_possible_reward, format_ratio):
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
    print(f"pd for planning only: {pd}")
    format_score = (max_possible_reward - min_possible_reward) * format_ratio
    # step1 : format check
    format_check_pass = False
    try:
        pd_json = json.loads(pd)
        if not isinstance(pd_json, list):
            print("Plan is not a list")
            return min_possible_reward # 格式错直接最低分
        format_check_pass = True
    except:
        print("Plan is not valid json")
        return min_possible_reward # 格式错直接最低分
    if not format_check_pass:
        return min_possible_reward
    min_possible_reward = min_possible_reward + format_score
    # step3 :  llm judge?
    # base_url = "https://vip.apiyi.com/v1"
    # api_key = "sk-5PYQRpTeWXyM9ibd96B5737aFdCc47B1B89a3937F6447eEe"
    base_url = "https://hk.n1n.ai/v1"
    api_key = "sk-EdZZPRcLsVDAAoyZLHloQC27ejjZKKqnemVvR6tnQUZ9pw5C"
    model_name = "gpt-5-nano"
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
    return (score * (max_possible_reward - min_possible_reward) + min_possible_reward)

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
        score = compute_planning_reward(input_str, ground_truth, predict_str, 5, -5, 0.3)
    elif type == 'tool_calling':
        score = compute_tool_call_reward(ground_truth, predict_str, 5, -5, 0.3)
    elif type == 'plan':
        input_str = extra_info.get("input_str", None)
        if input_str is None:
            raise ValueError("input_str is None")
        score = compute_planning_reward_for_plan(input_str, ground_truth, predict_str, 5, -5, 0.5)
    else:
        raise NotImplementedError
    print(f"pd is : {predict_str}, gt is :{ground_truth}, final score: {score}")
    return score