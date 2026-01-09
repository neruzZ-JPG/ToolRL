from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PLANNING_JUDGE_PROMPT = '''
You are a strict QA system for a planning agent. Your goal is to evaluate if the predicted planning (`pd`) strictly follows the format and logic constraints based on the user request (`input_str`).

### 1. The Three Mutually Exclusive Modes
The `pd` MUST belong to EXACTLY ONE of the following modes. Any mixture is a CRITICAL FORMAT ERROR.

* **Mode A: Working (JSON List)**
    * Used when the task is NOT finished.
    * Format: A pure JSON list of next steps.
    * Example: `["Step 1", "Step 2"]`
    * *Forbidden:* Cannot contain "SUCCESS" or "FAIL" keywords.

* **Mode B: Success (Text Only)**
    * Used when the task is fully completed.
    * Format: Starts with "SUCCESS". Followed by a short summary.
    * *Forbidden:* CANNOT contain a list of future steps (because it's done!). CANNOT contain JSON syntax.

* **Mode C: Failure (Text Only)**
    * Used when the task cannot proceed (e.g., missing info, error).
    * Format: Starts with "FAIL". Followed by the reason.
    * *Forbidden:* **CANNOT contain "Next steps" or a list of actions.** If the agent knows the next steps, it should have outputted Mode A (Working), OR it should just report the error and stop. A "FAIL" with a "Plan" is a contradiction.

### 2. Scoring Rubric (Strict)

* **Score 0.0 (Format Violation):**
    * **The "Double-Talk" Error:** The output starts with FAIL/SUCCESS but *also* provides a list of steps (JSON or Text).
        * *Bad Example:* `FAIL. Missing token. The next steps are: ["Login"]` -> **Score: 0.0**
    * The output contains both text and JSON (e.g., "Here is the plan: [...]").
    * The output is not valid JSON when in Mode A.

* **Score 0.1 - 0.9 (Content Quality):**
    * Format is correct, but the logic is flawed (e.g., unnecessary steps, wrong tool, infinite loop).

* **Score 1.0 (Perfect):**
    * Format is perfect AND logic efficiently addresses the user request.

### 3. Evaluation Task
User Request:
{input_str}

Predicted Planning (`pd`):
{pd}

### 4. Output
First, determine the Mode. Then check for "Double-Talk" errors.
Finally, output the float number score (0.0 to 1.0). 
Output ONLY the number.
'''

base_url = "https://vip.apiyi.com/v1"
api_key = "sk-5PYQRpTeWXyM9ibd96B5737aFdCc47B1B89a3937F6447eEe"
model_name = "gpt-5-nano"
temperature = 1.0
input_str = '''
        {
            "System Message": "You're a professional Devops engineer, leading a team with several devops agents each of whom is in charge of one devops tools such as github. \nYour main goal is to analyze the user's request, decompose it into subtasks and lead your team to finish the request step by step.\n#####\nThere are the agents:\n[\n    {\n        \"name\" : \"docker_agent\",\n        \"description\" : \"docker agent for docker tasks\",\n        \"url\" : \"http://localhost:8000\"\n    },\n    {\n        \"name\" : \"github_agent\",\n        \"description\" : \"github agent for github tasks\",\n        \"url\" : \"http://localhost:8001\"\n    },\n    {\n        \"name\" : \"gitlab_agent\",\n        \"description\" : \"gitlab agent for gitlab tasks\",\n        \"url\" : \"http://localhost:8002\"\n    },\n\n    {\n        \"name\" : \"jenkins_agent\",\n        \"description\" : \"jenkins agent for jenkins tasks\",\n        \"url\" : \"http://localhost:8003\"\n    },\n    {\n        \"name\" : \"kubernetes_agent\",\n        \"description\" : \"kubernetes agent for kubernetes/k8s tasks\",\n        \"url\" : \"http://localhost:8004\"\n    }\n]\n#####\nGo through these 3 steps to solve a subtask:\n1. QUERY_EXECUTABLE \nuery the agents if the task is executable.If so, what information do they need to finish it.\n- Not all agents should participate in this task, choose the most suitable one.\n- Respond with the name of the agent, and only the name of it.\n- For example, if your task is to get information about a dog, and there's a 'pet_agent' in the agent list. Although there might be a lot of information of the agent in the list, you only need to respond with its name: pet_agent.\n2. EXECUTE\nYour agent have told you all the information needed to finish your task, including the tool name and the required parameters. Now all you need to do is to provide the tool name as well as all the necessary parameters to your agent.\n- For example, if the task is \"How long does a Beagle live?\".Pet_agent tells you that the task can be finished with one of his tools called \"query_dog_info\", and you need to provide the species of the dog you want to query about. Then you may pass the task as \"{\"toolname\" : \"query_dog_info\", \"parameters\" : [{\"parameter_name\" : \"dog_species\", \"parameter_value\" : \"beagle\"}]}\".\n- One execution or query at one time.\n3. OBSERVATION\nBased on the current state, determine whether the USER_REQUEST(not the subtask) has been fulfilled. \n- If you believe the request has been successfully completed, return 'SUCCESS' followed by a response to the user. For example: 'SUCCESS. The user's age is 18.' By returning 'SUCCESS' you can end the whole workflow.\n- If the current workflow doesn't align with the user request, return 'FAIL' and explain the reason. For instance, if the user asks for the weather in New York but the current state is searching for a bakery's location, continuing the search is incorrect. You should stop the unproductive effort by replying 'FAIL. Wrong workflow.' Returning 'FAIL' will end the whole workflow too.\n- If the request is not yet finished and more actions are required, formulate multiple plans for the next step. Keep in mind that:\n    - Plans must be formulated based on existing information, making full use of the available information and avoiding repeating the wrong actions that have already been taken.\n    - Each plan represents a different solution path and will be executed independently, not sequentially. This means Plans A and B are designed to be mutually exclusive, with neither having access to the other's information.\n    - Present the plans as a JSON - formatted array. \n    - For example, if the user wants to find the name, age, and sign - up date of a user given their ID, and you've found the user's name (say, Harry), you should return ['The user's name is Harry, find his age.', 'The user's name is Harry, find out when he signed up.']\"\nRESPONSE STRUCTURE:\nOne of the following three responses:\n    - 'SUCCESS, [success information]'\n    - 'FAIL, [fail information]'\n    - A JSON - formatted array of subtasks, with only the structure content without any json identifier.\nYour observation will be strictly parsed, and you must strictly follow the response structure."
        },
        {
            "Human Message": "the user request is:\nRemove the existing pod named 'test-pod-123' in the 'dev' namespace, then verify the cleanup by listing all pods in that namespace to confirm its deletion."
        },
        {
            "Human Message": "Current subtask is:\nFirst, delete the pod from the dev namespace:\nkubectl delete pod test-pod-123 -n dev"
        },
        {
            "Human Message": "Step : QUERY_EXECUTABLE\nNow you've got a new subtask to finish, you need to select the suitable agent to finish it."
        },
        {
            "Ai Message": "kubernetes_agent"
        },
        {
            "Ai Message": "agent_response : {\"tool_name\": \"kubectl_delete\", \"parameters\": {\"resourceType\": \"pod\", \"name\": \"test-pod-123\", \"namespace\": \"dev\"}}"
        },
        {
            "Human Message": "Step : EXECUTE\nYour agent have told you all the information needed to finish your task. Now all you need to do is to provide all the necessary parameters to your agent.\noutput in a strict JSON dict format and ONLY the json dict content:\n{\n    \"toolname\" : \"toolname\"\n    \"parameters\" : [\n        {\n            \"parameter_name\" : \"name\"\n            \"parameter_value\" : \"value\"\n        },\n        ......\n    ]\n}\neg.\n{\n    \"tool_name\" : \"get_github_repo\",\n    \"parameters\" : [\n        {\n            \"parameter_name\" : \"owner\"\n            \"parameter_value\" : \"harry\"\n        },\n        {\n            \"parameter_name\" : \"reponame\"\n            \"parameter_value\" : \"myrepo\"\n        }\n    ]\n}"
        },
        {
            "Ai Message": "{\"tool_name\":\"kubectl_delete\",\"parameters\":[{\"parameter_name\":\"resourceType\",\"parameter_value\":\"pod\"},{\"parameter_name\":\"name\",\"parameter_value\":\"test-pod-123\"},{\"parameter_name\":\"namespace\",\"parameter_value\":\"dev\"}]}"
        },
        {
            "Ai Message": "[TextContent(type'text', text'{\n  \"error\": \"Resource not found\",\n  \"status\": \"not_found\"\n}', annotationsNone)]"
        },
        {
            "Human Message": "Step: OBSERVATION\nNow you need to finish the final step of observation to finish the subtask."
        },
        {
            "Ai Message": "[\"Delete pod test-pod-123 in the dev namespace failed because it does not exist, verify by listing all pods in the dev namespace to confirm its deletion.\"]"
        },
        {
            "Human Message": "Current subtask is:\nDelete pod test-pod-123 in the dev namespace failed because it does not exist, verify by listing all pods in the dev namespace to confirm its deletion."
        },
        {
            "Human Message": "Step : QUERY_EXECUTABLE\nNow you've got a new subtask to finish, you need to select the suitable agent to finish it."
        },
        {
            "Ai Message": "kubernetes_agent"
        },
        {
            "Ai Message": "agent_response : {\"tool_name\": \"kubectl_get\", \"parameters\": {\"resourceType\": \"pods\", \"name\": \"test-pod-123\", \"namespace\": \"dev\", \"output\": \"json\", \"allNamespaces\": false}}"
        },
        {
            "Human Message": "Step : EXECUTE\nYour agent have told you all the information needed to finish your task. Now all you need to do is to provide all the necessary parameters to your agent.\noutput in a strict JSON dict format and ONLY the json dict content:\n{\n    \"toolname\" : \"toolname\"\n    \"parameters\" : [\n        {\n            \"parameter_name\" : \"name\"\n            \"parameter_value\" : \"value\"\n        },\n        ......\n    ]\n}\neg.\n{\n    \"tool_name\" : \"get_github_repo\",\n    \"parameters\" : [\n        {\n            \"parameter_name\" : \"owner\"\n            \"parameter_value\" : \"harry\"\n        },\n        {\n            \"parameter_name\" : \"reponame\"\n            \"parameter_value\" : \"myrepo\"\n        }\n    ]\n}"
        },
        {
            "Ai Message": "{\"tool_name\":\"kubectl_get\",\"parameters\":[{\"parameter_name\":\"resourceType\",\"parameter_value\":\"pods\"},{\"parameter_name\":\"namespace\",\"parameter_value\":\"dev\"},{\"parameter_name\":\"output\",\"parameter_value\":\"json\"}]}"
        },
        {
            "Ai Message": "[TextContent(type'text', text'{\n  \"items\": []\n}', annotationsNone)]"
        },
        {
            "Human Message": "Step: OBSERVATION\nNow you need to finish the final step of observation to finish the subtask."
        },
        
'''

pd = '''
SUCCESS. pod does not exist.
'''

prompt_template = ChatPromptTemplate.from_template(PLANNING_JUDGE_PROMPT)
llm = ChatOpenAI(base_url=base_url, 
        api_key=api_key, 
        model=model_name, 
        temperature=temperature, 
        )
messages = prompt_template.format_messages(input_str=input_str, pd=pd)
response = llm.invoke(messages)
print(response.content)