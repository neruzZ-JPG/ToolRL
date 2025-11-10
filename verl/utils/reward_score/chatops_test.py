from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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
3. The planning should not contain any unnecessary steps. If user request is finished, the planning should begin with "SUCCESS"; If the user request cannot be finished, the planning should begin with "FAIL".
4. The planning should be as short and concise as possible. And it should not contain any previously-executed steps.
##### 
Your evaluation score should be a float number between 0 and 1. 
Output the number and ONLY the number!
'''

base_url = "https://vip.apiyi.com/v1"
api_key = "sk-5PYQRpTeWXyM9ibd96B5737aFdCc47B1B89a3937F6447eEe"
model_name = "gpt-4.1-nano"

input_str = '''
{
        "System Message": "You're a professional Devops engineer, leading a team with several devops agents each of whom is in charge of one devops tools such as github. \nYour main goal is to analyze the user's request, decompose it into subtasks and lead your team to finish the request step by step.\n#####\nThere are the agents:\n[\n    {\n        \"name\" : \"docker_agent\",\n        \"description\" : \"docker agent for docker tasks\",\n        \"url\" : \"http://localhost:8000\"\n    },\n    {\n        \"name\" : \"github_agent\",\n        \"description\" : \"github agent for github tasks\",\n        \"url\" : \"http://localhost:8001\"\n    },\n    {\n        \"name\" : \"gitlab_agent\",\n        \"description\" : \"gitlab agent for gitlab tasks\",\n        \"url\" : \"http://localhost:8002\"\n    },\n\n    {\n        \"name\" : \"jenkins_agent\",\n        \"description\" : \"jenkins agent for jenkins tasks\",\n        \"url\" : \"http://localhost:8003\"\n    },\n    {\n        \"name\" : \"kubernetes_agent\",\n        \"description\" : \"kubernetes agent for kubernetes/k8s tasks\",\n        \"url\" : \"http://localhost:8004\"\n    }\n]\n#####\nGo through these 3 steps to solve a subtask:\n1. QUERY_EXECUTABLE \nuery the agents if the task is executable.If so, what information do they need to finish it.\n- Not all agents should participate in this task, choose the most suitable one.\n- Respond with the name of the agent, and only the name of it.\n- For example, if your task is to get information about a dog, and there's a 'pet_agent' in the agent list. Although there might be a lot of information of the agent in the list, you only need to respond with its name: pet_agent.\n2. EXECUTE\nYour agent have told you all the information needed to finish your task, including the tool name and the required parameters. Now all you need to do is to provide the tool name as well as all the necessary parameters to your agent.\n- For example, if the task is \"How long does a Beagle live?\".Pet_agent tells you that the task can be finished with one of his tools called \"query_dog_info\", and you need to provide the species of the dog you want to query about. Then you may pass the task as \"query the average lifespan of a Beagle dog using query_dog_info\".\n- One execution or query at one time.\n3. OBSERVATION\nBased on the current state, determine whether the USER_REQUEST(not the subtask) has been fulfilled. \n- If you believe the request has been successfully completed, return 'SUCCESS' followed by a response to the user. For example: 'SUCCESS. The user's age is 18.' By returning 'SUCCESS' you can end the whole workflow.\n- If the current workflow doesn't align with the user request, return 'FAIL' and explain the reason. For instance, if the user asks for the weather in New York but the current state is searching for a bakery's location, continuing the search is incorrect. You should stop the unproductive effort by replying 'FAIL. Wrong workflow.' Returning 'FAIL' will end the whole workflow too.\n- If the request is not yet finished and more actions are required, formulate multiple plans for the next step. Keep in mind that:\n    - Plans must be formulated based on existing information, making full use of the available information and avoiding repeating the wrong actions that have already been taken.\n    - Each plan represents a different solution path and will be executed independently, not sequentially. This means Plans A and B are designed to be mutually exclusive, with neither having access to the other's information.\n    - Present the plans as a JSON - formatted array. \n    - For example, if the user wants to find the name, age, and sign - up date of a user given their ID, and you've found the user's name (say, Harry), you should return ['The user's name is Harry, find his age.', 'The user's name is Harry, find out when he signed up.']\"\nRESPONSE STRUCTURE:\nOne of the following three responses:\n    - 'SUCCESS, [success information]'\n    - 'FAIL, [fail information]'\n    - A JSON - formatted array of subtasks, with only the structure content without any json identifier.\nYour observation will be strictly parsed, and you must strictly follow the response structure."
    },
    {
        "Human Message": "the user request is:\nI need to verify the details of the volume named 'data_backup_volume' used for storing Docker container data on the local machine, including its driver and creation date."
    },
    {
        "Human Message": "Current subtask is:\nStart by identifying the volume's details using Docker commands. Think about which command can provide comprehensive info about volumes, and focus on how to find the driver and creation date specifically."
    },
    {
        "Human Message": "Step : QUERY_EXECUTABLE\nNow you've got a new subtask to finish, you need to select the suitable agent to finish it."
    },
    {
        "Ai Message": "docker_agent"
    },
    {
        "Ai Message": "The work_agent is not available."
    },
    {
        "Human Message": "Step: OBSERVATION\nNow you need to finish the final step of observation to finish the subtask."
    },
    {
        "Ai Message": "[\n\"Plan A: Use docker_agent to execute docker volume inspect data_backup_volume and read the Driver and CreatedAt values from the resulting JSON output.\",\n\"Plan B: If Plan A indicates the volume does not exist or is not listed, first run docker volume ls --format '{{.Name}}' to verify presence, and if data_backup_volume is listed, then run docker volume inspect data_backup_volume to obtain Driver and CreatedAt.\",\n\"Plan C: Verify the correct Docker context is active by running docker context show and, if needed, switch to the appropriate context, then run docker volume inspect data_backup_volume to retrieve the Driver and CreatedAt fields.\"\n]"
    },
    {
        "Human Message": "Current subtask is:\nPlan B: If Plan A indicates the volume does not exist or is not listed, first run docker volume ls --format '{{.Name}}' to verify presence, and if data_backup_volume is listed, then run docker volume inspect data_backup_volume to obtain Driver and CreatedAt."
    },
    {
        "Human Message": "Step : QUERY_EXECUTABLE\nNow you've got a new subtask to finish, you need to select the suitable agent to finish it."
    },
    {
        "Ai Message": "docker_agent"
    },
    {
        "Ai Message": "agent_response : \"{\\n  \\\"tool_name\\\": \\\"list_volumes\\\",\\n  \\\"parameters\\\": {}\\n}\""
    },
    {
        "Human Message": "Step : EXECUTE\nYour agent have told you all the information needed to finish your task. Now all you need to do is to provide all the necessary parameters to your agent."
    },
    {
        "Ai Message": "{\"tool_name\":\"list_volumes\",\"parameters\":{}}"
    },
    {
        "Ai Message": "{\"content\": \"{'status_code': 200, 'content': b'{\\\"Volumes\\\":[{\\\"CreatedAt\\\":\\\"2025-10-17T21:47:25+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"Purpose\\\":\\\"Q2 Campaign\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/CustomerDataBackup/_data\\\",\\\"Name\\\":\\\"CustomerDataBackup\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-18T01:43:21+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"clientID\\\":\\\"1234\\\",\\\"project\\\":\\\"alpha\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/client-data-volume/_data\\\",\\\"Name\\\":\\\"client-data-volume\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:23:37+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"purpose\\\":\\\"client onboarding\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/project-data-01/_data\\\",\\\"Name\\\":\\\"project-data-01\\\",\\\"Options\\\":{},\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:28:29+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"env\\\":\\\"production\\\",\\\"project\\\":\\\"order-service\\\",\\\"purpose\\\":\\\"db-data\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/orders_db_data_vol/_data\\\",\\\"Name\\\":\\\"orders_db_data_vol\\\",\\\"Options\\\":{},\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:42:48+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"project\\\":\\\"website\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/user_uploads/_data\\\",\\\"Name\\\":\\\"user_uploads\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:19:42+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"client_project\\\":\\\"true\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/project_data_vol/_data\\\",\\\"Name\\\":\\\"project_data_vol\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:36:19+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"purpose\\\":\\\"customer information\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/customer-data/_data\\\",\\\"Name\\\":\\\"customer-data\\\",\\\"Options\\\":{},\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:55:10+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"purpose\\\":\\\"client_data\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/client_data_volume/_data\\\",\\\"Name\\\":\\\"client_data_volume\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T22:07:06+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"purpose\\\":\\\"e-commerce\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/customer_data/_data\\\",\\\"Name\\\":\\\"customer_data\\\",\\\"Options\\\":{},\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T19:02:04+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":null,\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/jenkins-data/_data\\\",\\\"Name\\\":\\\"jenkins-data\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:24:27+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"purpose\\\":\\\"client onboarding\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/project_data_volume/_data\\\",\\\"Name\\\":\\\"project_data_volume\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:32:53+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"project\\\":\\\"alpha\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/ProjectDataVolume/_data\\\",\\\"Name\\\":\\\"ProjectDataVolume\\\",\\\"Options\\\":{},\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-18T02:20:15+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"coordinates\\\":\\\"34.0522N,118.2437W\\\",\\\"project\\\":\\\"SV2024\\\",\\\"region\\\":\\\"Green Valley\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/sv2024_greenvalley/_data\\\",\\\"Name\\\":\\\"sv2024_greenvalley\\\",\\\"Options\\\":{\\\"device\\\":\\\"/var/lib/docker/volumes/green_valley_sv2024\\\",\\\"o\\\":\\\"bind\\\",\\\"type\\\":\\\"none\\\"},\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:17:03+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"service\\\":\\\"backup\\\",\\\"year\\\":\\\"2024\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/project_backup_2024/_data\\\",\\\"Name\\\":\\\"project_backup_2024\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T21:29:53+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"customer\\\":\\\"AcmeCorp\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/CustomerDataVolume/_data\\\",\\\"Name\\\":\\\"CustomerDataVolume\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"},{\\\"CreatedAt\\\":\\\"2025-10-17T23:49:45+08:00\\\",\\\"Driver\\\":\\\"local\\\",\\\"Labels\\\":{\\\"project\\\":\\\"project_data\\\",\\\"purpose\\\":\\\"data-analysis\\\"},\\\"Mountpoint\\\":\\\"/var/snap/docker/common/var-lib-docker/volumes/project_data/_data\\\",\\\"Name\\\":\\\"project_data\\\",\\\"Options\\\":null,\\\"Scope\\\":\\\"local\\\"}],\\\"Warnings\\\":null}\\\\n'}\", \"additional_kwargs\": {}, \"response_metadata\": {}, \"type\": \"tool\", \"name\": \"http_request\", \"id\": \"5aca5445-a6ce-4a79-beb5-70a1455a0610\", \"tool_call_id\": \"call_4NCQH6dji8NRt137vi13qRo5\", \"artifact\": null, \"status\": \"success\"}"
    },
    {
        "Human Message": "Step: OBSERVATION\nNow you need to finish the final step of observation to finish the subtask."
    },
    {
        "Ai Message": "[\n  \"Plan D: Re-verify presence by running docker volume ls --format '{{.Name}}'. If data_backup_volume appears, run docker volume inspect data_backup_volume to obtain Driver and CreatedAt.\",\n  \"Plan E: If data_backup_volume is not listed, search for similarly named volumes (e.g., data_backup, backup_data, CustomerDataBackup) and inspect any matches to determine driver and creation date.\",\n  \"Plan F: If the exact volume data_backup_volume does not exist and the user requires it, create the volume with docker volume create data_backup_volume, then run docker volume inspect data_backup_volume to retrieve Driver and CreatedAt.\"\n]"
    },
    {
        "Human Message": "Current subtask is:\nPlan D: Re-verify presence by running docker volume ls --format '{{.Name}}'. If data_backup_volume appears, run docker volume inspect data_backup_volume to obtain Driver and CreatedAt."
    },
    {
        "Human Message": "Step : QUERY_EXECUTABLE\nNow you've got a new subtask to finish, you need to select the suitable agent to finish it."
    },
    {
        "Ai Message": "docker_agent"
    },
    {
        "Ai Message": "The work_agent is not available."
    },
    {
        "Human Message": "Step: OBSERVATION\nNow you need to finish the final step of observation to finish the subtask."
    },
    {
        "Ai Message": "[\n  \"Plan D: Re-run docker volume ls --format '{{.Name}}' and if data_backup_volume appears, run docker volume inspect data_backup_volume to obtain Driver and CreatedAt.\",\n  \"Plan E: If data_backup_volume is not listed, search for similarly named volumes (e.g., data_backup, backup_data, CustomerDataBackup) and inspect any matches to determine Driver and CreatedAt.\",\n  \"Plan F: If the exact volume does not exist and the user requires it, create the volume with docker volume create data_backup_volume, then run docker volume inspect data_backup_volume to retrieve Driver and CreatedAt.\"\n]"
    },
    {
        "Human Message": "Current subtask is:\nPlan E: If data_backup_volume is not listed, search for similarly named volumes (e.g., data_backup, backup_data, CustomerDataBackup) and inspect any matches to determine Driver and CreatedAt."
    },
    {
        "Human Message": "Step : QUERY_EXECUTABLE\nNow you've got a new subtask to finish, you need to select the suitable agent to finish it."
    },
    {
        "Ai Message": "docker_agent"
    },
    {
        "Ai Message": "The work_agent is not available."
    },
    {
        "Human Message": "Step: OBSERVATION\nNow you need to finish the final step of observation to finish the subtask."
    },
'''

pd = '''
FAIL. 
'''

prompt_template = ChatPromptTemplate.from_template(PLANNING_JUDGE_PROMPT)
llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model_name)
messages = prompt_template.format_messages(input_str=input_str, pd=pd)
response = llm.invoke(messages)
print(response.content)