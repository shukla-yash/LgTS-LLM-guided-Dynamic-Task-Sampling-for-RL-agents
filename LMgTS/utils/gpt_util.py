import os 
import openai
import ast

openai.api_key = os.getenv('OPENAI_API_KEY')
# print(os.getenv('OPENAI_API_KEY'))

default_states = """
(S_1) Face the bread loaf 
(S_2) Face the bread slice
(S_3) Face the toaster 
(S_4) Put the bread slice in the toaster 
(S_5) Pick up the bread loaf
(S_6) Pick up the bread slice 
(S_7) Switch on the toaster 
(S_8) Slice the bread loaf 
"""


def generate_prompt(states=default_states, num_paths=1):
    prompt = """You are a reinforcement learning agent. The final goal for the agent is to cut a slice from the bread loaf and cook the slice in the toaster. The agent can reach the following unordered set of high-level states: 

    {states}

    Your task as a reinforcement learning agent is to produce a relevant chronological sequences (i.e. S_i, S_j, S_k) in which the above high-level states should be visited. Respond only using the given high-level states and discard irrelevant high-level states. It is possible that all the high-level states are not relevant. Do not generate new states. Your response should have numbers to denote the sequence in which the tasks should be done. Your response should begin with, The correct chronological sequence is: This statement should be followed by sentences describing the correct order. Please respond 10 most likely sequences only, without additional descriptions. Do not reply any words or sentences except the paths. Each sequence should be formatted in (S_i, S_j, S_k, ...) grouped in one line. It is allowed to reply identical paths."""
    return prompt 


gpt_model_list = ['text-davinci-003', 'gpt-3.5-turbo', 'gpt-4', 
                  'llama-2-70b-chat', 'llmam-2-13b-chat', 'llama-2-7b-chat']


def ask_gpt(gpt_model='text-davinci-003', temperature=0.6, state_descriptions=default_states, num_paths=1):
    assert gpt_model in gpt_model_list

    prompt = generate_prompt(state_descriptions, num_paths)

    messages = [
        {'role': "system", "content": "You are a robot that wants to cut a slice from the bread load and cook the slice in the toaster. You break this task into several sub-tasks. You have to figure out the order to complete these sub-tasks. The user would tell you the sub-tasks in the format of (S_i) then sub-task description. Your goal is to choose {num_paths} reasonable sequences of tasks to finish your final task. Please reply each sequence in one line and new sequences in new lines."},
        {'role': "system", "content": "It is possible that the tasks seem irrelevant. Please respond the most likely {num_paths} sequences exactly. Respond the numbers to denote the task. Do not reply words or sentences. Do not discard tasks unless you are extremely confident that it is useless. Do not generate new sub-tasks. You are allowed to generate identical sequences."},
        {"role": "user", "content": default_states}
    ]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages
    )
    return response


def get_list_of_paths(gpt_model='gpt-3.5-turbo', propmt_file='./prompts/minigrid_prompt.txt') -> list:
    assert gpt_model in gpt_model_list
    with open(propmt_file, 'r') as f:
        prompt = f.read()
    messages = [
        {'role': "user", 'content': prompt}
    ]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages
    )
    return string_to_list(response['choices'][0]['message']['content'])


def string_to_list(response: str) -> list:
    # return ast.literal_eval(response)
    start = response.find('[')
    end = response.find(']')+1
    if not (response.count('[') == 1 and response.count(']') == 1):
        raise 'The response should be a 1-D list.'
    substr = response[start:end]
    # print(isinstance(ast.literal_eval(substr), list))
    return ast.literal_eval(substr)

def string_to_lists(response: str) -> list:
    stack = 0
    ret_list = []
    start, end = 0, 0
    for i in range(len(response)):
        if response[i] == '[':
            stack += 1
            if stack == 1:
                start = i
        elif response[i] == ']':
            stack -= 1
            if stack == 0:
                end = i+1
                substr = response[start:end]
                ret_list.append(ast.literal_eval(substr))

    if stack != 0:
        raise Exception('Parsing error: found inconsistent number of "[" and "]".')

    return ret_list

