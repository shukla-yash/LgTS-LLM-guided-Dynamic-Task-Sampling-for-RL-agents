# Large Language Models guided Teacher-Student Learning

## Curriculum Learning using LLMs
Recent advancements in reasoning abilities of Large Language Mod-
els (LLM) has promoted their usage in problems that require high-
level planning for robots and artificial agents. LgTS (LLM-guided Teacher-Student learning) is a novel approach
that explores the planning abilities of LLMs to provide a graphical
representation of the sub-goals to a reinforcement learning (RL)
agent that does not have access to the transition dynamics of the
environment. The RL agent uses Teacher-Student learning algo-
rithm to learn a set of successful policies for reaching the goal state
from the start state while simultaneously minimizing the number
of environmental interactions. Our approach does not assume access to a propreitary
or a fine-tuned LLM, nor does it require pre-trained policies that
achieve the sub-goals proposed by the LLM. Our experiments demonstrate that generating a graphical structure
of sub-goals helps in learning policies for the LLM proposed sub-
goals and the Teacher-Student learning algorithm minimizes the
number of environment interactions when the transition dynamics
are unknown.

The requirements are listed in the file: requirements.txt

To install: `pip install -r requirements.txt`

The experiments were conducted using a 64-bit Linux Machine, having Intel(R) Core(TM) i9-9940X CPU @ 3.30GHz processor and 126GB RAM memory. The maximum duration for running the experiments was set at 24 hours.

## Step 1a: Running the Minigrid Experiments:

To run the minigrid experiment:

- set up `environment` as 'minigrid-ninerooms' in `config.yaml` 
- Copy the environments: `$ cp -r LMgTS/minigrid {path-to-miniconda3/anaconda}/envs/{Name-of-env}/lib/{python-version}/site-packages/minigrid` 
- set up `prompt_file` with the path to your prompt file in `config.yaml`
- set up `response_file` with the path to your response in `config.yaml` 
- set up your OPEN API token as `api_key` in `config.yaml`. More info here: https://openai.com/blog/openai-api
- run `$ python -m LMgTS.main.py` to start training 
- The training runs will save the all the `logs` in a compressed `.npz` file

## Step 1b: Running the Search-and-rescue Experiments:

To run the search and rescue experiment:
- set up `environment` as 'fireman-easy' in `config.yaml` 
- copy the environments: `$ cp -r LMgTS/minigrid {path-to-miniconda3/anaconda}/envs/{Name-of-env}/lib/{python-version}/site-packages/minigrid` 
- set up `prompt_file` with the path to your prompt file in `config.yaml` 
- set up `response_file` with the path to your response in `config.yaml` 
- set up your OpenAI API token as `api_key` in `config.yaml`
- run `$ python -m LMgTS.main.py` to start training
- The training runs will save the all the `logs` in a compressed `.npz` file
