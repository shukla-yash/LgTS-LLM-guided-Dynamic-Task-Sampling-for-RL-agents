from LMgTS.environments.graphical import GraphEnv
from LMgTS.environments.fireman_graph_env import FiremanGraphEnv
from LMgTS.environments.minigrid_graph_env2 import MinigridGraphEnv
import gymnasium as gym
from LMgTS.utils.gpt_util import *
from LMgTS.utils.util import *
from LMgTS.agents.ppo import PPO, RolloutBuffer
import numpy as np
import os
import json
import pickle
import yaml


environments = ['minigrid-ninerooms', 'fireman-easy']

if __name__ == '__main__':
    # Setting up VPN
    # socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 7891)
    # socket.socket = socks.socksocket

    # Test initialization

    with open('LMgTS/config.yaml', 'r') as f:
        config = yaml.safe_load(f) 

    experiment_conf = config['experiment'] 
    training_conf = config['training']
    agent_conf = config['ppo']
    environment_conf = config['environment']
    llm_conf = config['llm'] 

    if experiment_conf['environment'] == 'minigrid-ninerooms':

        graph_env = MinigridGraphEnv(
            K=experiment_conf['num_paths'],
            seed=experiment_conf['random_seed'],
            ts_algo=experiment_conf['TS_algorithm'],

            episodes_in_each_iter=training_conf['episodes_in_each_iter'],
            ppo_update_timestep=agent_conf['update_episode'],
            max_ep_len=environment_conf['max_ep_len'],
            llm_prompt_file='prompts/fireman_prompt.txt',
            llm_model=llm_conf['model'],
            llm_response_file=llm_conf['response_file'],
            load_response=llm_conf['load_response'],
            render_mode=False,
            conf=experiment_conf,
        ) 
    
    elif experiment_conf['environment'] == 'fireman-easy':
        graph_env = FiremanGraphEnv(
            K=experiment_conf['num_paths'],
            seed=experiment_conf['random_seed'],
            ts_algo=experiment_conf['TS_algorithm'],

            episodes_in_each_iter=training_conf['episodes_in_each_iter'],
            ppo_update_timestep=agent_conf['update_episode'],
            max_ep_len=environment_conf['max_ep_len'],
            render_mode=False,

            llm_prompt_file='/home/airlab/Downloads/AAMAS2024_supplemental_977/Supplementary/LMgTS/prompts/fireman_prompt.txt',
            llm_model=llm_conf['model'],
            llm_response_file=llm_conf['response_file'],
            load_response=llm_conf['load_response'],
            openai_key=llm_conf['api_key'],

            conf=experiment_conf,
        ) 
    agents = graph_env.init_agents(agent_conf)
    graph_env.train(agents)
    graph_env.test(agents)
