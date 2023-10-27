# from environments.graphical import GraphEnv
# import openai
import gymnasium as gym
import numpy as np
import socks 
import socket 
import os 
import yaml 
import pickle

from LMgTS.utils.gpt_util import *
from LMgTS.utils.util import *
from LMgTS.utils.consulters import GptConsulter, ConsulterException, LLMException, LLMConsulter, LLamaConsulter

from LMgTS.agents.ppo import PPO, RolloutBuffer
from LMgTS.agents.TSagents.agts import AGTS

from LMgTS.environments.graphical import GraphEnv

# from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
import firemangrid



class FiremanGraphEnv(GraphEnv):
    """
    Graph environment for Minigrid domain. Each node is a high-level state and each edge is a sub-task.

    High-Level States:
        0: At(Start)
        1: PickedUp(Key)
        2: Unlocked(Door)
        3: PickedUp(FireExtinguisher)
        4: Extinguished(Fire)
        5: Moved(Debris)
        6: Saved(Survivor)

    Sub-tasks:
        0: 0-1: Start2Key
        1: 0-2: Start2Door 
        2: 0-3: Start2FireExtinguisher 
        3: 0-4: Start2Fire
        4: 0-5: Start2Debris
        5: 0-6: Start2Survivor

        6: 1-0: Key2Start 
        7: 1-2: Key2Door 
        8: 1-3: Key2FireExtinguisher 
        9: 1-4: Key2Fire
        10: 1-5: Key2Debris
        11: 1-6: Key2Survivor

        12: 2-0: Door2Start
        13: 2-1: Door2Key
        14: 2-3: Door2FireExtinguisher
        15: 2-4: Door2Fire
        16: 2-5: Door2Debris
        17: 2-6: Door2Survivor

        18: 3-0: FireExtinguisher2Start
        19: 3-1: FireExtinguisher2Key
        20: 3-2: FireExtinguisher2Door
        21: 3-4: FireExtinguisher2Fire
        22: 3-5: FireExtinguisher2Debris
        23: 3-6: FireExtinguisher2Survivor

        24: 4-0: Fire2Start
        25: 4-1: Fire2Key'
        26: 4-2: Fire2Door
        27: 4-3: Fire2FireExtinguisher
        28: 4-5: Fire2Debris
        29: 4-6: Fire2Survivor

        30: 5-0: Debris2Start
        31: 5-1: Debris2Key
        32: 5-2: Debris2Door
        33: 5-3: Debris2FireExtinguisher
        34: 5-4: Debris2Fire
        35: 5-6: Debris2Survivor
    """

    def __init__(self, experiment_name='fireman_agts',
                 K=4,  seed=0, 
                 openai_key='',
                 max_training_timestep=5e7,
                 ppo_update_timestep=2000,
                 episodes_in_each_iter=500,
                 max_ep_len=500, render_mode=False, 
                 save_model_freq=100000,
                 load_response=False, 
                 ts_algo='AGTS',
                 llm_prompt_file='/prompts/fireman_prompt.txt',
                 llm_model='gpt-3.5-turbo', llm_max_consult_time=5,
                 llm_response_file='/responses/fireman.txt',
                 conf={}):
        self.K = K
        self.episodes_per_iter = episodes_in_each_iter
        self.max_ep_len = max_ep_len
        self.max_training_timestep = max_training_timestep

        # Environment initializations
        self.env_names = [
            'FiremanGrid-Start2Key-v0', 'FiremanGrid-Start2Door-v0', 'FiremanGrid-Start2FireExtinguisher-v0', 'FiremanGrid-Start2Fire-v0', 'FiremanGrid-Start2Debris-v0', 'FiremanGrid-Start2Survivor-v0',
            'FiremanGrid-Key2Start-v0', 'FiremanGrid-Key2Door-v0', 'FiremanGrid-Key2FireExtinguisher-v0', 'FiremanGrid-Key2Fire-v0', 'FiremanGrid-Key2Debris-v0', 'FiremanGrid-Key2Survivor-v0', 
            'FiremanGrid-Door2Start-v0', 'FiremanGrid-Door2Key-v0', 'FiremanGrid-Door2FireExtinguisher-v0', 'FiremanGrid-Door2Fire-v0', 'FiremanGrid-Door2Debris-v0', 'FiremanGrid-Door2Survivor-v0',
            'FiremanGrid-FireExtinguisher2Start-v0', 'FiremanGrid-FireExtinguisher2Key-v0', 'FiremanGrid-FireExtinguisher2Door-v0', 'FiremanGrid-FireExtinguisher2Fire-v0', 'FiremanGrid-FireExtinguisher2Debris-v0', 'FiremanGrid-FireExtinguisher2Survivor-v0',
            'FiremanGrid-Fire2Start-v0', 'FiremanGrid-Fire2Key-v0', 'FiremanGrid-Fire2Door-v0', 'FiremanGrid-Fire2FireExtinguisher-v0', 'FiremanGrid-Fire2Debris-v0', 'FiremanGrid-Fire2Survivor-v0',
            'FiremanGrid-Debris2Start-v0', 'FiremanGrid-Debris2Key-v0', 'FiremanGrid-Debris2Door-v0', 'FiremanGrid-Debris2FireExtinguisher-v0', 'FiremanGrid-Debris2Fire-v0', 'FiremanGrid-Debris2Survivor-v0',
            'FiremanGrid-Survivor2Start-v0', 'FiremanGrid-Survivor2Key-v0', 'FiremanGrid-Survivor2Door-v0', 'FiremanGrid-Survivor2FireExtinguisher-v0', 'FiremanGrid-Survivor2Fire-v0', 'FiremanGrid-Survivor2Debris-v0'
        ]

        self.state_map = {
            'At(Start)': 0, 
            'Start': 0,
            'PickedUp(Start)': 0,
            'Unlocked(Start)': 0,
            'Extinguished(Start)': 0,
            'Moved(Start)': 0,
            'Saved(Start)': 0,

            'Key': 1,
            'PickedUp(Key)': 1, 
            'Unlocked(Key)': 1,
            'At(Key)': 1,
            'Moved(Key)': 1,
            'Extinguished(Key)': 1,
            'Saved(Key)': 1,
            
            'Door': 2,
            'Unlocked(Door)': 2,
            'PickedUp(Door)': 2,
            'At(Door)': 2, 
            'Moved(Door)': 2,
            'Extinguished(Door)': 2,
            'Saved(Door)': 2,
            
            'FireExtinguisher': 3,
            'PickedUp(FireExtinguisher)': 3,
            'Unlocked(FireExtinguisher)': 3,
            'At(FireExtinguisher)': 3,
            'Moved(FireExtinguisher)': 3,
            'Extinguished(FireExtinguisher)': 3,
            'Saved(FireExtinguisher)': 3,

            'Fire': 4,
            'PickedUp(Fire)': 4, 
            'Unlocked(Fire)': 4,
            'At(Fire)': 4,
            'Moved(Fire)': 4,
            'Extinguished(Fire)': 4,
            'Saved(Fire)': 4,

            'Debris': 5,
            'PickedUp(Debris)': 5,
            'Unlocked(Debris)': 5,
            'At(Debris)': 5,
            'Moved(Debris)': 5,
            'Extinguished(Debris)': 5,
            'Saved(Debris)': 5,

            'Survivor': 6,
            'PickedUp(Survivor)': 6,
            'Unlocked(Survivor)': 6,
            'At(Survivor)': 6,
            'Moved(Survivor)': 6,
            'Extinguished(Survivor)': 6,
            'Saved(Survivor)': 6
        }

        self.edges = {} 
        cur_env = 0

        for i in range(7):
            for j in range(7):
                if i == j: 
                    self.edges[(i, j)] = -1
                    continue 
                self.edges[(i, j)] = cur_env
                cur_env += 1

        # print(cur_env)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.edges_dfa = {}

        self.adjacency_mat = [[0 for _ in range(7)] for _ in range(7)] 
        self.sequences = []

        self.envs = [None for _ in self.env_names]
        self.env_counts = [0 for _ in self.envs]

        # self.learned = [False for _ in self.env_names]
        self.init_envs(render_mode)
        self.goal_ordering = []
        self.update_timestep = ppo_update_timestep

        # self.current_state = [0 for _ in range(self.K)]

        self.response_file = llm_response_file

        # LLM initializations
        if not load_response:
            if llm_model in ['gpt-3.5-turbo']:
                self.llm = GptConsulter(
                    openai_key=openai_key,
                    max_consult_time=llm_max_consult_time,
                    gpt_model=llm_model,
                    prompt_file=llm_prompt_file
                )
            else:
                self.llm = LLamaConsulter(
                    max_consult_time=llm_max_consult_time,
                    gpt_model=llm_model,
                    prompt_file=llm_prompt_file
                )
        else:
            self.llm = LLamaConsulter.load_response(llm_response_file)

        self.active_tasks = []

        self.init_graph()
        self.ts_model = AGTS(edges=self.edges_dfa, num_states=7, num_envs=49, active_tasks=self.active_tasks, goal_state=6)

        self.all_ts_params = {
            'exp_name': experiment_name,

            # environment settings
            'env_names': self.env_names,
            'max_ep_len': self.max_ep_len,
            'has_continuous_action_space': False,

            # training settings
            'episodes_in_each_iter': self.episodes_per_iter,
            'update_timestep': ppo_update_timestep,
            'save_model_freq': save_model_freq, 
            'print_freq': conf['print_freq'],
            'max_train_timestep': conf['max_train_timestep'],
            'log_freq': conf['log_freq'],
            
            'action_std_decay_freq': -1,
            'action_std_decay_rate': 0,
            'min_action_std': 0,
            
            'random_seed': seed,
            'model_dir': 'models/fireman',
            'log_dir': 'logs'
        }

    def init_envs(self, render_mode=False):
        for i in self.edges.keys():
            if not isinstance(self.edges[i], int):
                continue
            if self.edges[i] == -1:
                continue
            env_id = self.edges[i]
            if render_mode:
                env = gym.make(self.env_names[env_id], max_episode_steps=self.max_ep_len, render_mode="human")
            else:
                env = gym.make(self.env_names[env_id], max_episode_steps=self.max_ep_len)
            self.envs[env_id] = env

    def parse_path(self, path):
        """
        Parse a single path given by LLM. Returns the corresponding sequence of tasks if valid,
        otherwise raises an error that could be passed directly to LLM.

        :param path:
        :return:
        """
        state_sequence = []
        edge_sequence = []
        for p in path:
            if p not in self.state_map.keys():
                return False
            state_sequence.append(self.state_map[p])

        for i in range(len(state_sequence)-1):
            if state_sequence[i] == state_sequence[i+1]:
                continue 
            self.adjacency_mat[state_sequence[i]][state_sequence[i+1]] = 1
            edge_sequence.append(self.edges[(state_sequence[i], state_sequence[i+1])])
        if edge_sequence[0] not in self.active_tasks:
            self.active_tasks.append(edge_sequence[0])
        print('In this path, the edges to be visited are: {}'.format(edge_sequence))
        return True

    def init_graph(self):
        k = 0
        while k < self.K:
            paths = self.llm.get_one_path() 
            for path in paths:
                if path in self.sequences:
                    continue
                is_path_valid = self.parse_path(path) 
                if is_path_valid:
                    self.sequences.append(path) 
                    k += 1 
                    if k == self.K:
                        break
            if k != self.K:
                self.llm.regenerate_response('Please give me more sequences.')
        for i in range(7):
            for j in range(7):
                if self.adjacency_mat[i][j] != 0:
                    self.edges_dfa[self.edges[(i, j)]] = (i, j)

    def init_paths(self):
        done = False
        while not done:
            paths = self.llm.get_one_path()
            for path in paths:
                try:
                    p = self.parse_path(path)
                    self.goal_ordering.append(p)
                    print('{} paths successfully parsed'.format(len(self.goal_ordering)))
                except LLMException as e:
                    print('Got an exception while parsing the {}th path.'.format(len(self.goal_ordering)+1))
                    try:
                        self.llm.regenerate_response(str(e))
                    except ConsulterException as e2:
                        print(e2)
                        return
                done = len(self.goal_ordering) == self.K
        self.llm.save_response(self.response_file)

    def run_one_episode(self, agent, task_num, global_timestep=0, save_video=False):
        if task_num == -1 or self.envs[task_num] is None:
            return None
        current_timesteps = 0
        current_ep_reward = 0.
        if not save_video:
            video_record = None
        else:
            video_record = []
        env = self.envs[task_num]
        if save_video:
            video_env = gym.make(self.env_names[task_num], max_episode_steps=self.max_ep_len, render_mode='human')
        state, info = env.reset()
        if save_video:
            video_state, _ = video_env.reset()
        done_buffer, reward_buffer = [], []
        cur_global_timestep = global_timestep
        while current_timesteps <= self.max_ep_len:
            if save_video:
                video_record.append(video_state['image'])
            state['image'] = np.swapaxes(state['image'], 0, 2)
            state['image'] = np.expand_dims(state['image'], axis=0)
            action = agent.choose_action(state['image'])

            state, reward, done, truncated, _ = env.step(action)
            if save_video:
                video_state, _, _, _, _ = video_env.step(action)

            current_timesteps += 1
            cur_global_timestep += 1
            current_ep_reward += reward
            agent.buffer.rewards.append(reward)
            reward_buffer.append(reward)
            agent.buffer.is_terminals.append(True if done or truncated else False)
            if cur_global_timestep % self.update_timestep == 0:
                agent.update()
            if done or truncated:
                break
        succeeded = current_ep_reward > 0.
        if save_video:
            env = gym.make(self.env_names[task_num], max_episode_steps=self.max_ep_len, render_mode='rgb_array')

        return succeeded, current_ep_reward, current_timesteps, video_record

    def init_agents(self, kwargs):
        agents = []
        for env_i in self.envs:
            if env_i is None:
                agents.append(None)
            else:
                state_dim = env_i.observation_space.shape
                action_dim = env_i.action_space.n-1
                agent = PPO(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    lr_actor=kwargs['lr_actor'],
                    lr_critic=kwargs['lr_critic'],
                    K_epochs=kwargs['num_epochs'],
                    eps_clip=kwargs['eps_clip'],
                    gamma=kwargs['gamma'],
                    has_continuous_action_space=False
                )
                agents.append(agent)
        return agents

    def choose_task(self):
        return self.ts_model.choose_task()

    def update_teacher(self, task_num, reward):
        self.ts_model.update_teacher(task_num, reward)
    
    def train(self, agents):
        self.ts_model.train(self.envs, agents, self.all_ts_params)

    def run_a_path(self, path_i, agents, log_freq=100, video_freq=100, max_path_interactions=5e5):
        '''
        Runs a path with certain maximum interactions.

        :param: path_i: index of paths to be run
        :param: agents: agents to be trained or pre-trained
        :param: log_freq: number of episodes for logging. -1 for non-logging
        :param: video_freq: number of episodes for video logging. -1 for non-logging
        :param: max_path_interactions: maximum interactions allowed for this path. -1 for infinity
        :return: 
            current_path_interactions: total interactions in this path
            success_rate_buffer: a list of success rates for goal task
            per_task_success_rates: a list of lists. 

        '''
        cur_idx = 0
        cur_task = self.goal_ordering[path_i][0]
        global_timesteps = [0 for _ in self.envs]
        total_eps = [0 for _ in self.envs]
        current_path_interactions = 0
        goal_achieved = False
        reward_buffer, success_rate_buffer = [], []
        per_task_sucess_rates = [[] for _ in self.envs]

        while True:
            done_arr, reward_arr = [], []
            current_task_converged = False
            if current_path_interactions >= max_path_interactions:
                break

            for e in range(self.episodes_per_iter):
                if current_path_interactions >= max_path_interactions: 
                    break
                if self.learned[cur_task] and cur_task != self.goal_task:
                    print('Already learned task {}.'.format(cur_task))
                    cur_idx += 1
                    cur_task = self.goal_ordering[path_i][cur_idx]
                    break

                video = (video_freq != -1) and (total_eps[cur_task] % video_freq == 0)
                succeeded, current_ep_reward, current_ep_timesteps, video_record = self.run_one_episode(
                    agents[cur_task],
                    cur_task,
                    global_timesteps[cur_task],
                    video
                )

                current_path_interactions += current_ep_timesteps
                done_arr.append(succeeded)
                reward_arr.append(current_ep_reward)
                global_timesteps[cur_task] += current_ep_timesteps
                total_eps[cur_task] += 1

                if total_eps[cur_task] % log_freq == 0:
                    print('Env {}:\t {} episodes\t {} interactions\t  average reward: {}\t  success rate: {}'.format(cur_task, total_eps[cur_task], global_timesteps[cur_task], np.mean(reward_arr[-50:]), np.mean(done_arr[-50:])))
                if video:
                    save_video(video_record, '{}/episode{}.gif'.format(self.env_names[cur_task], total_eps[cur_task]))
                
                if cur_task == self.goal_task:
                    success_rate_buffer.extend([np.mean(done_arr[max(-50, -len(done_arr)):]) for _ in range(current_ep_timesteps)])
                else:
                    success_rate_buffer.extend([0. for _ in range(current_ep_timesteps)])
                
                per_task_sucess_rates[cur_task].extend([np.mean(done_arr[max(-10, -len(done_arr)):]) for _ in range(current_ep_timesteps)])
                
                if goal_achieved:
                    if e == 0:
                        agents[cur_task].save('models/PPO_{}_{}.pth'.format(self.env_names[cur_task], total_eps[cur_task]))
                    continue

            # check if converged
                if not goal_achieved and len(reward_arr) >= 50 and np.mean(reward_arr[-50:]) >= 0.9:
                    print('{} converged!'.format(self.env_names[cur_task]))
                    agents[cur_task].save('models/PPO_{}_{}.pth'.format(self.env_names[cur_task], total_eps[cur_task]))
                    self.learned[cur_task] = True
                    cur_idx += 1
                    if cur_idx < len(self.goal_ordering[path_i]):
                        cur_task = self.goal_ordering[path_i][cur_idx]
                    else:
                        print('Path {} finished!!!'.format(path_i))
                        goal_achieved = True
                    current_task_converged = True
                    break
                
            # Update teacher using success rate in last batch
            self.update_teacher(cur_task, np.mean(done_arr))

        return current_path_interactions, success_rate_buffer, per_task_sucess_rates


# if __name__ == '__main__':
    # Setting up VPN
    # socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 7891)
    # socket.socket = socks.socksocket

    # print(config)
    # agents = minigrid_graph_env.init_agents(config['ppo'])
    # minigrid_graph_env.train(agents)
    # agent = agents[1]
    # # print(type(agent))
    # # minigrid_graph_env.run_one_episode(agent, 0)
    # num_success = 0
    #
    # global_timesteps = [0 for _ in agents]
    #
    # for t in range(10000):
    #     if t % 2000 == 0:
    #         succeeded, cur_ep_timesteps, video = minigrid_graph_env.run_one_episode(agent, 1, global_timesteps[1], True)
    #         save_video(video, 'environment{}episode{}.gif'.format(1, t))
    #     else:
    #         succeeded, cur_ep_timesteps, _ = minigrid_graph_env.run_one_episode(agent, 1, global_timesteps[1])
    #     num_success += succeeded
    #     global_timesteps[1] += cur_ep_timesteps
    #     if t % 100 == 0:
    #         print('After {} episodes, {} timesteps are stored, num success becomes: {}'.format(t+1, global_timesteps[1], num_success))



    # file = '../prompts/test_util.txt'
    # with open(file, 'r') as f:
    #     content = f.read()
    # print(content)
    # print('-----------------------------------------')
    # l = string_to_lists(content)
    # print(string_to_lists(content))
    # print('-----------------------------------------')
    # print(isinstance(l, list))

    # llm_response_file = '../responses/minigrid.txt'
    # llm_prompt_file = '/'
    # with open(llm_response_file, 'r') as f:
    #     print(f.read())