# from environments.graphical import GraphEnv
# import openai
import gymnasium as gym
import numpy as np
import random
import socks 
import socket 
import os 
import json 
import yaml 
import imageio
import pickle

from LMgTS.utils.gpt_util import *
from LMgTS.utils.util import *
from LMgTS.utils.consulters import GptConsulter, ConsulterException, LLMException, LLMConsulter, LLamaConsulter

from LMgTS.agents.ppo import PPO, RolloutBuffer
from LMgTS.agents.TSagents.agts import AGTS

from LMgTS.environments.graphical import GraphEnv

from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper



class MinigridGraphEnv(GraphEnv):
    """
    Graph environment for Minigrid domain. Each node is a high-level state and each edge is a sub-task.

    High-Level States:
        0: At(Room1)
        1: PickedUp(EasyKey)
        2: PickedUp(HardKey)
        3: Opened(Door)
        4: At(Goal)

    Sub-tasks:
        0: 0-1: NineRoomsRoom2EasyKey, Get easy key starting from Room 1.
        1: 0-2: NineRoomsRoom2HardKey, Get hard key starting from Room 1.
        2: 0-3: NineRoomsRoom2Door, Open door starting from Room 1.
        3: 0-4: NineRoomsRoom2Goal, Reach goal starting from Room 1
        4: 1-0: NineRoomsEasyKey2Room, Return Room1 starting from Easy Key room
        5: 1-2: NineRoomsEasyKey2HardKey, Get hard key starting from Easy Key room
        6: 1-3: NineRoomsEasyKey2Door, Open Door starting from Easy Key room
        7: 1-4: NineRoomsEasyKey2Goal, Reach Goal starting from Easy Key room
        8: 2-0: NineRoomsHardKey2Room, Return Room1 starting from Hard Key room
        9: 2-1: NineRoomsHardKey2EasyKey, Get easy key starting from Hard Key room
        10: 2-3: NineRoomsHardKey2Door, Open Door starting from Hard Key room
        11: 2-4: NineRoomsHardKey2Goal, Reach Goal starting from Hard Key room
        12: 3-0: NineRoomsDoor2Room, Return Room1 starting from Door
        13: 3-1: NineRoomsDoor2Easy, Return Easy Key room starting from Door
        14: 3-2: NineRoomsDoor2Hard, Return Hard Key room starting from Door
        15: 3-4: NineRoomsDoorGoal, Reach Goal
    """

    def __init__(self, experiment_name='minigrid_agts',
                 K=4,  seed=0, 
                 openai_key='',
                 max_training_timestep=5e7,
                 ppo_update_timestep=2000,
                 episodes_in_each_iter=500,
                 max_ep_len=500, render_mode=False, 
                 save_model_freq=100000,
                 load_response=False, 
                 ts_algo='AGTS',
                 llm_prompt_file='/prompts/minigrid_prompt.txt',
                 llm_model='gpt-3.5-turbo', llm_max_consult_time=5,
                 llm_response_file='/responses/minigrid.txt',
                 conf={}):
        # super().__init__(num_agents, total_paths, goal_ordering)
        self.K = K
        self.episodes_per_iter = episodes_in_each_iter
        self.max_ep_len = max_ep_len
        self.max_training_timestep = max_training_timestep

        # Environment initializations
        self.env_names = [
            'MiniGrid-NineRooms-Room2EasyKey-v0', 'MiniGrid-NineRooms-Room2HardKey-v0',
            'MiniGrid-NineRooms-Room2Door-v0', 'MiniGrid-NineRooms-Room2Goal-v0',
            'MiniGrid-NineRooms-EasyKey2Room-v0', 'MiniGrid-NineRooms-EasyKey2HardKey-v0',
            'MiniGrid-NineRooms-EasyKey2Door-v0',  'MiniGrid-NineRooms-EasyKey2Goal-v0',
            'MiniGrid-NineRooms-HardKey2Room-v0', 'MiniGrid-NineRooms-HardKey2EasyKey-v0', 
            'MiniGrid-NineRooms-HardKey2Door-v0', 'MiniGrid-NineRooms-HardKey2Goal-v0',
            'MiniGrid-NineRooms-Door2Room-v0', 'MiniGrid-NineRooms-Door2EasyKey-v0',
            'MiniGrid-NineRooms-Door2HardKey-v0', 'MiniGrid-NineRooms-Door2Goal-v0'
        ]

        self.state_map = {
            'At(Start)': 0, 
            'Start': 0,

            'EasyKey': 1,
            'PickedUp(EasyKey)': 1, 
            'Unlocked(EasyKey)': 1,
            'At(EasyKey)': 1,
            
            'HardKey': 2,
            'Unlocked(HardKey)': 2,
            'PickedUp(HardKey)': 2,
            'At(HardKey)': 2, 
            
            'Door': 3,
            'PickedUp(Door)': 3,
            'Unlocked(Door)': 3,
            'At(Door)': 3, 

            'Goal': 4,
            'PickedUp(Goal)': 4, 
            'Unlocked(Goal)': 4,
            'At(Goal)': 4
        }

        self.edges = {
            (0, 0): -1,
            (0, 1): 0,
            (0, 2): 1,
            (0, 3): 2,
            (0, 4): 3,
            (1, 0): 4,
            (1, 1): -1,
            (1, 2): 5,
            (1, 3): 6,
            (1, 4): 7,
            (2, 0): 8,
            (2, 1): 9,
            (2, 2): -1,
            (2, 3): 10,
            (2, 4): 11,
            (3, 0): 12,
            (3, 1): 13,
            (3, 2): 14,
            (3, 3): -1,
            (3, 4): 15
        }

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.edges_dfa = {}

        self.adjacency_mat = [[0 for _ in range(5)] for _ in range(5)] 
        self.sequences = []

        self.envs = [None for _ in self.env_names]
        self.env_counts = [0 for _ in self.envs]

        self.learned = [False for _ in self.env_names]
        self.init_envs(render_mode)
        self.goal_ordering = []
        self.update_timestep = ppo_update_timestep

        self.current_state = [0 for _ in range(self.K)]

        self.response_file = llm_response_file

        # LLM initializations
        if not load_response:
            self.llm = LLamaConsulter(
                max_consult_time=llm_max_consult_time,
                gpt_model=llm_model,
                prompt_file=llm_prompt_file
            )
        else:
            self.llm = LLamaConsulter.load_response(llm_response_file)

        self.active_tasks = []

        self.init_graph()   
        if ts_algo == 'AGTS':
            self.ts_model = AGTS(edges=self.edges_dfa, num_states=5, num_envs=16, active_tasks=self.active_tasks,goal_state=4)

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
            'print_freq': 5000,
            'max_train_timestep': 3e7,
            'log_freq': 100,
            
            'action_std_decay_freq': -1,
            'action_std_decay_rate': 0,
            'min_action_std': 0,
            
            'random_seed': seed,
            'model_dir': 'models/minigrid',
            'log_dir': 'logs'
        }

        # print(len(self.goal_ordering))
    def init_envs(self, render_mode=False):
        print('------------------------------------------------------------------')
        print('Initializing {} environments.'.format(len(self.env_names)))
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
            self.envs[env_id] = FullyObsWrapper(env)

    def parse_path(self, path):
        """
        Parse a single path given by LLM. Returns the corresponding sequence of tasks if valid,
        otherwise raises an error that could be passed directly to LLM.

        :param path:
        :return:
        """
        if self.state_map[path[-1]] != 4:
            return False
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
        for i in range(5):
            for j in range(5):
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
                    print('{} paths successfully parsed!!!'.format(len(self.goal_ordering)))
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
            video_env = RGBImgObsWrapper(gym.make(self.env_names[task_num], max_episode_steps=self.max_ep_len))
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
            env = FullyObsWrapper(env)
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

        # print(success_rate_buffer)
        return current_path_interactions, success_rate_buffer, per_task_sucess_rates