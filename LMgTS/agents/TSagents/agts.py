import numpy as np
import copy
import os
import time
import pickle as pkl

from datetime import datetime
from LMgTS.agents.TSagents.basic import TSModel


class AGTS(TSModel):
    def __init__(self, edges: dict, num_states=5, num_envs=16, active_tasks=[], goal_state=4, strategy = "q-value"):
        self.num_envs = num_envs
        self.num_states = num_states
        self.strategy = strategy

        self.edges = edges

        self.active_tasks = active_tasks
        self.learned_tasks = []
        self.discarded_tasks = []

        # self.goal_task = 4 # 
        self.goal_state = goal_state

        self.student_rewards = {enum : [] for enum in range(self.num_envs)}
        # self.init_active()
        if self.strategy == "q-value":
            self.qvalue = QValue(self.num_envs, self.active_tasks)
        elif self.strategy == "ucb":
            self.qvalue = UCB(self.num_envs, self.active_tasks)
    
    def init_active(self):
        in_deg = [0 for _ in range(self.num_states)]
        # print(in_deg)
        for key, edge in self.edges.items():
            in_deg[edge[1]] += 1
        # print(in_deg)
        for key, edge in self.edges.items():
            if in_deg[edge[0]] == 0:
                self.active_tasks.append(key)

    def learned_task(self, task):
        if self.edges[task][1] == self.goal_state:
            print('Learned to reach goal')
            print('Learned task: ', self.learned_tasks)
            print('Discarded tasks: ', self.discarded_tasks)
            return True

        self.active_tasks.remove(task)
        self.qvalue.teacher_q_values[task] = -np.inf
        self.learned_tasks.append(task)

        src, dst = self.edges[task][0], self.edges[task][1]
        active_task_dst = dst
        self.edges.pop(task) 
        nodes_to_check = []
        while True:
            edges_to_discard = []
            for key,value in self.edges.items():
                if value[1] == dst:
                    edges_to_discard.append(key)
                    self.discarded_tasks.append(key)
                    if value[0] not in nodes_to_check:
                        nodes_to_check.append(value[0])
            for item in edges_to_discard:
                self.edges.pop(item)
                self.qvalue.teacher_q_values[item] = -np.inf
                if item in self.active_tasks:
                    self.active_tasks.remove(item)
            if len(nodes_to_check) == 0:
                break
            else:
                while True:
                    empty_nodes_to_check = 0
                    should_not_be_discarded = 0
                    if len(nodes_to_check) > 0:
                        dst = nodes_to_check.pop()
                    else:
                        empty_nodes_to_check = 1
                        break
                    for key,value in self.edges.items():
                        if dst == value[0]:
                            should_not_be_discarded = 1
                            break
                    if should_not_be_discarded == 1:
                        continue
                    else:
                        break
                if empty_nodes_to_check == 1:
                    break

        for key,value in self.edges.items():
            if value[0] == active_task_dst:
                self.active_tasks.append(key)
                self.qvalue.teacher_q_values[key] = 0

        return 0
    
    def update_teacher(self, env_num, reward):
        if len(self.student_rewards[env_num]) > 0:
            old_reward = self.student_rewards[env_num][-1]
        else:
            old_reward = 0
        self.student_rewards[env_num].append(reward)
        if self.strategy == "q-value":
            print("Current reward: {}, Prev reward : {}".format(reward, old_reward))
            reward = reward - old_reward
            self.qvalue.update_teacher_q_table(env_num,reward)

    def choose_task(self):
        if self.strategy == "q-value":
            task = self.qvalue.choose_task(self.active_tasks)
            print("Q value:" , self.qvalue.teacher_q_values)
        return task
    
    def train(self, envs, agents, params):
        print(self.edges)
        max_ep_len = params['max_ep_len']
        env_names = params['env_names']
        max_train_timestep = eval(params['max_train_timestep'])
        update_timestep = params['update_timestep']
        episodes_in_each_iter = params['episodes_in_each_iter']
        has_continuous_action_space = params['has_continuous_action_space']
        action_std_decay_freq = params['action_std_decay_freq']
        action_std_decay_rate = params['action_std_decay_rate']
        min_action_std = params['min_action_std']
        save_model_freq = params['save_model_freq']
        model_dir = params['model_dir']
        seed = params['random_seed']
        log_dir = params['log_dir']
        log_freq = params['log_freq']
        print_freq = params['print_freq']

        checkpoint_path_list = []
        log_fs = []

        time_spent_on_updating = 0.

        for env_name in env_names:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            cur_log_dir = os.path.join(log_dir, env_name)
            if not os.path.exists(cur_log_dir):
                os.makedirs(cur_log_dir)
            #### get number of log files in log directory
            run_num = 0
            current_num_files = next(os.walk(cur_log_dir))[2]
            run_num = len(current_num_files)
            #### create new log file for each run
            log_f_name = cur_log_dir + '/AGTS' + "_log_" + str(run_num) + ".csv"
            print("current logging run number for " + env_name + " : ", run_num)
            print("logging at : " + log_f_name)
            # logging file
            log_f = open(log_f_name,"w+")
            log_fs.append(log_f)
            log_f.write('global_timestep,task_timestep,episode,is_active,reward,converged\n')
        print('--------------------------------------------------------------------------------------------')
        print('Environments settings: ')
        print('max episode length: ', max_ep_len)
        print('has continuous action space: ', has_continuous_action_space)
        print('--------------------------------------------------------------------------------------------\n')

        print('--------------------------------------------------------------------------------------------')
        print('Training settings: ')
        print('PPO update timestep: ', update_timestep)
        print('episodes in each iteration: ', episodes_in_each_iter)
        print('action std decay frequency: ', action_std_decay_freq)
        print('action std decay rate: ', action_std_decay_rate)
        print('minimum action std: ', min_action_std)
        print('save model frequency: ', save_model_freq)
        print('--------------------------------------------------------------------------------------------\n')

        print('--------------------------------------------------------------------------------------------')
        print('Log settings: ')
        print('logging at ', log_dir)
        

        for i in range(self.num_envs):
            print('Saving PPO models at {}'.format(model_dir))
            checkpoint_path_list.append(os.path.join(model_dir, 'PPO_{}_{}.pth'.format(i, seed)))

        print("============================================================================================")

        ################# training procedure ################
        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")

        # dfa_instance = DFA()
        global_timestep = 0
        environment_total_timestep = [0 for i in range(self.num_envs)]
        environment_total_episode = [0 for i in range(self.num_envs)]

        is_final_task = 0
        final_task_performance_timesteps = []
        final_task_performance_reward = []
        final_task_performance_done = []
        time_spent_on_transer = 0. 

        all_tasks_done = [[] for _ in range(self.num_envs)]
        all_tasks_timesteps = [[] for _ in range(self.num_envs)]
        average_timesteps_learned_tasks = [0 for _ in range(self.num_envs)]

        while True:
            if global_timestep >= max_train_timestep:
                print('Max training timestep reached, stop training.')
            current_task = self.choose_task()

            env = envs[current_task]
            # env.seed(seed)
            ppo_agent = agents[current_task]

            episodes_in_current_iter = 0
            timesteps_in_current_iter = 0
            # printing and logging variables
            print_running_episodes = 0

            log_running_reward = 0
            log_running_episodes = 0
            is_task_leared = False
            done_arr = []
            reward_arr = []
            # training loop
            while episodes_in_current_iter <= episodes_in_each_iter:

                state, info = env.reset(seed=seed)
                current_ep_reward = 0
                print_avg_reward = 0
                timestep_per_episode_in_this_iter = []
                
                # Break if reaches maximum training timestep
                if global_timestep >= max_train_timestep:
                    print('Max training timestep reached. Stop training.')
                    break

                for t in range(1, max_ep_len+1):

                    # select action with policy
                    state['image'] = np.swapaxes(state['image'],0,2)
                    state['image'] = np.expand_dims(state['image'], axis=0)

                    transfer_start = time.time()
                    action = ppo_agent.choose_action(state['image'])    
                    transfer_end = time.time()
                    time_spent_on_transer += transfer_end - transfer_start

                    state, reward, terminated, truncated, info = env.step(action)

                    # saving reward and is_terminals
                    ppo_agent.buffer.rewards.append(reward)
                    if terminated or truncated:
                        ppo_agent.buffer.is_terminals.append(True)
                        timestep_per_episode_in_this_iter.append(t)
                    else:
                        ppo_agent.buffer.is_terminals.append(False)

                    current_ep_reward += reward
                    global_timestep += 1
                    timesteps_in_current_iter +=1

                    # update PPO agent
                    if timesteps_in_current_iter % update_timestep == 0:
                        update_start = time.time()
                        ppo_agent.update()
                        update_finish = time.time()
                        time_spent_on_updating += update_finish-update_start

                    # if continuous action space; then decay action std of ouput action distribution
                    if has_continuous_action_space and timesteps_in_current_iter % action_std_decay_freq == 0:
                        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                    # printing average reward
                    if timesteps_in_current_iter % print_freq == 0:

                        print("Environment: {} \t\t Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(current_task, environment_total_episode[current_task], environment_total_timestep[current_task]+timesteps_in_current_iter, np.mean(reward_arr[-50:])))


                        # # print average reward till last episode
                        # print_avg_reward = print_running_reward / print_running_episodes
                        # print_avg_reward = round(print_avg_reward, 2)

                        # print("Environment: {} \t\t Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(current_task, episodes_in_current_iter, environment_total_timestep[current_task]+timesteps_in_current_iter, np.mean(reward_arr[-50:])))
                        # print("Q-Values : {}".format(dfa_instance.qvalue.teacher_q_values))

                        # print_running_reward = 0
                        # print_running_episodes = 0
                    # log data
                    if timesteps_in_current_iter % log_freq == 0:
                        # TODO: Logging
                        for i in range(self.num_envs):
                            pass
                    # save model weights
                    if timesteps_in_current_iter % save_model_freq == 0:
                        print("--------------------------------------------------------------------------------------------")
                        print("saving model at : " + checkpoint_path_list[current_task])
                        ppo_agent.save(checkpoint_path_list[current_task])
                        print("model saved")
                        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                        print('Total time spent on updating: ', time_spent_on_updating)
                        print('Total time spent on transferring data: ', time_spent_on_transer)
                        print("--------------------------------------------------------------------------------------------")

                    # break; if the episode is over
                    if terminated and current_ep_reward > 0:
                        done_arr.append(1)
                        all_tasks_done[current_task].append(1)
                        all_tasks_timesteps[current_task].append(environment_total_timestep[current_task]+timesteps_in_current_iter)
                        reward_arr.append(current_ep_reward)
                        # Check if the destination node is goal state
                        if self.edges[current_task][1] == self.goal_state: 
                            final_task_performance_reward.append(current_ep_reward)
                            final_task_performance_done.append(1)
                            final_task_performance_timesteps.append(environment_total_timestep[current_task]+timesteps_in_current_iter)
                        break
                    elif terminated or truncated:
                        done_arr.append(0)
                        all_tasks_done[current_task].append(0)
                        all_tasks_timesteps[current_task].append(environment_total_timestep[current_task]+timesteps_in_current_iter)
                        reward_arr.append(current_ep_reward)
                        if self.edges[current_task][1] == self.goal_state:
                            final_task_performance_reward.append(current_ep_reward)
                            final_task_performance_done.append(0)
                            final_task_performance_timesteps.append(environment_total_timestep[current_task]+timesteps_in_current_iter)
                        break                    

                if len(reward_arr) > 50 and np.mean(reward_arr[-50:]) > 0.9:
                    print("saving converged model at : " + checkpoint_path_list[current_task])
                    ppo_agent.save(checkpoint_path_list[current_task])
                    average_timesteps_learned_tasks[current_task] = np.mean(timestep_per_episode_in_this_iter[-50:])
                    # all_tasks_timesteps[current_task].append(environment_total_timestep[current_task]+timesteps_in_current_iter)
                    is_final_task = self.learned_task(current_task)
                    is_task_leared =True
                    break
                episodes_in_current_iter += 1
                environment_total_episode[current_task] += 1
                # print_running_episodes += 1
                log_running_reward += current_ep_reward
                log_running_episodes += 1

            environment_total_timestep[current_task] += timesteps_in_current_iter
            print("done arr mean: ", np.mean(done_arr))
            if not is_task_leared:
                self.update_teacher(current_task, np.mean(done_arr))        
            # dfa_instance.update_teacher(current_task, np.mean(done_arr))
            if is_final_task:
                break
        for log_f in log_fs:
            log_f.close() 

        # env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print('Total time spent on updating: ', time_spent_on_updating)
        print("============================================================================================")

        experiment_file_name_sunk_timesteps = 'randomseed_' + str(seed) + '_sunk_timesteps'
        path_to_save_sunk_timesteps = log_dir + os.sep + experiment_file_name_sunk_timesteps + '.npz'

        experiment_file_name_sunk_episodes = 'randomseed_' + str(seed) + '_sunk_episodes'
        path_to_save_sunk_episodes = log_dir + os.sep + experiment_file_name_sunk_episodes + '.npz'

        experiment_file_name_final_reward = 'randomseed_' + str(seed) + '_final_reward'
        path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'

        experiment_file_name_final_dones = 'randomseed_' + str(seed) + '_final_dones'
        path_to_save_final_dones = log_dir + os.sep + experiment_file_name_final_dones + '.npz'

        experiment_file_name_final_timesteps = 'randomseed_' + str(seed) + '_final_timesteps'
        path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'

        np.savez_compressed(path_to_save_sunk_timesteps, sunk_timesteps = environment_total_timestep)
        np.savez_compressed(path_to_save_sunk_episodes, sunk_episodes = environment_total_episode)    
        np.savez_compressed(path_to_save_final_reward, final_reward = final_task_performance_reward)
        np.savez_compressed(path_to_save_final_dones, final_dones = final_task_performance_done)
        np.savez_compressed(path_to_save_final_timesteps, final_timesteps = final_task_performance_timesteps)

        interacted_tasks = [i for i in range(self.num_envs) if len(all_tasks_timesteps[i]) > 0] 
        experiment_file_name_all_task_timesteps = 'seed_' + str(seed) + '_all_task_timesteps' 
        path_to_save_all_task_timesteps = log_dir + os.sep + experiment_file_name_all_task_timesteps + '.pkl'
        # np.savez_compressed(path_to_save_all_task_timesteps, interacted_tasks=interacted_tasks, all_tasks_timesteps=all_tasks_timesteps, seed=seed) 
        with open(path_to_save_all_task_timesteps, 'wb') as f:
            pkl.dump((interacted_tasks, all_tasks_timesteps), f)

        experiment_file_name_all_task_dones = 'seed_' + str(seed) + '_all_task_dones' 
        path_to_save_all_task_dones = log_dir + os.sep + experiment_file_name_all_task_dones + '.pkl' 
        # np.savez_compressed(path_to_save_all_task_dones, interacted_tasks=interacted_tasks, all_tasks_dones=all_tasks_done, seed=seed)
        with open(path_to_save_all_task_dones, 'wb') as f:
            pkl.dump((interacted_tasks, all_tasks_done), f)


        print("Sunk timesteps: ", environment_total_timestep)
        print("Sunk episodes: ", environment_total_episode)
        print("final episodes: ", len(final_task_performance_timesteps))



class QValue:
    def __init__(self, num_envs, active_tasks, teacher_learning_rate = 0.1, exploration = 0.3):
        self.num_envs = num_envs
        self.active_tasks = active_tasks
        self.exploration = exploration
        self.teacher_q_values = []
        for i in range(num_envs):
            self.teacher_q_values.append(-np.inf)
        for i in active_tasks:
            self.teacher_q_values[i] = 0
        self.teacher_learning_rate = teacher_learning_rate

    def update_teacher_q_table(self, env_num, teacher_reward):
        self.teacher_q_values[env_num] = self.teacher_learning_rate*teacher_reward + (1-self.teacher_learning_rate)*self.teacher_q_values[env_num]
    
    def choose_task(self, active_tasks):
        if np.random.uniform() < self.exploration:
            task_number = np.random.choice(active_tasks)
        else:
            maxIndices = [i for i in range(len(self.teacher_q_values)) if self.teacher_q_values[i] == np.asarray(self.teacher_q_values).max()]
            task_number = np.random.choice(maxIndices)
        if task_number not in active_tasks:
            print("task number {} not in active tasks {}".format(task_number,active_tasks))
            print("q values {}".format(self.teacher_q_values))
        return task_number
    
class UCB:
    def __init__(self, num_envs, active_tasks, ucb_confidence_rate = 1.4, exploration =0.3):
        self.num_envs = num_envs
        self.active_tasks = active_tasks
        self.exploration = exploration
        self.teacher_q_values = []
        for i in range(num_envs):
            self.teacher_q_values.append(-np.inf)
        for i in active_tasks:
            self.teacher_q_values[i] = 0
        self.ucb_confidence_rate = ucb_confidence_rate
        self.total_times_arms_pulled = 0
        self.each_arm_count = [0 for i in range(num_envs)]
    
    def update_teacher_q_table(self, env_num, teacher_reward):
        self.teacher_q_values[env_num] = self.ucb_confidence_rate*teacher_reward + (1-self.ucb_confidence_rate)*self.teacher_q_values[env_num]
    
    def choose_task(self, active_tasks):
        self.total_times_arms_pulled += 1 
        bonus = [0 for i in range(self.num_envs)]
        ucb_values = copy.deepcopy(self.teacher_q_values)
        for i in range(self.num_envs):
            bonus[i] += self.ucb_confidence_rate*np.sqrt(np.log(self.total_times_arms_pulled)/self.each_arm_count[i]+1)
            ucb_values[i] += bonus[i]        
        task_number = np.argmax(ucb_values)
        if task_number not in active_tasks:
            print("task number {} not in active tasks {}".format(task_number,active_tasks))
            print("q values {}".format(self.teacher_q_values))
        self.each_arm_count[task_number] += 1
        return task_number    


if __name__ == '__main__':

    dfa_test = AGTS()
    val = 0
    while True:
        task = dfa_test.choose_task()
        print("task chosen: ", task)
        reward = np.random.randint(0,5)
        dfa_test.update_teacher(task,reward)
        if reward == 4:
            val = dfa_test.learned_task(task)
        print("Active tasks: {}, Learned tasks: {}, Task : {}, reward = {} ".format(dfa_test.active_tasks, dfa_test.learned_tasks, task,reward))
        if val == 1:
            break
    print("here")
