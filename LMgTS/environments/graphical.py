import numpy as np
import copy
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper


class GraphEnv:
    def __init__(self, num_agents, total_paths, goal_ordering):
        self.num_agents = num_agents
        self.total_paths = total_paths
        self.goal_ordering = goal_ordering
        self.learned_tasks = [[] for _ in range(self.total_paths)]
        self.alpha = 0.9
        q_value_per_path = [-np.inf for i in range(self.num_agents)]
        self.q_values = [copy.deepcopy(q_value_per_path) for _ in range(self.total_paths)]

        self.active_tasks = [-np.inf for path in range(self.total_paths)]
        self.q_values[0][2] = 0

        for path in range(total_paths):
            self.q_values[path][self.goal_ordering[path][0]] = 0
            self.active_tasks[path] = self.goal_ordering[path][0]
        # print("q values: ", self.q_values)
        self.print_q_values()
        print("active tasks: ", self.active_tasks)

    def print_q_values(self):
        print('==============Printing Q Values==============')
        for i, q_value in enumerate(self.q_values):
            # print('================Printing Q Values==================')
            print('Path {}, current Graph Environment Q values: {}'.format(self.goal_ordering[i], q_value))
        print('=============================================\n')

    def choose_task(self):
        """
        Choose a task via epsilon-greedy
        """
        print('Choosing task...')
        self.print_q_values()
        sampled_q_values = []
        for path in range(self.total_paths):
            sampled_q_values.append(self.q_values[path][self.active_tasks[path]])

        if np.random.uniform() < 0.2:
            path = np.random.randint(0, self.total_paths)
            # print('Random task is chosen')
        else:
            path = np.argmax(sampled_q_values)
            # print('Greedy task is chosen')
        task = self.active_tasks[path]

        return path, task

    def learned_task(self, path, task_num):
        learned_task_index = np.where(np.asarray(self.goal_ordering[path]) == task_num)[0][0]
        self.learned_tasks[path].append(task_num)
        if task_num == self.goal_ordering[path][-1]:
            print("Done final task! Check")
        else:
            task_to_add = self.goal_ordering[path][learned_task_index + 1]
            self.active_tasks.insert(path, task_to_add)
            self.active_tasks.pop(path + 1)
        print("Learned tasks: ", self.learned_tasks)
        print("Q values: ", self.q_values)

    def update_teacher(self, path, task_num, reward):
        # task_index = np.where(np.asarray(self.goal_ordering[path]) == task_num)[0][0]
        # print("task index:", task_index)
        # print("reward: ", reward)
        self.q_values[path][task_num] = self.alpha * reward + (1 - self.alpha) * self.q_values[path][task_num]


# class MinigridGraphEnv(GraphEnv):
#     """
#     Graph environment for Minigrid domain. Each node is a high-level state and each edge is a sub-task.
#
#     High-Level States:
#         0: At(Room1)
#         1: Picked(EasyKey)
#         2: Picked(HardKey)
#         3: Opened(Door)
#         4: At(Goal)
#
#     Sub-tasks:
#         0: 0-1: NineRoomsEasyKey, Get easy key starting from Room 1.
#         1: 0-2: NineRoomsHardKey, Get hard key starting from Room 1
#         None: 0-3: Invalid
#         None: 0-4: Invalid
#         2: 1-0: NineRoomsEasyKey2Room, Return Room1 starting from Easy Key room
#         3: 1-2: NineRoomsEasyKey2HardKey, Get hard key starting from Easy Key room
#         4: 1-3: NineRoomsEasyDoor, Open Door starting from Easy Key room
#         None: 1-4: Invalid
#         5: 2-0: NineRoomsHardKey2Room, Return Room1 starting from Hard Key room
#         6: 2-1: NineRoomsHardKey2EasyKey, Get easy key starting from Hard Key room
#         7: 2-3: NineRoomsHardDoor, Open Door starting from Hard Key room
#         None: 2-4: Invalid
#         8: 3-0: NineRoomsDoor2Room, Return Room1 starting from Door
#         9: 3-1: NineRoomsDoor2Easy, Return Easy Key room starting from Door
#         10: 3-2: NineRoomsDoor2Hard, Return Hard Key room starting from Door
#         11: 3-4: NineRoomsDoorGoal, Reach Goal
#     """
#
#     def __init__(self, num_agents, total_paths, goal_ordering, allow_cyclic=True):
#         super().__init__(num_agents, total_paths, goal_ordering)
#         self.envs = [
#             'MiniGrid-NineRoomsEasyKey-v0', 'MiniGrid-NineRoomsHardKey-v0',
#             'MiniGrid-NineRoomsEasyKey2Room-v0', 'MiniGrid-NineRoomsEasyKey2HardKey-v0',
#             'MiniGrid-NineRoomsEasyDoor-v0', 'MiniGrid-NineRoomsHardKey2Room-v0',
#             'MiniGrid-NineRoomsHardKey2EasyKey-v0', 'MiniGrid-NineRoomsHardDoor-v0',
#             'MiniGrid-NineRoomsDoor2Room-v0', 'MiniGrid-NineRoomsDoor2Easy-v0',
#             'MiniGrid-NineRoomsDoor2Hard-v0', 'MiniGrid-NineRoomsDoorGoal-v0'
#         ]
#
#         self.state_map = {
#             'At(Room1)': 0,
#             'Picked(EasyKey)': 1,
#             'Picked(HardKey)': 2,
#             'Opened(Door)': 3,
#             'At(Goal)': 4
#         }
#
#         self.edges = {
#             (0, 1): 0,
#             (0, 2): 1,
#             (0, 3): 'Cannot open the door without an key',
#             (0, 4): 'Cannot reach the goal without opening the door.',
#             (1, 0): 'No need to return to Room1.',
#             (1, 2): 'No need to get another key.',
#             (1, 3): 4,
#             (1, 4): 'Cannot reach the goal without opening the door.',
#             (2, 0): 'No need to return to Room1.',
#             (2, 1): 'No need to get another key.',
#             (2, 3): 7,
#             (2, 4): 'Cannot reach the goal without opening the door.',
#             (3, 0): 'No need to return to Room1.',
#             (3, 1): 'No need to get another key.',
#             (3, 2): 'No need to get another key.',
#             (3, 4): 11
#         }
#
#         if allow_cyclic:
#             new_edges = {
#                 (1, 0): 2,
#                 (1, 2): 3,
#                 (2, 0): 5,
#                 (2, 1): 6,
#                 (3, 0): 8,
#                 (3, 1): 9,
#                 (3, 2): 10
#             }
#             self.edges.update(new_edges)
#
#         self.current_state = [0 for _ in range(self.num_agents)]
#
#     def parse_path(self, path):
#         """
#         Parse a single path given by LLM. Returns the corresponding sequence of tasks if valid,
#         otherwise raises an error that could be passed directly to LLM.
#
#         :param path:
#         :return:
#         """
#         state_sequence = []
#         edge_sequence = []
#         for p in path:
#             if p not in self.state_map.keys():
#                 raise "I cannot recognize {} in your last response.".format(p)
#             state_sequence.append(self.state_map[p])
#
#         for i, s in enumerate(state_sequence, start=1):
#             if not isinstance(self.edges[state_sequence[i-1]][s], int):
#                 raise "In your last response, {} to {} is invalid. {}".format(path[state_sequence[i-1]],
#                                                                               path[i], self.edges[(i-1, i)])
#             edge_sequence.append(self.edges[state_sequence[i-1]][s])
#         return edge_sequence

