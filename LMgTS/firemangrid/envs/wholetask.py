import numpy as np 
import gymnasium as gym 
import sys 

from firemangrid.core.grid import *
from firemangrid.core.constants import * 
from firemangrid.core.world_object import * 
from firemangrid.fireman_env import FiremanEnv


class FiremanWholeEnv(FiremanEnv):
    def __init__(self, grid_size=13, max_steps=500, render_mode='rgb_array', task='start2key'):
        super().__init__(grid_size, max_steps, render_mode)  
        print('task is: ', task)
        self.task = task
        self.tasks = ['start2key', 'start2door', 'start2fireextinguisher', 'start2fire', 'start2debris', 'start2survivor',
                      'key2start', 'key2door', 'key2fireextinguisher', 'key2fire', 'key2debris', 'key2survivor',
                      'door2start', 'door2key', 'door2fireextinguisher', 'door2fire', 'door2debris', 'door2survivor',
                      'fireextinguisher2start', 'fireextinguisher2key', 'fireextinguisher2door', 'fireextinguisher2fire', 'fireextinguisher2debris', 'fireextinguisher2survivor',
                      'fire2start', 'fire2key', 'fire2door', 'fire2fireextinguisher', 'fire2debris', 'fire2survivor',
                      'debris2start', 'debris2key', 'debris2door', 'debris2fireextinguisher', 'debris2fire', 'debris2survivor', 
                      'survivor2start', 'survivor2key', 'survivor2door', 'survivor2fireextinguisher', 'survivor2fire', 'survivor2debris']

    def _gen_grid(self, width, height):
        self.grid = Grid(width=width, height=height)

        # Surrounding Walls
        for i in range(0, width):
            self.grid.set(i, 0, Wall()) 
            self.grid.set(i, width-1, Wall()) 
        
        for j in range(0, height):
            self.grid.set(0, j, Wall()) 
            self.grid.set(height-1, j, Wall()) 

        # Survivor's room
        for i in range(1, 4):
            self.grid.set(i, 4, Wall())

        for j in range(1, 5):
            self.grid.set(4, j ,Wall())     

        # Obstacle walls
        for i in range(1, 12):
            self.grid.set(i, 8, Wall())

        for j in range(8, 12):
            self.grid.set(j, 8, Wall())

        for j in range(10, 12):
            self.grid.set(8, j, Wall()) 

        for i in range(5, 11):
            self.grid.set(4, i, Wall())   

        for i in range(6, 11):
            self.grid.set(i, 4, Wall())

        for i in range(6, 11):
            self.grid.set(i, 2, Wall())

        self.grid.set(6, 9, Wall()) 
        self.grid.set(6, 10, Wall())
        self.grid.set(2, 10, Wall())
        self.grid.set(3, 10, Wall())
        self.grid.set(8, 2, Wall())
        self.grid.set(8, 3, Wall())

        # self.grid.set(, 6)
        # Lava
        self.grid.set(1, 7, Wall())
        self.grid.set(3, 7, Wall())
        self.grid.set(1, 5, Wall())
        self.grid.set(2, 5, Wall())
        self.grid.set(3, 5, Wall())
        self.grid.set(2, 8, None) 
        for i in range(6, 11):
            self.grid.set(i, 6, Wall())
        # for j in range(2, 8):
        #     self.grid.set(8, j, Lava())
        # for i in range(1, 4):
        #     self.grid.set(i, 8, Lava())
        # self.grid.set(5, 5, Lava())
        # self.grid.set(6, 8, Lava()) 

        fire_pos = (4, 2)
        fe_pos = (2, 5)
        key_pos = (3, 6)
        debris_pos = (5, 8) 
        door_pos = (5, 8)

        # fe_pos_x = np.random.choice([8, 9, 10])
        # fe_pos_y = np.random.choice([5, 6, 7])

        fe_candidates = []
        for i in range(8, 11):
            for j in range(1, 7):
                if self.grid.get(i, j) is None:
                    fe_candidates.append((i, j))
        fe_pos = fe_candidates[np.random.choice(len(fe_candidates))]

        self.grid.set(*door_pos, Door('yellow'))   
        self.task_id = self.tasks.index(self.task)

        if self.task_id < self.tasks.index('key2start') or self.task == 'door2key' or self.task == 'fireextinguisher2key' or self.task == 'fire2key' or self.task == 'debris2key' or self.task == 'survivor2key':
            self.grid.set(*key_pos, Key('yellow'))
        if self.task_id < self.tasks.index('door2start'):
            self.grid.get(*door_pos).is_locked = True
            self.grid.get(*door_pos).is_open = False 
        else:
            self.grid.get(*door_pos).is_locked = False
            self.grid.get(*door_pos).is_open = True
        if self.task_id < self.tasks.index('fireextinguisher2start') or self.task == 'fire2fireextinguisher' or self.task == 'debris2fireextinguisher' or self.task == 'survivor2extinguisher':
            self.grid.set(*fe_pos, FireExtinguisher())
        if self.task_id < self.tasks.index('fire2start') or self.task == 'debris2fire' or self.task == 'survivor2fire':
            self.grid.set(*fire_pos, Fire()) 
        # if self.task_id < self.tasks.index('debris2start'):
        #     self.grid.set(*debris_pos, Debris())

        # if not (self.task == 'fire2start' or self.task == 'fire2key' or self.task == 'fire2door' or self.task == 'fire2fireextinguisher' or self.task == 'fire2debris' or self.task == 'fire2survivor'):
        #     self.grid.set(*fire_pos, Fire())

        # if not (self.task == 'fireextinguisher2start' or self.task == 'fireextinguisher2key' or self.task == 'fireextinguisher2door' or self.task == 'fireextinguisher2fire' or self.task == 'fireextinguisher2debris' or self.task == 'fireextinguisher2survivor'):
        #     self.grid.set(*fe_pos, FireExtinguisher())  

        # if not (self.task == 'key2start' or self.task == 'key2door' or self.task == 'key2fireextinguisher' or self.task == 'key2fire' or self.task == 'key2debris' or self.task == 'key2survivor'):
        #     self.grid.set(*key_pos, Key('yellow')) 

        # if not (self.task == 'debris2start' or self.task == 'debris2key' or self.task == 'debris2door' or self.task == 'debris2fireextinguisher' or self.task == 'debris2fire' or self.task == 'debris2survivor'):
        #     self.grid.set(*debris_pos, Debris())

        survivor_x = np.random.choice([1, 2, 3])
        survivor_y = np.random.choice([1, 2, 3]) 
        self.grid.set(survivor_x, survivor_y, Survivor())

        for i in range(9, 12):
            for j in range(9, 12):
                self.grid.set(i, j, Start())

        # Initialize the agent 
        if self.task == 'start2key' or self.task == 'start2door' or self.task == 'start2fireextinguisher' or self.task == 'start2fire' or self.task == 'start2debris' or self.task == 'start2survivor':
            init_canditates = [] 
            for i in range(1, width-1): 
                for j in range(1, height-1): 
                    if isinstance(self.grid.get(i, j), Start): 
                        init_canditates.append((i, j)) 
        
            self.agent_pos = init_canditates[np.random.choice(len(init_canditates))] 

        if self.task == 'key2start' or self.task == 'key2door' or self.task == 'key2fireextinguisher' or self.task == 'key2fire' or self.task == 'key2debris' or self.task == 'key2survivor':
            self.agent_pos = key_pos
        
        if self.task == 'door2start' or self.task == 'door2key' or self.task == 'door2fireextinguisher' or self.task == 'door2fire' or self.task == 'door2debris' or self.task == 'door2survivor':
            self.agent_pos = door_pos
        
        if self.task == 'fireextinguisher2start' or self.task == 'fireextinguisher2key' or self.task == 'fireextinguisher2door' or self.task == 'fireextinguisher2fire' or self.task == 'fireextinguisher2debris' or self.task == 'fireextinguisher2survivor':
            self.agent_pos = fe_pos
        
        if self.task == 'fire2start' or self.task == 'fire2key' or self.task == 'fire2door' or self.task == 'fire2fireextinguisher' or self.task == 'fire2debris' or self.task == 'fire2survivor':
            self.agent_pos = fire_pos
        
        if self.task == 'debris2start' or self.task == 'debris2key' or self.task == 'debris2door' or self.task == 'debris2fireextinguisher' or self.task == 'debris2fire' or self.task == 'debris2survivor':
            self.agent_pos = debris_pos
        
        # self.agent_pos = init_canditates[np.random.choice(len(init_canditates))] 
        # print(self.agent_pos)
        
        self.agent_dir = np.random.choice(len(DIR_TO_VEC))
        # print(self.agent_dir)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.agent_pos = (-1, -1)
        self.agent_dir = -1 
        self.step_count = 0

        self._gen_grid(self.grid_size, self.grid_size)

        # Check some conditions
        assert self.agent_pos[0] >= 0 and self.agent_pos[0] < self.grid.width
        assert self.agent_pos[1] >= 0 and self.agent_pos[1] < self.grid.height 
        assert self.agent_dir >= 0 and self.agent_dir < len(DIR_TO_VEC) 

        self.carrying = None 

        # TODO: Add starting states based on the graph environment 

        if self.task == 'key2start' or self.task == 'key2door' or self.task == 'key2fireextinguisher' or self.task == 'key2fire' or self.task == 'key2debris' or self.task == 'key2survivor':
            self.carrying = Key('yellow') 
        
        if self.task == 'fireextinguisher2start' or self.task == 'fireextinguisher2key' or self.task == 'fireextinguisher2door' or self.task == 'fireextinguisher2fire' or self.task == 'fireextinguisher2debris' or self.task == 'fireextinguisher2survivor':
            self.carrying = FireExtinguisher()

        if self.render_mode == 'human':
            self.render() 

        obs = self.observation()
        return obs, {} 
    
    def step(self, action):
        assert action >= 0 and action < self.action_space.n 
        self.step_count += 1 
        reward, terminated, truncated = 0, False, False

        fwd_pos = DIR_TO_VEC[self.agent_dir] + self.agent_pos

        fwd_cell = self.grid.get(*fwd_pos) 
        cur_cell = self.grid.get(*self.agent_pos) 

        task_id = self.tasks.index(self.task)
        if task_id >= self.tasks.index('survivor2start'):
            terminated = True

        if action == self.actions.left: 
            self.agent_dir -= 1 
            if self.agent_dir < 0:
                self.agent_dir += len(DIR_TO_VEC) 
        
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % len(DIR_TO_VEC) 

        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos 
            elif fwd_cell is not None: 
                # TODO: add more conditions based on the graph environment 
                if fwd_cell.type == 'lava':
                    terminated = True 
        
        elif action == self.actions.pickup:
            if fwd_cell is not None and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell 
                    # print('Picked up ', fwd_cell.type)
                    self.grid.set(*fwd_pos, None) 
                else:
                    pass # Currently we allow only pickup one thing at one time! 
                if self.carrying.type == 'key' and (self.task == 'start2key' or self.task == 'door2key' or self.task == 'fireextinguisher2key' or self.task == 'fire2key' or self.task == 'debris2key'):
                    reward = self._reward()
                    terminated = True 
                if self.carrying.type == 'fireextinguisher' and (self.task == 'start2fireextinguisher' or self.task == 'key2fireextinguisher' or self.task == 'door2fireextinguisher' or self.task == 'fire2fireextinguisher' or self.task == 'debris2fireextinguisher'):
                    reward = self._reward()
                    terminated = True
        
        elif action == self.actions.spray:
            # The agent is allowed to waste spray 
            if self.carrying is None:
                pass # There is nothihg to spray
            elif self.carrying.can_spray():
                # self.carrying = None
                if fwd_cell is not None and fwd_cell.can_be_sprayed():
                    # TODO: add more conditions based on the graph environment  
                    if fwd_cell.type == 'fire' and (self.task == 'start2fire' or self.task == 'key2fire' or self.task == 'door2fire' or self.task == 'fireextinguisher2fire' or self.task == 'debris2fire'):
                        reward = self._reward()
                        terminated = True
                    self.grid.set(*fwd_pos, None)
                    self.carrying = None

        elif action == self.actions.toggle:
            if fwd_cell is None or not fwd_cell.can_toggle(): # Invalid action
                pass 
            else: 
                fwd_cell.toggle(self, fwd_pos) 
                if fwd_cell.type == 'door' and fwd_cell.is_open == True and (self.task == 'start2door' or self.task == 'key2door' or self.task == 'fireextinguisher2door' or self.task == 'fire2door' or self.task == 'debris2door'):
                    reward = self._reward()
                    terminated = True
                # TODO: add more conditions based on the graph environment 
        
        elif action == self.actions.save: 
            if fwd_cell is None or not fwd_cell.can_save():
                pass # Nothing to save
            else: 
                if self.grid.is_safe():
                    if self.task == 'start2survivor' or self.task == 'key2survivor' or self.task == 'door2survivor' or self.task == 'fireextinguisher2survivor' or self.task == 'fire2survivor' or self.task == 'debris2survivor':
                        reward = self._reward()
                        terminated = True

        elif action == self.actions.move:
            if fwd_cell is None or not fwd_cell.can_move():
                pass 
            else:
                if fwd_cell.type == 'debris' and (self.task == 'start2debris' or self.task == 'key2debris' or self.task == 'door2debris' or self.task == 'fireextinguisher2debris' or self.task == 'fire2debris'):
                    reward = self._reward()
                    terminated = True
                self.grid.set(*fwd_pos, None)
        
        elif action == self.actions.hold:
            # Do nothing!
            pass 

        else:
            raise ValueError(f"Unknown action: {action}")
        
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == 'human':
            self.render() 
        
        obs = self.observation()
        return obs, reward, terminated, truncated, {}
