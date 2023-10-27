import sys
from typing import Any
import numpy as np 
import gymnasium as gym 
import matplotlib.pyplot as plt

from enum import IntEnum
from abc import abstractmethod 
from gymnasium import spaces 

from firemangrid.utils.window import Window
from firemangrid.core.world_object import * 
from firemangrid.core.constants import * 
from firemangrid.core.grid import * 


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    pickup = 3
    spray = 4 
    toggle = 5 
    save = 6
    # move = 7
    hold = 8 


class FiremanEnv(gym.Env):
    metadata = { 
        'render.modes': ['human', 'rgb_array', 'cli'],
        'render_fps': 10 
    }

    def __init__(self, 
                 grid_size=10,
                 max_steps=100,
                 render_mode='rgb_array', 
                 ):  
        # Action space settings 
        self.actions = Actions 
        self.action_space = gym.spaces.Discrete(len(self.actions)) 

        INVENTORY_SIZE = len(OBJECT_TO_IDX) # The number of different objects in the world! 

        # Observation space settings
        image_obs_space = spaces.Box(low=0, high=255, 
                                     shape=(grid_size, grid_size, 3), # We assume the agent sees the whole grid
                                     dtype=np.uint8)
        
        self.observation_space = spaces.Dict(
            {
                'image': image_obs_space,
                'direction': spaces.Discrete(len(DIR_TO_VEC)), 
                'inventory': spaces.MultiDiscrete([INVENTORY_SIZE]),
            }
        )

        self.window = Window 
        self.grid_size = grid_size

        # Agent settings 
        self.agent_pos = None
        self.agent_dir = None 

        # Grid settings 
        self.grid = Grid(width=grid_size, height=grid_size) 
        self.carrying = None  

        # Environment settings 
        self.max_steps = max_steps
        self.render_mode = render_mode

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def observation(self):
        img = self.grid.encode() 
        obs = {
            'image': img,
            'direction': self.agent_dir,
            'inventory': np.zeros(len(OBJECT_TO_IDX), dtype=np.uint8) # TODO: Add inventory
        } 
        obs['image'][self.agent_pos[0], self.agent_pos[1], :] = np.array([OBJECT_TO_IDX['agent'], COLOR_TO_IDX['green'], self.agent_dir])
        return obs 
    
    def _reward(self):
        return 1.0 - (self.step_count / self.max_steps)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.agent_pos = (-1, -1)
        self.agent_dir = -1 
        self.step_count = 0 
        # print('environment reset')

        self._gen_grid(self.grid_size, self.grid_size)

        # Check some conditions
        assert self.agent_pos[0] >= 0 and self.agent_pos[0] < self.grid.width
        assert self.agent_pos[1] >= 0 and self.agent_pos[1] < self.grid.height 
        assert self.agent_dir >= 0 and self.agent_dir < len(DIR_TO_VEC) 

        self.carrying = None 

        # TODO: Add starting states based on the graph environment 

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
                pass 
        
        elif action == self.actions.pickup:
            if fwd_cell is not None and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell 
                    self.grid.set(*fwd_pos, None) 
                else:
                    pass # Currently we allow only pickup one thing at one time! 
        
        elif action == self.actions.spray:
            # The agent is allowed to waste spray 
            if self.carrying is None:
                pass # There is nothihg to spray
            elif self.carrying.can_spray():
                # self.carrying = None
                if fwd_cell is not None and fwd_cell.can_be_sprayed():
                    # TODO: add more conditions based on the graph environment  
                    self.grid.set(*fwd_pos, None)
                    self.carrying = None

        elif action == self.actions.toggle:
            if fwd_cell is None or not fwd_cell.can_toggle(): # Invalid action
                pass 
            else: 
                fwd_cell.toggle(self, fwd_pos) 
                # TODO: add more conditions based on the graph environment 
        
        elif action == self.actions.save: 
            if fwd_cell is None or not fwd_cell.can_save():
                pass # Nothing to save
            else: 
                if self.grid.is_safe():
                    terminated = True 
                    reward = self._reward()

        # elif action == self.actions.move:
        #     if fwd_cell is None or not fwd_cell.can_move():
        #         pass 
        #     else:
        #         self.grid.set(*fwd_pos, None)
        
        # elif action == self.actions.hold:
        #     # Do nothing!
        #     pass 

        else:
            raise ValueError(f"Unknown action: {action}")
        
        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == 'human':
            self.render() 
        
        obs = self.observation()
        return obs, reward, terminated, truncated, {} 
    
    def render(self):
        if self.render_mode == 'human':
            if self.window is None:
                self.window = Window('FiremanGrid') 
                self.window.show(block=False) 
            plt.imshow(self.grid.render(
                self.agent_pos,
                self.agent_dir
            ))
            plt.show()
            # self.window.show_img(self.grid.render(
            #     self.agent_pos,
            #     self.agent_dir
            # ))  
        elif self.render_mode == 'rgb_array':
            # TODO: rgb rendering 
            pass

        elif self.render_mode == 'cli':
            img = self.grid.render(self.agent_pos, self.agent_dir, render_mode='cli') 
            print(img.shape)
            print(np.transpose(img, (1, 0)))

    def close(self):
        if self.window:
            self.window.close(self.window)
        
