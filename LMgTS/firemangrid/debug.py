# Allow user to play in the environment for debugging purpose 
import sys
import gymnasium as gym 
import pygame 

from gymnasium import Env

from firemangrid.fireman_env import FiremanEnv, Actions 
from firemangrid.envs.extinguish_fire import ExtinguishFireEnv

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "tab": Actions.pickup,
            "left shift": Actions.spray,
            "enter": Actions.save,
            'h': Actions.hold
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


class CLIControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.key_to_action = {
            "a": Actions.left,
            "d": Actions.right,
            "w": Actions.forward,
            "j": Actions.toggle,
            "k": Actions.pickup,
            "l": Actions.spray,
            "u": Actions.save,
            # 'i': Actions.move,
            'o': Actions.hold
        }

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        truncated, terminated = False, False
        while not (truncated or terminated):
            action = input('Enter action: ') 
            if not (action in self.key_to_action.keys()):
                print('Invalid action!')
                return
            action = self.key_to_action[action]
            self.step(action)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()


if __name__ == "__main__":

    env_id = 'FiremanGrid-Start2Key-v0'

    env: ExtinguishFireEnv = gym.make(
        env_id,
        render_mode="human",
    )

    # TODO: check if this can be removed
    cli_control = CLIControl(env)
    cli_control.start()


