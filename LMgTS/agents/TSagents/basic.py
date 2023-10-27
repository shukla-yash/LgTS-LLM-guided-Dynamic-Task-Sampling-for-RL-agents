from abc import abstractmethod


class TSModel:

    @abstractmethod
    def __init__(self, edges: list):
        pass

    @abstractmethod
    def choose_task(self) -> int:
        pass
    
    @abstractmethod
    def update_teacher(self, task_num, reward):
        pass

    @abstractmethod
    def train(self, envs: list, agents: list):
        pass

