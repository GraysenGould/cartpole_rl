from collections import deque
import torch
import numpy as np



class Experience():
    def __init__ (self, state, next_state = None, action = None):
        self.state = state
        self.next_state = next_state
        self.action = action
        
    def add_next_state (self, next_state):
        self.next_state = next_state

    def add_action(self, action):
        self.action = action

class ReplayBuffer ():
    def __init__(self, n_experiences = 1000):
        self.n_experiences = n_experiences

        self.states = np.zeros((n_experiences, 4), np.float32)
        self.actions = np.zeros((n_experiences, 2), np.float32)
        self.next_states = np.zeros((n_experiences, 4), np.float32)
        self.rewards = np.zeros((n_experiences, 1), np.float32)
        self.dones = np.zeros((n_experiences, 1), np.bool_)

        # Index at which to add data. Circular index. Override old values
        self.index = 0
        # becomes true when buffer is full. Else, sample from 0 to index if False
        self.buffer_full: bool = False

    # Take in an experience in the form ()
    def add_experience(self, state, action: float, next_state, reward: float, done: bool) -> None:
        action_array = [
            0 if action == 0 else 1,
            1 if action == 0 else 0
        ]

        np.insert(self.states, self.index, state)
        np.insert(self.actions, self.index, action_array)
        np.insert(self.next_states, self.index, next_state)
        np.insert(self.rewards, self.index, reward)
        np.insert(self.dones, self.index, done)

        if self.index == self.n_experiences - 1:
            self.buffer_full = True
        # modular arithmetic to keep index circular
        self.index = (self.index + 1) % self.n_experiences
        print("index: ", self.index)

    def sample_experience (self):
        idx = self.index
        if self.buffer_full:
            idx = self.n_experiences - 1


        return (np.random.choice(self.states[:idx + 1], size = 1, replace=True),
                np.random.choice(self.actions[:idx + 1], size = 1, replace=True),
                np.random.choice(self.next_states[:idx + 1], size = 1, replace=True),
                np.random.choice(self.rewards[:idx + 1], size = 1, replace=True),
                np.random.choice(self.dones[:idx + 1], size = 1, replace=True))
        
        

