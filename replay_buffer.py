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
        self.actions = np.zeros((n_experiences, 1), np.float32)
        self.next_states = np.zeros((n_experiences, 4), np.float32)
        self.rewards = np.zeros((n_experiences, 1), np.float32)
        self.dones = np.zeros((n_experiences, 1), np.int8)

        # Index at which to add data. Circular index. Override old values
        self.index = 0
        # becomes true when buffer is full. Else, sample from 0 to index if False
        self.buffer_full: bool = False

    # Take in an experience in the form ()
    def add_experience(self, state, action: float, next_state, reward: float, done: bool) -> None:
        self.states[self.index] = state
        self.actions[self.index] = action
        self.next_states[self.index] = next_state
        self.rewards[self.index] = reward
        self.dones[self.index] = done

        if self.index == self.n_experiences - 1:
            self.buffer_full = True
        # modular arithmetic to keep index circular
        self.index = (self.index + 1) % self.n_experiences
        #print("index: ", self.index)

    def sample_experience(self, samples):
        max_idx = self.index - 1
        if self.buffer_full:
            max_idx = self.n_experiences - 1

        
        sample_idx = np.random.choice(max_idx + 1, size=samples, replace=True)


        return (torch.from_numpy(self.states[sample_idx]),
                torch.from_numpy(self.actions[sample_idx]),
                torch.from_numpy(self.next_states[sample_idx]),
                torch.from_numpy(self.rewards[sample_idx]),
                torch.from_numpy(self.dones[sample_idx]))
        
        

