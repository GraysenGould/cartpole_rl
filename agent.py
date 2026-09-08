import torch
from q_network import QNeuralNetwork
from replay_buffer import ReplayBuffer
import random


device = "cpu"

class Agent ():

    def __init__ (self):
        self.model = QNeuralNetwork().to(device)
        self.target_model = QNeuralNetwork().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.replay_buffer = ReplayBuffer(10000)
        self.sample_size = 64
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = 5e-4
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        self.gamma = 0.99 # discount factor
        self.tau = 1e-2 # soft replacement rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #number of steps at which to learn
        self.t_step = 0
        self.learn_interval = 4
        # Represents C, the iteration interval to update the target network
        self.refresh_interval = 30
        self.learning_iteration = 0


    def step (self, state, action, new_state, reward, done):
        self.replay_buffer.add_experience(state, action, new_state, reward, done)

        if self.t_step % self.learn_interval == 0 and (self.replay_buffer.buffer_full or self.replay_buffer.index > self.sample_size):
            self.learn()

        self.t_step += 1

    def act (self, observation, learning: bool = True):
        action = self.model.forward(torch.from_numpy(observation))

        # choose a random value with a probability epsilon
        if random.random() < self.epsilon and learning:
            return random.choice([0, 1])

        return torch.argmax(action).item()

    def decay_epsilon (self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


    def learn (self):
        # Get samples, 
        # calculate error
        # do backprop 

        states, actions, next_states, rewards, dones = self.replay_buffer.sample_experience(self.sample_size)

        # for DDQN, use targe
        Q_next_argmax = self.model(next_states).detach().argmax(1, keepdim=True)
        Q_targets_next = self.target_model(next_states).detach().gather(1, Q_next_argmax) 

        #print(f"x: {Q_target_argmax}, type: {type(Q_target_argmax)}")
        #print(f"x: {Q_targets_next}, type: {type(Q_targets_next)}")

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.model(states).gather(1, actions.long())

        loss = self.criterion(Q_expected, Q_targets)
        # need to somehow grab all samples at a time, find the target and actual value, calculate loss, 
        self.optimizer.zero_grad() #clear old gradients
        loss.backward()
        self.optimizer.step()


        self.soft_update(self.model, self.target_model)

        # if self.learning_iteration % self.refresh_interval == 0:
        #     self.hard_update()
        self.learning_iteration += 1 


    # update target nework weights with main network
    # Run every C cycles
    def hard_update (self):
        self.target_model.load_state_dict(self.model.state_dict())


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)