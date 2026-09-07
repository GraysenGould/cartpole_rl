import torch
from q_network import QNeuralNetwork
from replay_buffer import ReplayBuffer


device = "cpu"

class Agent ():

    def __init__ (self):
        self.model = QNeuralNetwork().to(device)
        self.target_model = QNeuralNetwork().to(device)
        self.replay_buffer = ReplayBuffer(1000)
        self.sample_size = 500
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = 1e-3
        self.epsilon_start = 0.10
        self.epsilon_end = 0.0001
        self.epsilon_decay = 0.995
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.iteration = 0
        # Represents C, the iteration interval to update the target network
        self.refresh_interval = 1000

    def step (self, state, action, new_state, reward, done):
        self.replay_buffer.add_experience(state, action, new_state, reward, done)

    def act (self, observation):
        action = self.model.forward(torch.from_numpy(observation))

        return torch.argmax(action)


    def learn (self, gamma):
        # Get samples, 
        # calculate error
        # do backprop 

        states, actions, next_states, rewards, dones = self.replay_buffer.sample_experience(self.sample_size)

        Q_targets_next = self.target_model(next_states).detatch().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = self.criterion(Q_expected, Q_targets)
        # need to somehow grab all samples at a time, find the target and actual value, calculate loss, 
        self.model()
        self.optimizer.zero_grad() #clear old gradients
        loss = self.criterion()
        loss.backward()
        self.optimizer.step()
        if iteration % self.refresh_interval == 0:
            self.update()
        iteration += 1


    # update target nework weights with main network
    # Run every C cycles
    def update (self):
        self.target_model.load_state_dict(self.model.state_dict())