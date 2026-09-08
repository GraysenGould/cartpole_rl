import gymnasium as gym
from replay_buffer import ReplayBuffer, Experience
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class TrainAgent ():

    def __init__ (self):
        # ------- Hyperparameters -------
        self.n_episodes = 1500
        self.solved_threshold = 500.0
        self.max_episode_steps= 600
        self.agent = Agent()
    
    def training_loop (self):
        #env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
        env = gym.make("CartPole-v1", max_episode_steps=self.max_episode_steps)

        scores = []
        scores_window = deque(maxlen=100)
        for ep in range(self.n_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.agent.step(state, action, new_state, reward, done)
                
                #print("output of step: ", new_state, reward, terminated, truncated, info)

                state = new_state
                total_reward += reward
            
            self.agent.decay_epsilon()

            print(f"Iteration: {ep} reward: {total_reward}")
            scores.append(total_reward)
            scores_window.append(total_reward)
            if np.mean(scores_window) >= self.solved_threshold:
                break

        env.close()
        return scores

    def visual_loop (self, n_examples):
        env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)

        for ep in range(n_examples):
            state, info = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.agent.act(state, False)
                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                #print("output of step: ", new_state, reward, terminated, truncated, info)

                state = new_state
                total_reward += reward

            print(f"Iteration: {ep} reward: {total_reward}")
            scores.append(total_reward)

        env.close()

if __name__ == "__main__":
    train = TrainAgent()
    scores = train.training_loop()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('DQN Training Performance on CartPole-v1')
    plt.grid(True)
    plt.savefig("results.png")
    train.visual_loop(10)
    