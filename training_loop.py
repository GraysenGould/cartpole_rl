import gymnasium as gym
from replay_buffer import ReplayBuffer, Experience
from agent import Agent


class TrainAgent ():

    def __init__ (self):
        # ------- Hyperparameters -------
        #maximum number of steps
        self.max_t = 500
        self.n_episodes = 5000
        self.discount = 0.99
        self.learning_rate = 0.01
        
        self.agent = Agent()

    def training_loop (self):
        env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
        #env = gym.make("CartPole-v1", max_episode_steps=500)

        scores = []
        for ep in range(self.n_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            while not done:

                action = self.agent.act(state)
                #print("action: ", action.item())
                new_state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                self.agent.step(state, action, new_state, reward, done)
                
                #print("output of step: ", new_state, reward, terminated, truncated, info)

                state = new_state
                total_reward += reward
            
            print("score: ", total_reward)
            scores.append(total_reward)

        env.close()

if __name__ == "__main__":
    train = TrainAgent()
    train.training_loop()