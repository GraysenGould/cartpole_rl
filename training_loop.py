import gymnasium as gym
from replay_buffer import ReplayBuffer, Experience


class TrainAgent ():

    def __init__ (self):
        # ------- Hyperparameters -------
        #maximum number of steps
        self.max_t = 500
        self.n_episodes = 5000
        self.discount = 0.99
        self.learning_rate = 0.01
        self.replay_buffer = ReplayBuffer(1000)

    def training_loop (self):
        env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
        #env = gym.make("CartPole-v1", max_episode_steps=500)

        total_reward = 0
        for ep in range(self.n_episodes):
            observation, info = env.reset()
            # simulate first step
            observation, reward, terminated, truncated, info = env.step(0)
            episode_over = False
            while not episode_over:

                state = observation
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.replay_buffer.add_experience(state, action, observation, reward, done)


                print("action: ", action)
                print("output of step: ", observation, reward, terminated, truncated, info)
                total_reward += reward
                episode_over = terminated or truncated


        env.close()

if __name__ == "__main__":
    train = TrainAgent()
    train.training_loop()