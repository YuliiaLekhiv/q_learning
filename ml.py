import gym
import numpy as np
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env_name='Acrobot-v1', buckets=15, learning_rate=0.1, gamma=0.95, epochs=5000):
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high

        self.discrete_obs_size = [buckets] * len(self.obs_low)
        self.bucket_size = (self.obs_high - self.obs_low) / self.discrete_obs_size
        self.q_table = np.random.uniform(high=3, low=0, size=(self.discrete_obs_size + [self.action_size]))

        self.lr = learning_rate
        self.gamma = gamma
        self.epochs = epochs
        self.eps = 1
        self.eps_start_decay = 1
        self.eps_end_decay = 3 * episodes // 4
        self.eps_decay_rate = self.eps / (self.eps_end_decay - self.eps_start_decay)

    def discretize(self, state):
        discrete_state = (state - self.obs_low) / self.bucket_size
        discrete_state = np.clip(discrete_state.astype(int), None, 14)
        return tuple(discrete_state)

    def train(self):
        rewards = []
        for i in range(self.epochs):
            state = self.discretize(self.env.reset())
            done = False

            while not done:
                if np.random.uniform(0, 1) < self.eps:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                step_result = self.env.step(action)
                new_state, reward, done, _ = step_result[:4]
                new_state = self.discretize(new_state)

                if not done:
                    self.q_table[state][action] += self.lr * (reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])
                else:
                    self.q_table[state][action] = reward

                if self.eps_start_decay <= i < self.eps_end_decay:
                    self.eps -= self.eps_decay_rate

                state = new_state

            rewards.append(reward)



            if reward == 0.0:
                print(f"Epochs: {i}/{self.epochs}     Epsilon:{self.eps}      Reward:{reward}")



        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards per Epochs')
        plt.show()

    def test(self):
        state = self.discretize(self.env.reset())
        done = False
        while not done:
            action = np.argmax(self.q_table[state])
            new_state, _, done, _ = self.env.step(action)[:4]
            state = self.discretize(new_state)

        self.env.close()

agent = QLearningAgent()
agent.train()
agent.test()
