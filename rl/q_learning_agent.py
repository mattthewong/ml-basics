import numpy as np
import gym
import time as time
import matplotlib.pyplot as plt
import datetime

class QLearningAgent(object):
    def __init__(self,  problem_space='Taxi-v3', episodes=30000, gamma=0.95, alpha=0.2, epsilon=0):
        self.Q = None
        self.alpha = alpha
        self.gamma = gamma
        self.env = gym.make(problem_space).env
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.epsilon_decay = 0.00007
        self.epsilon_update_thres = 0.07
        self.episodes = episodes
        self.rewards = []
        self.iterations = []
        self.times = []
        self.problem_space = problem_space

    def solve(self):
        """Create the Q table"""
        print(f"Running q_learning for problem {self.problem_space} with epsilon {round(self.start_epsilon, 2)}...")
        start = time.time()
        optimal = [0] * self.env.observation_space.n
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        for i in range(self.episodes):
            t_reward = 0
            state = self.env.reset()
            complete = False
            j = 0
            max_steps = 1000000
            for j in range(max_steps):
                if complete:
                    break
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(self.env.action_space.n)
                else:
                    action = np.argmax(self.Q[state, :])

                next_s, r, complete, _ = self.env.step(action)
                t_reward += r
                next_best_a = np.argmax(self.Q[next_s, :])
                update = self.alpha * (r + self.gamma * self.Q[next_s, next_best_a] - self.Q[state, action])
                self.Q[state, action] += update
                state = next_s
                j += 1
                if complete:
                    break
            self.rewards.append(t_reward)
            self.iterations.append(j)
            if self.epsilon > self.epsilon_update_thres:
                self.epsilon = self.epsilon * (1 - self.epsilon_decay)

        for k in range(self.env.observation_space.n):
            optimal[k] = np.argmax(self.Q[k, :])
        # print(optimal)
        self.env.close()
        end = time.time()
        exeuction_time = end - start

        # Plot results
        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(self.episodes / 50)
        chunks = list(chunk_list(self.rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        plt.plot(range(0, len(self.rewards), size), averages)
        plt.title(f'Q-Learning Average Reward vs. Iterations (epsilon={round(self.start_epsilon, 2)})')
        plt.xlabel('Iterations')
        plt.grid()
        plt.ylabel('Average Reward')
        plt.savefig(f"q_learning_{self.problem_space}_average_reward_epsilon_{round(self.start_epsilon, 2)}_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        self.env.close()
        return exeuction_time

    def Q_table(self, state, action):
        return self.Q[state][action]