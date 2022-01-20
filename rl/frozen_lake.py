import numpy as np
import gym
import time as time
from utils import policy_iteration, evaluate_policy, value_iteration, extract_policy
import matplotlib.pyplot as plt
import datetime
from q_learning_agent import QLearningAgent

run_policy_iteration = True
run_value_iteration = True
run_q_learning = True


def main():
    if run_policy_iteration:
        env_name ='FrozenLake-v1'
        env = gym.make(env_name)
        env = env.unwrapped
        times = [0] * 10
        gammas = [0] * 10
        iterations = [0] * 10
        score_set = [0] * 10
        for i in range(0, 10):
            print(f"Running policy iteration for gamma {i}...")
            start = time.time()
            optimal_policy, k = policy_iteration(env, gamma=(i + 0.5) / 10)
            scores = evaluate_policy(env, optimal_policy, gamma=(i + 0.5) / 10)
            end = time.time()
            gammas[i] = (i + 0.5) / 10
            score_set[i] = np.mean(scores)
            iterations[i] = k
            times[i] = end - start
        # execution time
        plt.plot(gammas, times)
        plt.xlabel('Gamma Values')
        plt.grid()
        plt.ylabel('Time of execution')
        plt.title("Policy Iteration Execution Time Frozen Lake")
        plt.savefig(f"policy_iteration_frozen_lake_execution_time{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        print('Average scores = ', np.mean(scores))
        # scoring
        plt.plot(gammas, score_set)
        plt.xlabel('Gamma Values')
        plt.ylabel('Average Rewards')
        plt.title('Policy Iteration Rewards Frozen Lake')
        plt.grid()
        plt.savefig(f"policy_iteration_frozen_lake_rewards_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # convergence
        plt.plot(gammas, iterations)
        plt.xlabel('Gamma Values')
        plt.ylabel('Iterations to Converge')
        plt.title('Policy Iteration Convergence Frozen Lake')
        plt.grid()
        plt.savefig(f"policy_iteration_frozen_lake_convergence_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()

    if run_value_iteration:
        env_name = 'FrozenLake-v1'
        env = gym.make(env_name)
        env = env.unwrapped
        times = [0] * 10
        gammas = [0] * 10
        iterations = [0] * 10
        score_set = [0] * 10
        for i in range(0, 10):
            print(f"Running value iteration for gamma {i}...")
            start = time.time()
            optimal_v, k = value_iteration(env, gamma=(i + 0.5) / 10)
            policy = extract_policy(env, optimal_v, gamma=(i + 0.5) / 10)
            policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / 10, n=1000)
            end = time.time()
            score_set[i] = np.mean(policy_score)
            gammas[i] = (i + 0.5) / 10
            iterations[i] = k
            times[i] = end - start
        # execution time
        plt.plot(gammas, times)
        plt.xlabel('Gamma Values')
        plt.title("Value Iteration Execution Time Frozen Lake")
        plt.ylabel('Time of execution')
        plt.grid()
        plt.savefig(f"value_iteration_frozen_lake_execution_time{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        print('Value average score = ', policy_score)
        # scoring
        plt.plot(gammas, score_set)
        plt.xlabel('Gamma Values')
        plt.ylabel('Average Rewards')
        plt.title('Value Iteration Rewards Frozen Lake')
        plt.grid()
        plt.savefig(f"value_iteration_frozen_lake_rewards_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # convergence
        plt.plot(gammas, iterations)
        plt.xlabel('Gamma Values')
        plt.ylabel('Iterations to Converge')
        plt.title('Value Iteration Convergence Frozen Lake')
        plt.grid()
        plt.savefig(f"value_iteration_frozen_lake_convergence_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()

    if run_q_learning:
        times = []
        for epsilon in [0.05, 0.15, 0.25, 0.5, 0.75, 0.90]:
            agent = QLearningAgent('FrozenLake-v1',  alpha=0.85, gamma=0.95, episodes=100000, epsilon=epsilon)
            execution_time = agent.solve()
            times.append(execution_time)
        plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.90], times)
        plt.xlabel('Epsilon Values')
        plt.grid()
        plt.title('Q Learning Execution Time vs. Epsilon Values Frozen Lake')
        plt.ylabel('Execution Time (s)')
        plt.savefig(f"q_learning_frozen_lake_execution_times_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()


if __name__ == "__main__":
    main()