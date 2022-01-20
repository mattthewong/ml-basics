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
        env_name = 'Taxi-v3'
        env = gym.make(env_name)
        env = env.unwrapped
        times = [0] * 10
        gammas = [0] * 10
        iterations = [0] * 10
        score_set = [0] * 10
        for i in range(3, 10):
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
        plt.ylabel('Time of execution')
        plt.grid()
        plt.title("Policy Iteration Execution Time Taxi V3")
        plt.savefig(f"policy_iteration_taxi_v3_execution_time{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        print('Average scores = ', np.mean(scores))
        # scores
        plt.plot(gammas, score_set)
        plt.xlabel('Gamma Values')
        plt.ylabel('Average Rewards')
        plt.title('Policy Iteration Rewards Taxi V3')
        plt.grid()
        plt.savefig(f"policy_iteration_taxi_v3_rewards_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # convergence
        plt.plot(gammas, iterations)
        plt.xlabel('Gamma Values')
        plt.ylabel('Iterations to Converge')
        plt.title('Policy Iteration Convergence Taxi V3')
        plt.grid()
        plt.savefig(f"policy_iteration_taxi_v3_convergence_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()

    if run_value_iteration:
        env_name = 'Taxi-v3'
        env = gym.make(env_name)
        env = env.unwrapped
        times = [0] * 10
        gammas = [0] * 10
        iterations = [0] * 10
        score_set = [0] * 10
        for i in range(2, 10):
            start = time.time()
            optimal_v, k = value_iteration(env, gamma=(i + 0.5) / 10);
            policy = extract_policy(env, optimal_v, gamma=(i + 0.5) / 10)
            policy_score = evaluate_policy(env, policy, gamma=(i + 0.5) / 10, n=1000)
            end = time.time()
            gammas[i] = (i + 0.5) / 10
            iterations[i] = k
            score_set[i] = np.mean(policy_score)
            times[i] = end - start
        plt.plot(gammas, times)
        plt.xlabel('Gamma Values')
        plt.title("Value Iteration Execution Time Taxi V3")
        plt.ylabel('Time of execution')
        plt.grid()
        plt.savefig(f"value_iteration_taxi_v3_execution_time{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        print('Policy average score = ', policy_score)
        # scoring
        plt.plot(gammas, score_set)
        plt.xlabel('Gamma Values')
        plt.ylabel('Average Rewards')
        plt.title('Value Iteration Rewards Taxi V3')
        plt.grid()
        plt.savefig(f"value_iteration_taxi_v3_rewards_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # convergence
        plt.plot(gammas, iterations)
        plt.xlabel('Gamma Values')
        plt.ylabel('Iterations to Converge')
        plt.title('Value Iteration Convergence Taxi V3')
        plt.grid()
        plt.savefig(f"value_iteration_taxi_v3_convergence_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
    if run_q_learning:
        agent = QLearningAgent('Taxi-v3', alpha=1.0, gamma=1.0, episodes=20000)
        agent.solve()


if __name__ == "__main__":
    main()
