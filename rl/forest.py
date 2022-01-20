import numpy as np
import time as time
import matplotlib.pyplot as plt
import datetime
import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp

run_policy_iteration = False
run_value_iteration = False
run_q_learning = True


def main():
    if run_policy_iteration:
        P, R = example.forest(S=2000)
        times = [0] * 10
        gammas = [0] * 10
        policy = [0] * 10
        iterations = [0] * 10
        score_set = [0] * 10
        for i in range(0, 10):
            print(f"Running policy iteration for gamma {i}...")
            start = time.time()
            pi = mdp.PolicyIteration(P, R, (i + 0.5) / 10)
            pi.run()
            gammas[i] = (i + 0.5) / 10
            policy[i] = pi.policy
            end = time.time()
            gammas[i] = (i + 0.5) / 10
            score_set[i] = np.mean(pi.V)
            iterations[i] = pi.iter
            times[i] = end - start
        # execution time
        plt.plot(gammas, times)
        plt.xlabel('Gamma Values')
        plt.grid()
        plt.ylabel('Time of execution')
        plt.title("Policy Iteration Execution Time Forest Management")
        plt.savefig(f"policy_iteration_forest_execution_time{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # print('Average scores = ', np.mean(scores))
        # scoring
        plt.plot(gammas, score_set)
        plt.xlabel('Gamma Values')
        plt.ylabel('Value Function')
        plt.title('Policy Iteration Value Function Forest Management')
        plt.grid()
        plt.savefig(f"policy_iteration_forest_vf_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # convergence
        plt.plot(gammas, iterations)
        plt.xlabel('Gamma Values')
        plt.ylabel('Iterations to Converge')
        plt.title('Policy Iteration Convergence Forest Management')
        plt.grid()
        plt.savefig(f"policy_iteration_forest_convergence_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()

    if run_value_iteration:
        P, R = example.forest(S=2000)
        policy = [0] * 10
        times = [0] * 10
        gammas = [0] * 10
        iterations = [0] * 10
        score_set = [0] * 10
        for i in range(0, 10):
            print(f"Running value iteration for gamma {i}...")
            start = time.time()
            pi = mdp.PolicyIteration(P, R, (i + 0.5) / 10)
            pi.run()
            gammas[i] = (i + 0.5) / 10
            policy[i] = pi.policy
            end = time.time()
            gammas[i] = (i + 0.5) / 10
            score_set[i] = np.mean(pi.V)
            iterations[i] = pi.iter
            times[i] = end - start
        # execution time
        plt.plot(gammas, times)
        plt.xlabel('Gamma Values')
        plt.title("Value Iteration Execution Time Forest Management")
        plt.ylabel('Time of execution')
        plt.grid()
        plt.savefig(f"value_iteration_forest_execution_time{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # scoring
        plt.plot(gammas, score_set)
        plt.xlabel('Gamma Values')
        plt.ylabel('Value Function')
        plt.title('Value Iteration Value Function Forest Management')
        plt.grid()
        plt.savefig(f"value_iteration_forest_vf_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        # convergence
        plt.plot(gammas, iterations)
        plt.xlabel('Gamma Values')
        plt.ylabel('Iterations to Converge')
        plt.title('Value Iteration Convergence Forest Management')
        plt.grid()
        plt.savefig(f"value_iteration_forest_convergence_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()

    if run_q_learning:
        times = []
        P, R = example.forest(S=2000, p=0.01)
        value_f = []
        policy = []
        times = []
        Q_table = []
        rew_array = []
        for epsilon in [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]:
            st = time.time()
            pi = mdp.QLearning(P, R, .99)
            pi.max_iter = 100000
            end = time.time()
            pi.epsilon = epsilon
            pi.run()
            stats = pi.run_stats
            max_v_set = []
            avg_reward = []
            for stat in stats:
                # print(stat)
                max_v = stat['Max V']
                reward = stat['Reward']
                avg_reward.append(reward)
                max_v_set.append(max_v)
            value_f.append(max_v_set)
            # print(epsilon, round(avg_reward/len(stats), 2))
            rew_array.append(avg_reward)
            policy.append(pi.policy)
            times.append(end - st)
            Q_table.append(pi.Q)
        # execution time
        # plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.90], times)
        # plt.xlabel('Epsilon Values')
        # plt.grid()
        # plt.title('Q Learning Execution Time vs. Epsilon Values Forest Management')
        # plt.ylabel('Execution Time (s)')
        # plt.savefig(f"q_learning_forest_execution_times_{str(datetime.datetime.now().isoformat())}.png")
        # plt.clf()
        # rewards
        plt.grid()
        plt.plot(range(0, 100000), value_f[0], label='epsilon=0.05')
        plt.plot(range(0, 100000), value_f[1], label='epsilon=0.15')
        plt.plot(range(0, 100000), value_f[2], label='epsilon=0.25')
        plt.plot(range(0, 100000), value_f[3], label='epsilon=0.50')
        plt.plot(range(0, 100000), value_f[4], label='epsilon=0.75')
        plt.plot(range(0, 100000), value_f[5], label='epsilon=0.95')
        plt.legend()
        plt.grid()
        plt.ylabel('Value Function')
        plt.xlabel('Iterations')
        plt.title('Q Learning Value Function vs. Epsilon Values Forest Management')
        plt.savefig(f"q_learning_forest_vf_{str(datetime.datetime.now().isoformat())}.png")
        plt.clf()
        plt.grid()
        plt.plot(range(0, 100000), rew_array[0], label='epsilon=0.05')
        plt.plot(range(0, 100000), rew_array[1], label='epsilon=0.15')
        plt.plot(range(0, 100000), rew_array[2], label='epsilon=0.25')
        plt.plot(range(0, 100000), rew_array[3], label='epsilon=0.50')
        plt.plot(range(0, 100000), rew_array[4], label='epsilon=0.75')
        plt.plot(range(0, 100000), rew_array[5], label='epsilon=0.95')
        plt.legend()
        plt.savefig(f"q_learning_forest_avg_reward_{str(datetime.datetime.now().isoformat())}.png")

        # plt.grid()
        # plt.plot([0.05, 0.15, 0.25, 0.5, 0.75, 0.95], rew_array)
        # plt.ylabel('Average Reward/Iteration')
        # plt.xlabel('Epsilon Value')
        # plt.title('Q Learning Average Reward vs. Epsilon Values Forest Management')
        # plt.savefig(f"q_learning_forest_avg_reward_{str(datetime.datetime.now().isoformat())}.png")
        # plt.clf()


if __name__ == "__main__":
    main()