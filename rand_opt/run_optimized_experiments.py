import sys
from utils import plot_fitness_graph, plot_alg_runtimes, plot_iter_length
from func_four_peaks import FourPeaks
from func_continuous_peaks import ContinuousPeaks
from func_flip_flop import FlipFlop
from func_one_max import OneMax
from func_knapsack import KnapSack
from func_travelling_sales import TravellingSales
from func_maxkcolor import MaxKColor
from func_six_peaks import SixPeaks
from func_queens import Queens
import numpy as np
from numpy import zeros


def main():
    if len(sys.argv) != 1:
        print("Usage: python run_optimized_experiments.py")
        sys.exit(1)

    # Control setting boolean flags
    run_four_peaks = True  # run four peaks problem
    run_six_peaks = True  # run run_six_peaks problem
    run_continuous_peaks = False  # run continuous peaks problem
    run_flip_flop = False  # run flip flop problem
    run_knapsack = False  # run knapsack problem
    run_one_max = False  # run one_max problem
    run_queens = True  # run queens problem
    run_travelling_sales = False  # run travelling sales problem
    run_max_k_color = True  # run max_k_color sales problem
    plot_length = False
    plot_fitness = True  # plot fitness curve
    plot_times = True  # plot train and query times

    np.random.seed(51)

    max_iter_attempts = 5000

    lengths_to_evaluate = [25, 75, 125, 175]
    # Four Peaks
    if run_four_peaks:
        fp = FourPeaks()
        fp.set_configs(max_iter_attempts, max_iter_attempts)
        rhc_time, rhc_bs, rhc_bf, rhc_fc = fp.run_random_hill_climb()
        sa_time, sa_bs, sa_bf, sa_fc = fp.run_simulated_annealing()
        ga_time, ga_bs, ga_bf, ga_fc = fp.run_genetic_algorithm()
        mimic_time, mimic_bs, mimic_bf, mimic_fc = fp.run_mimic()
        if plot_fitness:
            plot_fitness_graph(fp, rhc_fc, sa_fc, ga_fc, mimic_fc)
        if plot_times:
            plot_alg_runtimes(fp.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            for length in lengths_to_evaluate:
                fp.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = fp.run_random_hill_climb()
                print(rhc_fc.shape[1])
                iterations_rhc[0] = rhc_fc.shape[1]
                sa_time, sa_bs, sa_bf, sa_fc = fp.run_simulated_annealing()
                iterations_sa[0] = sa_fc.shape[1]
                ga_time, ga_bs, ga_bf, ga_fc = fp.run_genetic_algorithm()
                iterations_ga[0] = ga_fc.shape[1]
                mimic_time, mimic_bs, mimic_bf, mimic_fc = fp.run_mimic()
                iterations_mimic[0] = mimic_fc.shape[1]
            plot_iter_length(fp, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)

    # Six Peaks
    if run_six_peaks:
        sp = SixPeaks()
        sp.set_configs(max_iter_attempts, max_iter_attempts)
        rhc_time, rhc_bs, rhc_bf, rhc_fc = sp.run_random_hill_climb()
        sa_time, sa_bs, sa_bf, sa_fc = sp.run_simulated_annealing()
        ga_time, ga_bs, ga_bf, ga_fc = sp.run_genetic_algorithm()
        mimic_time, mimic_bs, mimic_bf, mimic_fc = sp.run_mimic()
        if plot_fitness:
            plot_fitness_graph(sp, rhc_fc, sa_fc, ga_fc, mimic_fc)
        if plot_times:
            plot_alg_runtimes(sp.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            for length in lengths_to_evaluate:
                sp.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = sp.run_random_hill_climb()
                print(rhc_fc.shape[1])
                iterations_rhc[0] = rhc_fc.shape[1]
                sa_time, sa_bs, sa_bf, sa_fc = sp.run_simulated_annealing()
                iterations_sa[0] = sa_fc.shape[1]
                ga_time, ga_bs, ga_bf, ga_fc = sp.run_genetic_algorithm()
                iterations_ga[0] = ga_fc.shape[1]
                mimic_time, mimic_bs, mimic_bf, mimic_fc = sp.run_mimic()
                iterations_mimic[0] = mimic_fc.shape[1]
            plot_iter_length(sp, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)

    # Continuous Peaks
    if run_continuous_peaks:
        cp = ContinuousPeaks()
        # cp.set_configs(max_iter_attempts, max_iter_attempts)
        # rhc_time, rhc_bs, rhc_bf, rhc_fc = cp.run_random_hill_climb()
        # sa_time, sa_bs, sa_bf, sa_fc = cp.run_simulated_annealing()
        # ga_time, ga_bs, ga_bf, ga_fc = cp.run_genetic_algorithm()
        # mimic_time, mimic_bs, mimic_bf, mimic_fc = cp.run_mimic()
        # if plot_fitness:
        #     plot_fitness_graph(cp, rhc_fc, sa_fc, ga_fc, mimic_fc)
        # if plot_times:
        #     plot_alg_runtimes(cp.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            print("--CONTINUOUS PEAKS--")
            for i, length in enumerate(lengths_to_evaluate):
                cp.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = cp.run_random_hill_climb()
                iterations_rhc[i] = rhc_fc[:, 1].mean()
                print("Iterations for RHC at length {}: {}".format(length, iterations_rhc[i]))
                sa_time, sa_bs, sa_bf, sa_fc = cp.run_simulated_annealing()
                iterations_sa[i] = sa_fc[:, 1].mean()
                print("Iterations for SA at length {}: {}".format(length, iterations_sa[i]))
                ga_time, ga_bs, ga_bf, ga_fc = cp.run_genetic_algorithm()
                iterations_ga[i] = ga_fc[:, 1].mean()
                print("Iterations for GA at length {}: {}".format(length, iterations_ga[i]))
                mimic_time, mimic_bs, mimic_bf, mimic_fc = cp.run_mimic()
                iterations_mimic[i] = mimic_fc[:, 1].mean()
                print("Iterations for MIMIC at length {}: {}".format(length, iterations_mimic[i]))
            plot_iter_length(cp, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)

    # Knapsack
    if run_knapsack:
        ks = KnapSack()
        ks.set_configs(max_iter_attempts, max_iter_attempts)
        rhc_time, rhc_bs, rhc_bf, rhc_fc = ks.run_random_hill_climb()
        sa_time, sa_bs, sa_bf, sa_fc = ks.run_simulated_annealing()
        ga_time, ga_bs, ga_bf, ga_fc = ks.run_genetic_algorithm()
        mimic_time, mimic_bs, mimic_bf, mimic_fc = ks.run_mimic()
        if plot_fitness:
            plot_fitness_graph(ks, rhc_fc, sa_fc, ga_fc, mimic_fc)
        if plot_times:
            plot_alg_runtimes(ks.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            for length in lengths_to_evaluate:
                ks.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = ks.run_random_hill_climb()
                print(rhc_fc.shape[1])
                iterations_rhc[0] = rhc_fc.shape[1]
                sa_time, sa_bs, sa_bf, sa_fc = fp.run_simulated_annealing()
                iterations_sa[0] = sa_fc.shape[1]
                ga_time, ga_bs, ga_bf, ga_fc = fp.run_genetic_algorithm()
                iterations_ga[0] = ga_fc.shape[1]
                mimic_time, mimic_bs, mimic_bf, mimic_fc = fp.run_mimic()
                iterations_mimic[0] = mimic_fc.shape[1]
            plot_iter_length(ks, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)

    # Flip Flop
    if run_flip_flop:
        ff = FlipFlop()
        # ff.set_configs(max_iter_attempts, max_iter_attempts)
        # rhc_time, rhc_bs, rhc_bf, rhc_fc = ff.run_random_hill_climb()
        # sa_time, sa_bs, sa_bf, sa_fc = ff.run_simulated_annealing()
        # ga_time, ga_bs, ga_bf, ga_fc = ff.run_genetic_algorithm()
        # mimic_time, mimic_bs, mimic_bf, mimic_fc = ff.run_mimic()
        # if plot_fitness:
        #     plot_fitness_graph(ff, rhc_fc, sa_fc, ga_fc, mimic_fc)
        # if plot_times:
        #     plot_alg_runtimes(ff.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            print("--FLIP FLOP--")
            for i, length in enumerate(lengths_to_evaluate):
                ff.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = ff.run_random_hill_climb()
                iterations_rhc[i] = rhc_fc[:, 1].mean()
                print("Iterations for RHC at length {}: {}".format(length, iterations_rhc[i]))
                sa_time, sa_bs, sa_bf, sa_fc = ff.run_simulated_annealing()
                iterations_sa[i] = sa_fc[:, 1].mean()
                print("Iterations for SA at length {}: {}".format(length, iterations_sa[i]))
                ga_time, ga_bs, ga_bf, ga_fc = ff.run_genetic_algorithm()
                iterations_ga[i] = ga_fc[:, 1].mean()
                print("Iterations for GA at length {}: {}".format(length, iterations_ga[i]))
                mimic_time, mimic_bs, mimic_bf, mimic_fc = ff.run_mimic()
                iterations_mimic[i] = mimic_fc[:, 1].mean()
                print("Iterations for MIMIC at length {}: {}".format(length, iterations_mimic[i]))
            plot_iter_length(ff, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)

    # OneMax
    if run_one_max:
        om = OneMax()
        # om.set_configs(max_iter_attempts, max_iter_attempts)
        # rhc_time, rhc_bs, rhc_bf, rhc_fc = om.run_random_hill_climb()
        # sa_time, sa_bs, sa_bf, sa_fc = om.run_simulated_annealing()
        # ga_time, ga_bs, ga_bf, ga_fc = om.run_genetic_algorithm()
        # mimic_time, mimic_bs, mimic_bf, mimic_fc = om.run_mimic()
        # if plot_fitness:
        #     plot_fitness_graph(om, rhc_fc, sa_fc, ga_fc, mimic_fc)
        # if plot_times:
        #     plot_alg_runtimes(om.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            print("--ONE MAX--")
            for length in enumerate(lengths_to_evaluate):
                om.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = om.run_random_hill_climb()
                iterations_rhc[i] = rhc_fc[:, 1].mean()
                print("Iterations for RHC at length {}: {}".format(length, iterations_rhc[i]))
                sa_time, sa_bs, sa_bf, sa_fc = om.run_simulated_annealing()
                iterations_sa[i] = sa_fc[:, 1].mean()
                print("Iterations for SA at length {}: {}".format(length, iterations_sa[i]))
                ga_time, ga_bs, ga_bf, ga_fc = om.run_genetic_algorithm()
                iterations_ga[i] = ga_fc[:, 1].mean()
                print("Iterations for GA at length {}: {}".format(length, iterations_ga[i]))
                mimic_time, mimic_bs, mimic_bf, mimic_fc = om.run_mimic()
                iterations_mimic[i] = mimic_fc[:, 1].mean()
                print("Iterations for MIMIC at length {}: {}".format(length, iterations_mimic[i]))
            plot_iter_length(om, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)

    # Queens
    if run_queens:
        q = Queens()
        q.set_configs(max_iter_attempts, max_iter_attempts)
        rhc_time, rhc_bs, rhc_bf, rhc_fc = q.run_random_hill_climb()
        sa_time, sa_bs, sa_bf, sa_fc = q.run_simulated_annealing()
        ga_time, ga_bs, ga_bf, ga_fc = q.run_genetic_algorithm()
        mimic_time, mimic_bs, mimic_bf, mimic_fc = q.run_mimic()
        if plot_fitness:
            plot_fitness_graph(q, rhc_fc, sa_fc, ga_fc, mimic_fc)
        if plot_times:
            plot_alg_runtimes(q.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            for length in lengths_to_evaluate:
                fp.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = fp.run_random_hill_climb()
                print(rhc_fc.shape[1])
                iterations_rhc[0] = rhc_fc.shape[1]
                sa_time, sa_bs, sa_bf, sa_fc = fp.run_simulated_annealing()
                iterations_sa[0] = sa_fc.shape[1]
                ga_time, ga_bs, ga_bf, ga_fc = fp.run_genetic_algorithm()
                iterations_ga[0] = ga_fc.shape[1]
                mimic_time, mimic_bs, mimic_bf, mimic_fc = fp.run_mimic()
                iterations_mimic[0] = mimic_fc.shape[1]
            plot_iter_length(q, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)
    # Travelling Salesman
    if run_travelling_sales:
        ts = TravellingSales()
        ts.set_configs(max_iter_attempts, max_iter_attempts)
        rhc_time, rhc_bs, rhc_bf, rhc_fc = ts.run_random_hill_climb()
        sa_time, sa_bs, sa_bf, sa_fc = ts.run_simulated_annealing()
        ga_time, ga_bs, ga_bf, ga_fc = ts.run_genetic_algorithm()
        mimic_time, mimic_bs, mimic_bf, mimic_fc = ts.run_mimic()
        if plot_fitness:
            plot_fitness_graph(ts, rhc_fc, sa_fc, ga_fc, mimic_fc)
        if plot_times:
            plot_alg_runtimes(ts.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            for length in lengths_to_evaluate:
                fp.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = fp.run_random_hill_climb()
                print(rhc_fc.shape[1])
                iterations_rhc[0] = rhc_fc.shape[1]
                sa_time, sa_bs, sa_bf, sa_fc = fp.run_simulated_annealing()
                iterations_sa[0] = sa_fc.shape[1]
                ga_time, ga_bs, ga_bf, ga_fc = fp.run_genetic_algorithm()
                iterations_ga[0] = ga_fc.shape[1]
                mimic_time, mimic_bs, mimic_bf, mimic_fc = fp.run_mimic()
                iterations_mimic[0] = mimic_fc.shape[1]
            plot_iter_length(ts, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)

    # MaxKcolor
    if run_max_k_color:
        mkc = MaxKColor()
        mkc.set_configs(max_iter_attempts, max_iter_attempts)
        rhc_time, rhc_bs, rhc_bf, rhc_fc = mkc.run_random_hill_climb()
        sa_time, sa_bs, sa_bf, sa_fc = mkc.run_simulated_annealing()
        ga_time, ga_bs, ga_bf, ga_fc = mkc.run_genetic_algorithm()
        mimic_time, mimic_bs, mimic_bf, mimic_fc = mkc.run_mimic()
        if plot_fitness:
            plot_fitness_graph(mkc, rhc_fc, sa_fc, ga_fc, mimic_fc)
        if plot_times:
            plot_alg_runtimes(mkc.func_name, rhc_time, sa_time, ga_time, mimic_time)
        if plot_length:
            iterations_rhc = zeros(4)
            iterations_sa = zeros(4)
            iterations_ga = zeros(4)
            iterations_mimic = zeros(4)
            for length in lengths_to_evaluate:
                fp.set_configs(max_iter_attempts, max_iter_attempts, length)
                rhc_time, rhc_bs, rhc_bf, rhc_fc = fp.run_random_hill_climb()
                print(rhc_fc.shape[1])
                iterations_rhc[0] = rhc_fc.shape[1]
                sa_time, sa_bs, sa_bf, sa_fc = fp.run_simulated_annealing()
                iterations_sa[0] = sa_fc.shape[1]
                ga_time, ga_bs, ga_bf, ga_fc = fp.run_genetic_algorithm()
                iterations_ga[0] = ga_fc.shape[1]
                mimic_time, mimic_bs, mimic_bf, mimic_fc = fp.run_mimic()
                iterations_mimic[0] = mimic_fc.shape[1]
            plot_iter_length(mkc, lengths_to_evaluate, iterations_rhc, iterations_sa, iterations_ga, iterations_mimic)


if __name__ == "__main__":
    main()
