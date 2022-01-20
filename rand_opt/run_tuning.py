import sys
from utils import find_opt_hyperparams
from func_four_peaks import FourPeaks
from func_continuous_peaks import ContinuousPeaks
from func_flip_flop import FlipFlop
from func_knapsack import KnapSack
from func_one_max import OneMax
from func_queens import Queens
from func_travelling_sales import TravellingSales
from func_maxkcolor import MaxKColor
from func_six_peaks import SixPeaks


def main():
    if len(sys.argv) != 1:
        print("Usage: python rand_opt_tuning.py")
        sys.exit(1)

    # Control setting boolean flags
    run_four_peaks = False  # run four peaks problem
    run_six_peaks = False  # run run_six peaks problem
    run_continuous_peaks = False  # run continuous peaks problem
    run_flip_flop = False  # run flip flop problem
    run_knapsack = False  # run knapsack problem
    run_queens = True  # run queens problem
    run_one_max = True  # run one_max problem
    run_travelling_sales = True  # run travelling sales problem
    run_maxkcolor = True  # run max k color problem

    # Four Peaks
    if run_four_peaks:
        fp = FourPeaks()
        results = find_opt_hyperparams(fp)
        print(results)

    # Six Peaks
    if run_six_peaks:
        sp = SixPeaks()
        results = find_opt_hyperparams(sp)
        print(results)

    # Continuous Peaks
    if run_continuous_peaks:
        cp = ContinuousPeaks()
        results = find_opt_hyperparams(cp)
        print(results)

    # Flip Flop
    if run_flip_flop:
        ff = FlipFlop()
        results = find_opt_hyperparams(ff)
        print(results)

    # Knapsack
    if run_knapsack:
        ks = KnapSack()
        results = find_opt_hyperparams(ks)
        print(results)

    # One Max
    if run_one_max:
        om = OneMax()
        results = find_opt_hyperparams(om)
        print(results)

    # Queens
    if run_queens:
        q = Queens()
        results = find_opt_hyperparams(q)
        print(results)

    # Travelling Sales
    if run_travelling_sales:
        ts = TravellingSales()
        results = find_opt_hyperparams(ts)
        print(results)

    # MaxKColor
    if run_maxkcolor:
        mkc = MaxKColor()
        results = find_opt_hyperparams(mkc)
        print(results)




if __name__ == "__main__":
    main()
