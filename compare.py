import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import nbinom
from scipy.special import comb
import numpy as np
import itertools
import progressbar
import argparse

SHOTS = 7
MIN_P = 0.1
MAX_P = 0.9
ERROR_THRESHOLD = 0.0001

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("t", choices=["heatmap", "table"])
    args = parser.parse_args()
    if args.t == "heatmap":
        plot_heatmap()
    elif args.t == "table":
        print_table()

def print_table():
    ps = [0.1, 0.25, 0.4, 0.5, 0.6, 0.835]
    l = len(ps)
    cells = [["x" for _ in range(l)]
        for _ in range(l)]
    for i, (pF, pS) in enumerate(itertools.product(ps, ps)):
        cells[i//l][i%l] = str(round(compute_p_first_wins(pF, pS), 4))
    fig, ax = plt.subplots()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.table(
        cellText=cells,
        cellLoc="center",
        colLabels=ps,
        rowLabels=ps,
        loc="center")
    ax.axis("off")
    plt.show()

def plot_heatmap():
    ps = np.arange(MIN_P, MAX_P, 0.01)
    grid = np.zeros((len(ps), len(ps)))
    with progressbar.ProgressBar(max_value=grid.size) as bar:
        for i, (pS, pF) in enumerate(
                itertools.product(reversed(ps), ps)):
            grid[i//len(ps), i%len(ps)] = compute_p_first_wins(pF, pS)
            bar.update(i)
    fig, ax = plt.subplots()
    hot = cm.get_cmap("viridis")
    im = ax.imshow(
        grid,
        extent=(MIN_P, MAX_P, MIN_P, MAX_P),
        cmap="hot",
        interpolation="antialiased",
        vmin=0.,
        vmax=1.)
    plt.xlabel("pF")
    plt.ylabel("pS")
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

def compute_p_first_wins(pF, pS):
    p = 0
    sum_s_probs = 0
    sum_f_probs = 0
    for s in itertools.count():
        prob_s = pascal_prob(s, pS)
        sum_s_probs += prob_s
        # We're tracking the sum of the probabilities
        # of all values of f <= s, as for those values,
        # the first player will win.
        sum_f_probs += pascal_prob(s, pF)
        p += prob_s * sum_f_probs
        if 1 - sum_s_probs < ERROR_THRESHOLD:
            # The remaining possible values of s can't
            # change the answer by more than ERROR_THRESHOLD.
            return p

def pascal_prob(k, p):
    return nbinom.pmf(k, SHOTS, p)

if __name__ == "__main__":
    main()
