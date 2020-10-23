import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import nbinom

SUCCESSES = 7

def main():
    ps = [0.1, 0.5, 0.835]
    for p in ps:
        plt.clf()
        plot_pascal_distr(p, SUCCESSES)
        plt.ylim(bottom=0)
        plt.legend()
        plt.xlabel("missed shots")
        plt.ylabel("probability")
        plt.grid(linestyle="--")
        plt.tight_layout()
        plt.savefig(f"imgs/pascal-p{p}.png")
        if p == ps[-1]:
            plt.show()

def plot_pascal_distr(p, n):
    xs = []
    ys = []
    cum = 0
    k = 0
    while cum < 0.99:
        prob = nbinom.pmf(k, n, p)
        cum += prob
        xs.append(k)
        ys.append(prob)
        k += 1
    plt.gca().axvline(x=nbinom.mean(n, p), color="red")
    plt.plot(
        xs,
        ys,
        label="p={}".format(p),
        marker="o",
        linestyle="None",
        color="purple")

if __name__ == "__main__":
    main()
