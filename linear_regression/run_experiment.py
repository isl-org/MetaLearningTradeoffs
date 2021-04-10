import numpy as np
import argparse, os

from algorithms import compare_mses

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--p", default=1, type=int)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--seed", type=int)

    ARGS = parser.parse_args()
    arg_dict = vars(ARGS)

    p = arg_dict["p"]
    beta = arg_dict["beta"]
    seed = arg_dict["seed"]
    np.random.seed(seed)

    experiment_path = "p-{}_beta-{}_seed-{}/".format(p, int(beta), seed)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    fig_path = experiment_path+"figures/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    compare_mses(p, beta, fig_path)
