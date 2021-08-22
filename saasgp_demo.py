import argparse

import numpy as np
import numpyro
from numpyro.util import enable_x64
from scipy.special import logsumexp

from saasgp import SAASGP


# create artificial dataset for demonstration purposes
def get_data(N_train=200, N_test=200, P=20, sigma_obs=0.1, seed=0):
    np.random.seed(seed)
    N = N_train + N_test
    X = np.random.rand(N, P)

    Y = X[:, 0] + np.cos(X[:, 1]) * (1.0 - np.sin(X[:, 2]))
    Y -= np.mean(Y)
    Y /= np.std(Y)
    Y += sigma_obs * np.random.randn(N)

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X[:N_train], Y[:N_train], X[N_train:], Y[N_train:]


# We demonstrate how to fit a GP equipped with a SAAS prior.
def main(args):
    X_train, Y_train, X_test, Y_test = get_data(N_train=args.num_data, P=args.P, seed=args.seed)

    # define SAASGP
    gp = SAASGP(
        alpha=args.alpha,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_tree_depth=args.mtd,
        num_chains=args.num_chains,
        thinning=args.thinning,
        kernel=args.kernel,
    )

    # fit SAASGP to training data
    gp = gp.fit(X_train, Y_train)

    # report inference stats (r_hat should be close to 1.0 if inference results are to be trusted)
    for k, v in gp.summary.items():
        print("median_r_hat[{}]: {:.4f}".format(k, np.median(v["r_hat"])))

    # compute predictions at test points X_test for each posterior sample
    mean, var = gp.posterior(X_test)

    # compare predictions to actual Y_test
    test_rmse = np.sqrt(np.mean(np.square(Y_test - np.mean(mean, axis=0))))
    test_ll = -0.5 * np.square(Y_test - mean) / var - 0.5 * np.log(2.0 * np.pi * var)
    test_ll = np.mean(logsumexp(test_ll, axis=0)) - np.log(mean.shape[0])
    print("test_rmse: {:.4f}   test_ll: {:.4f}".format(test_rmse, test_ll))


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.7")
    parser = argparse.ArgumentParser(description="We demonstrate how to fit a SAASGP.")
    parser.add_argument("-n", "--num-samples", default=128, type=int)
    parser.add_argument("--P", default=32, type=int, help="dimension of input space")
    parser.add_argument("--num-warmup", default=128, type=int)
    parser.add_argument("--num-chains", default=1, type=int)
    parser.add_argument("--num-data", default=64, type=int)
    parser.add_argument("--mtd", default=7, type=int, help="max tree depth (NUTS hyperparameter)")
    parser.add_argument("--thinning", default=4, type=int)
    parser.add_argument("--alpha", default=0.01, type=float, help="controls SAAS sparsity level")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--kernel", default="rbf", type=str, choices=["rbf", "matern"])
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    enable_x64()
    numpyro.set_host_device_count(args.num_chains)

    main(args)
