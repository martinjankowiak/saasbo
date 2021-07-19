import argparse

import torch
from torch.quasirandom import SobolEngine
from botorch.test_functions import Hartmann

import jax.lax as lax
import jax.numpy as jnp
from jax import value_and_grad
from jax.scipy.stats import norm
from jax.scipy.special import expit as jax_expit

from scipy.optimize import fmin_l_bfgs_b, minimize
from scipy.special import logit, expit

import numpy as np
from saasgp import SAASGP
from copy import deepcopy
import time
from numpyro.util import enable_x64
import numpyro


def stddev(x, gp):
    mu, var = gp.posterior(x)
    sigma = jnp.sqrt(var)
    return sigma.mean(axis=0)


def stddev_grad(x, gp):
    return stddev(x, gp).sum()


def ei(x, y_target, gp, xi=0.0, return_std=False, ei_y=False):
    mu, var = gp.posterior(x)
    std = jnp.maximum(jnp.sqrt(var), 1e-6)
    improve = y_target - xi - mu
    scaled = improve / std
    cdf, pdf = norm.cdf(scaled), norm.pdf(scaled)
    exploit = improve * cdf
    explore = std * pdf
    values = jnp.nan_to_num(exploit + explore, nan=0.0)
    if ei_y:
        values = values - mu
    if return_std:
        return values.mean(axis=0), std.mean(axis=0)
    return values.mean(axis=0)


def ei_grad(x, y_target, gp, xi=0.0, ei_y=False):
    return ei(x, y_target, gp, xi, ei_y=ei_y).sum()


def optimize_ei(y_target, gp, xi=0.0, n_restarts=5, n_init=1000, ei_y=False):
    def negative_ei_and_grad(x, y_target, gp, xi, ei_y):
        """Compute EI and its gradient and then flip the signs since BFGS minimizes"""
        x = jnp.array(x.copy())[None, :]
        ei_val, ei_val_grad = value_and_grad(ei_grad)(x, y_target, gp, xi, ei_y)
        return -1 * ei_val.item(), -1 * np.array(ei_val_grad)

    dim = gp.X_train.shape[-1]
    X_rand = SobolEngine(dimension=dim, scramble=True).draw(n=n_init).numpy()
    x_best = gp.X_train[gp.Y_train.argmin(), :]
    X_rand[0, :] = np.clip(x_best + 0.001 * np.random.randn(1, dim), a_min=0.0, a_max=1.0)
    X_rand = jnp.array(X_rand)

    ei_rand = ei(X_rand, y_target, gp, ei_y=ei_y)
    _, top_inds = lax.top_k(ei_rand, n_restarts)
    X_init = X_rand[top_inds, :]

    x_best, y_best = None, -float("inf")
    for x0 in X_init:
        x, fx, _ = fmin_l_bfgs_b(
            func=negative_ei_and_grad,
            x0=x0,
            fprime=None,
            bounds=[(0.0, 1.0) for _ in range(dim)],
            args=(y_target, gp, 0.0, ei_y),
            maxfun=100,  # this limits computational cost
        )
        fx = -1 * fx  # Back to maximization

        if fx > y_best:
            x_best, y_best = x.copy(), fx

    return x_best


def unconstrained_ei(x, y_target, gp, xi):
    return -jnp.sum(ei(jax_expit(x), y_target, gp, xi))


def ei_min_op(x0, y_target, gp, xi):
    result = minimize(
        lambda x: unconstrained_ei(x[None, :], y_target, gp, xi),
        jnp.array(x0, dtype=np.float64),
        method="BFGS",
        options=dict(maxiter=16, line_search_maxiter=3),
    )
    return result.x, result.fun, result.nit


def unconstrained_optimize_ei(y_target, gp, xi=0.0, n_restarts=5, n_init=1000):
    dim = gp.X_train.shape[-1]
    X_rand = jnp.array(SobolEngine(dimension=dim, scramble=True).draw(n=n_init))
    ei_rand = ei(X_rand, y_target, gp)
    _, top_inds = lax.top_k(ei_rand, n_restarts)
    X_init = logit(X_rand[top_inds, :])

    x_best, y_best = None, -float("inf")
    for x0 in X_init:
        x, fx, nit = ei_min_op(x0, y_target, gp, xi)
        fx = -1 * fx  # Back to maximization

        if fx > y_best:
            x_best, y_best = x.copy(), fx

    return expit(x_best)


def run_saasbo(f, lb, ub, n_evals, n_init, seed=None, alpha=0.1, num_warmup=512, num_samples=256, thinning=16):
    ei_y = False
    device = "cpu"
    numpyro.set_platform(device)
    if device == "cpu":
        enable_x64()
    numpyro.set_host_device_count(1)

    X = SobolEngine(dimension=len(lb), scramble=True, seed=seed).draw(n=n_init).numpy()
    Y = np.array([f(lb + (ub - lb) * x) for x in X])
    print(f"Starting from {Y.min().item():.3f}")

    while len(Y) < n_evals:
        print(f"Iteration {len(Y)}", flush=True)
        train_Y = (Y - Y.mean()) / Y.std()
        y_target = train_Y.min().item()

        try:
            start = time.time()
            gp = SAASGP(
                alpha=alpha,
                num_warmup=num_warmup,
                num_samples=num_samples,
                max_tree_depth=6,
                num_chains=1,
                thinning=thinning,
                verbose=False,
                observation_variance=1.e-6,
            )
            gp = gp.fit(X, train_Y)
            print(f"GP fitting took {time.time() - start:.2f} seconds")

            start = time.time()
            x_next = optimize_ei(y_target, gp, xi=0.0, n_restarts=1, n_init=1000, ei_y=ei_y)
            print(f"Optimizing EI took {time.time() - start:.2f} seconds")
        except Exception as ex:
            print("EXCEPTION!\n", ex)
            x_next = np.random.rand(
                len(lb),
            )  # Random point

        y_next = f(lb + (ub - lb) * x_next)

        X = np.vstack((X, deepcopy(x_next[None, :])))
        Y = np.hstack((Y, deepcopy(y_next)))

        print(f"EI value: {ei(x_next[None, :], y_target, gp).item():.2e}")
        print(f"Next function value: {y_next:.3f}    Best function value: {Y.min():.3f}")

        del gp

    return lb + (ub - lb) * X, Y


def hartmann6_50(x):
    return Hartmann(6)(torch.tensor([x[19], x[14], x[43], x[37], x[16], x[3]]))


# demonstrate how to run SAASBO on the Hartmann6 function embedded in D=50 dimensions
def main(args):
    lb = np.zeros(100)
    ub = np.ones(100)

    num_init_evals = 10

    if args.max_evals <= num_init_evals:
        raise ValueError("Must choose max_evals > num_init_evals.")

    run_saasbo(hartmann6_50, lb, ub, args.max_evals, num_init_evals,
               seed=args.seed, alpha=0.1, num_warmup=256, num_samples=256, thinning=16)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.5")
    parser = argparse.ArgumentParser(description="We demonstrate how to run SAASBO.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-evals", default=20, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    if args.device == "cpu":
        enable_x64()
    numpyro.set_host_device_count(args.num_chains)

    main(args)
