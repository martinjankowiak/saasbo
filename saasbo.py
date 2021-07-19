import math
import argparse
import time
from functools import partial

import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary

from jax import jit, vmap, value_and_grad
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from jax.scipy.stats import norm
from jax.scipy.special import expit as jax_expit

from numpyro.infer import MCMC, NUTS
from numpyro.util import enable_x64
from scipy.optimize import fmin_l_bfgs_b, minimize
from scipy.special import logsumexp, logit, expit
from torch.quasirandom import SobolEngine


root_five = math.sqrt(5.0)
five_thirds = 5.0 / 3.0


def get_chunks(L, chunk_size):
    num_chunks = L // chunk_size
    chunks = [jnp.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % chunk_size != 0:
        chunks.append(np.arange(L - L % chunk_size, L))
    return chunks


# fun should return tuples of arrays
def chunk_vmap(fun, array, chunk_size=4):
    L = array[0].shape[0]
    chunks = get_chunks(L, chunk_size)
    results = [vmap(fun)(*tuple([a[chunk] for a in array])) for chunk in chunks]
    num_results = len(results[0])
    return tuple([jnp.concatenate([r[k] for r in results]) for k in range(num_results)])


def kernel_diag(var, inv_length_sq, noise, jitter=1.0e-6, include_noise=True):
    if include_noise:
        return var + noise + jitter
    else:
        return var + jitter


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def rbf_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    k = var * jnp.exp(-0.5 * jnp.sum(deltaXsq, axis=-1))
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


# X, Z have shape (N_X, P) and (N_Z, P)
@partial(jit, static_argnums=(5,))
def matern_kernel(X, Z, var, inv_length_sq, noise, include_noise):
    deltaXsq = jnp.square(X[:, None, :] - Z) * inv_length_sq  # N_X N_Z P
    dsq = jnp.sum(deltaXsq, axis=-1)  # N_X N_Z
    exponent = root_five * jnp.sqrt(jnp.clip(dsq, a_min=1.0e-12))
    poly = 1.0 + exponent + five_thirds * dsq
    k = var * poly * jnp.exp(-exponent)
    if include_noise:
        k = k + (noise + 1.0e-6) * jnp.eye(X.shape[-2])
    return k  # N_X N_Z


class SAASGP(object):
    """
    This class contains the necessary modeling and inference code to fit a gaussian process with a SAAS prior.
    """
    def __init__(
        self,
        alpha=0.1,                 # controls sparsity
        num_warmup=200,            # number of HMC warmup samples
        num_samples=200,           # number of post-warmup HMC samples
        max_tree_depth=8,          # max tree depth used in NUTS
        num_chains=1,              # number of MCMC chains
        thinning=1,                # thinning > 1 reduces the computational cost at the risk of less robust model inferences
        verbose=True,              # whether to use std out for verbose logging
        observation_variance=0.0,  # observation variance to use; this is inferred if observation_variance==0.0
        kernel="matern"            # GP kernel to use (matern or rbf)
    ):
        if alpha <= 0.0:
            raise ValueError("The hyperparameter alpha should be positive.")
        if observation_variance < 0.0:
            raise ValueError("The hyperparameter observation_variance should be non-negative.")
        for i in [num_warmup, num_samples, max_tree_depth, num_chains, thinning]:
            if not isinstance(i, int) or i <= 0:
                raise ValueError("The hyperparameters num_warmup, num_samples, max_tree_depth, " +
                                 "num_chains, and thinning should be positive integers.")

        self.alpha = alpha
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.kernel = rbf_kernel if kernel == "rbf" else matern_kernel
        self.thinning = thinning
        self.verbose = verbose
        self.observation_variance = observation_variance
        self.learn_noise = (observation_variance == 0.0)
        self.Ls = None

    def model(self, X, Y):
        N, P = X.shape

        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
        if self.learn_noise:
            noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
        else:
            noise = 1e-6

        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(self.alpha))

        inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(P)))
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)
        k = self.kernel(X, X, var, inv_length_sq, noise, True)

        numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)

    def run_inference(self, rng_key, X, Y):
        start = time.time()
        kernel = NUTS(self.model, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(kernel, self.num_warmup, self.num_samples, num_chains=self.num_chains, progress_bar=self.verbose)
        mcmc.run(rng_key, X, Y)

        flat_samples = mcmc.get_samples(group_by_chain=False)
        chain_samples = mcmc.get_samples(group_by_chain=True)
        flat_summary = summary(flat_samples, prob=0.90, group_by_chain=False)

        rhat = flat_summary['kernel_inv_length_sq']['r_hat']
        print("[kernel_inv_length_sq] r_hat min/mean/max/median/q95:  {:.3f}  {:.3f}  {:.3f}  {:.3f}  {:.3f}".format(
              np.min(rhat), np.mean(rhat), np.max(rhat), np.median(rhat), np.percentile(rhat, [95.0]).item()))

        if self.verbose:
            mcmc.print_summary(exclude_deterministic=False)
            print("\nMCMC elapsed time:", time.time() - start)
        return chain_samples, flat_samples, flat_summary

    def compute_choleskys(self, chunk_size=8):
        def _cholesky(var, inv_length_sq, noise):
            k_XX = self.kernel(self.X_train, self.X_train, var, inv_length_sq, noise, True)
            return (cho_factor(k_XX, lower=True)[0],)

        n_samples = (self.num_samples * self.num_chains) // self.thinning
        vmap_args = (
            self.flat_samples["kernel_var"][:: self.thinning],
            self.flat_samples["kernel_inv_length_sq"][:: self.thinning],
            self.flat_samples["kernel_noise"][:: self.thinning] if self.learn_noise else 1e-6 * jnp.ones(n_samples),
        )

        self.Ls = chunk_vmap(_cholesky, vmap_args, chunk_size=chunk_size)[0]
    def predict(self, rng_key, X, Y, X_test, L, var, inv_length_sq, noise):
        k_pX = self.kernel(X_test, X, var, inv_length_sq, noise, False)
        mean = jnp.matmul(k_pX, cho_solve((L, True), Y))

        k_pp = kernel_diag(var, inv_length_sq, noise, include_noise=True)
        L_kXp = solve_triangular(L, jnp.transpose(k_pX), lower=True)
        diag_cov = k_pp - (L_kXp * L_kXp).sum(axis=0)

        return mean, diag_cov

    def fit(self, X_train, Y_train, seed=0):
        self.X_train, self.Y_train = X_train.copy(), Y_train.copy()
        self.rng_key_hmc, self.rng_key_predict = random.split(random.PRNGKey(seed), 2)
        self.chain_samples, self.flat_samples, self.summary = self.run_inference(self.rng_key_hmc, X_train, Y_train)
        return self

    def posterior(self, X_test):
        if self.Ls is None:
            self.compute_choleskys(chunk_size=8)

        n_samples = (self.num_samples * self.num_chains) // self.thinning
        vmap_args = (
            random.split(self.rng_key_predict, n_samples),
            self.flat_samples["kernel_var"][:: self.thinning],
            self.flat_samples["kernel_inv_length_sq"][:: self.thinning],
            self.flat_samples["kernel_noise"][:: self.thinning] if self.learn_noise else 1e-6 * jnp.ones(n_samples),
            self.Ls,
        )

        predict = lambda rng_key, var, inv_length_sq, noise, L: self.predict(
            rng_key, self.X_train, self.Y_train, X_test, L, var, inv_length_sq, noise
        )

        mean, var = chunk_vmap(predict, vmap_args, chunk_size=8)

        return mean, var


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
            maxfun=100,  # Because this is so damn slow
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


def main(args):
    X_train, Y_train, X_test, Y_test = get_data(
        N_train=args.num_data, P=args.P, seed=args.seed, sigma_obs=args.sigma_obs
    )

    gp = SAASGP(
        alpha=args.alpha,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        max_tree_depth=args.mtd,
        num_chains=args.num_chains,
        thinning=args.thinning,
        kernel=args.kernel,
    )

    gp = gp.fit(X_train, Y_train)
    for k, v in gp.summary.items():
        print("median_r_hat[{}]: {:.4f}".format(k, np.median(v["r_hat"])))

    mean, var = gp.posterior(X_test)

    test_rmse = np.sqrt(np.mean(np.square(Y_test - np.mean(mean, axis=0))))
    test_ll = -0.5 * np.square(Y_test - mean) / var - 0.5 * np.log(2.0 * np.pi * var)
    test_ll = np.mean(logsumexp(test_ll, axis=0)) - np.log(mean.shape[0])
    print("test_rmse: {:.4f}   test_ll: {:.4f}".format(test_rmse, test_ll))


if __name__ == "__main__":
    #assert numpyro.__version__.startswith("0.4.1")
    parser = argparse.ArgumentParser(description="GPBO")
    parser.add_argument("-n", "--num-samples", nargs="?", default=128, type=int)
    parser.add_argument("--P", default=32, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=128, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=64, type=int)
    parser.add_argument("--mtd", default=7, type=int)
    parser.add_argument("--thinning", default=4, type=int)
    parser.add_argument("--alpha", default=0.01, type=float)
    parser.add_argument("--sigma-obs", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--kernel", default="rbf", type=str, choices=["rbf", "matern"])
    parser.add_argument(
        "--prior",
        default="dirichlet",
        type=str,
        choices=["vanilla", "truncated", "regularized", "horseshoe", "dirichlet"],
    )
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    if args.device == "cpu":
        enable_x64()
    numpyro.set_host_device_count(args.num_chains)

    main(args)
