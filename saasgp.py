import math
import time
from functools import partial

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit, vmap
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from util import chunk_vmap

root_five = math.sqrt(5.0)
five_thirds = 5.0 / 3.0


# compute diagonal component of kernel
def kernel_diag(var, noise, jitter=1.0e-6, include_noise=True):
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

    See below for arguments.
    """

    def __init__(
        self,
        alpha=0.1,  # controls sparsity
        num_warmup=512,  # number of HMC warmup samples
        num_samples=256,  # number of post-warmup HMC samples
        max_tree_depth=7,  # max tree depth used in NUTS
        num_chains=1,  # number of MCMC chains
        thinning=16,  # thinning > 1 reduces the computational cost at the risk of less robust model inferences
        verbose=True,  # whether to use stdout for verbose logging
        observation_variance=0.0,  # observation variance to use; this scalar value is inferred if observation_variance==0.0
        kernel="matern",  # GP kernel to use (matern or rbf)
    ):
        if alpha <= 0.0:
            raise ValueError("The hyperparameter alpha should be positive.")
        if observation_variance < 0.0:
            raise ValueError("The hyperparameter observation_variance should be non-negative.")
        if kernel not in ["matern", "rbf"]:
            raise ValueError("Allowed kernels are matern and rbf.")
        for i in [num_warmup, num_samples, max_tree_depth, num_chains, thinning]:
            if not isinstance(i, int) or i <= 0:
                raise ValueError(
                    "The hyperparameters num_warmup, num_samples, max_tree_depth, "
                    + "num_chains, and thinning should be positive integers."
                )

        self.alpha = alpha
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.max_tree_depth = max_tree_depth
        self.num_chains = num_chains
        self.kernel = rbf_kernel if kernel == "rbf" else matern_kernel
        self.thinning = thinning
        self.verbose = verbose
        self.observation_variance = observation_variance
        self.learn_noise = observation_variance == 0.0
        self.Ls = None

    # define the surrogate model. users who want to modify e.g. the prior on the kernel variance
    # should make their modifications here.
    def model(self, X, Y):
        N, P = X.shape

        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
        noise = (
            numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0)) if self.learn_noise else self.observation_variance
        )
        tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(self.alpha))

        # note we use deterministic to reparameterize the geometry
        inv_length_sq = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(P)))
        inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)

        k = self.kernel(X, X, var, inv_length_sq, noise, True)
        numpyro.sample("Y", dist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k), obs=Y)

    # run gradient-based NUTS MCMC inference
    def run_inference(self, rng_key, X, Y):
        start = time.time()
        kernel = NUTS(self.model, max_tree_depth=self.max_tree_depth)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=self.verbose,
        )
        mcmc.run(rng_key, X, Y)

        flat_samples = mcmc.get_samples(group_by_chain=False)
        chain_samples = mcmc.get_samples(group_by_chain=True)
        flat_summary = summary(flat_samples, prob=0.90, group_by_chain=False)

        if self.verbose:
            rhat = flat_summary["kernel_inv_length_sq"]["r_hat"]
            print(
                "[kernel_inv_length_sq] r_hat min/max/median:  {:.3f}  {:.3f}  {:.3f}".format(
                    np.min(rhat), np.max(rhat), np.median(rhat)
                )
            )

            mcmc.print_summary(exclude_deterministic=False)
            print("\nMCMC elapsed time:", time.time() - start)

        return chain_samples, flat_samples, flat_summary

    # compute cholesky factorization of kernel matrices (necessary to compute posterior predictions)
    def compute_choleskys(self, chunk_size=8):
        def _cholesky(var, inv_length_sq, noise):
            k_XX = self.kernel(self.X_train, self.X_train, var, inv_length_sq, noise, True)
            return (cho_factor(k_XX, lower=True)[0],)

        n_samples = (self.num_samples * self.num_chains) // self.thinning
        vmap_args = (
            self.flat_samples["kernel_var"][:: self.thinning],
            self.flat_samples["kernel_inv_length_sq"][:: self.thinning],
            self.flat_samples["kernel_noise"][:: self.thinning]
            if self.learn_noise
            else self.observation_variance * jnp.ones(n_samples),
        )

        self.Ls = chunk_vmap(_cholesky, vmap_args, chunk_size=chunk_size)[0]

    # make predictions at test points X_test for a single set of SAAS hyperparameters
    def predict(self, rng_key, X, Y, X_test, L, var, inv_length_sq, noise):
        k_pX = self.kernel(X_test, X, var, inv_length_sq, noise, False)
        mean = jnp.matmul(k_pX, cho_solve((L, True), Y))

        k_pp = kernel_diag(var, noise, include_noise=True)
        L_kXp = solve_triangular(L, jnp.transpose(k_pX), lower=True)
        diag_cov = k_pp - (L_kXp * L_kXp).sum(axis=0)

        return mean, diag_cov

    # fit SAASGP to training data
    def fit(self, X_train, Y_train, seed=0):
        self.X_train, self.Y_train = X_train.copy(), Y_train.copy()
        self.rng_key_hmc, self.rng_key_predict = random.split(random.PRNGKey(seed), 2)
        self.chain_samples, self.flat_samples, self.summary = self.run_inference(self.rng_key_hmc, X_train, Y_train)
        return self

    # compute predictions at X_test using inferred SAAS hyperparameters
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
