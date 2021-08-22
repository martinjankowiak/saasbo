import argparse

import numpy as np
import numpyro
from numpyro.util import enable_x64

from hartmann import hartmann6_50
from saasbo import run_saasbo


# demonstrate how to run SAASBO on the Hartmann6 function embedded in D=50 dimensions
def main(args):
    lb = np.zeros(50)
    ub = np.ones(50)
    num_init_evals = 20

    run_saasbo(
        hartmann6_50,
        lb,
        ub,
        args.max_evals,
        num_init_evals,
        seed=args.seed,
        alpha=0.01,
        num_warmup=256,
        num_samples=256,
        thinning=32,
        device=args.device,
    )


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.7")
    parser = argparse.ArgumentParser(description="We demonstrate how to run SAASBO.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-evals", default=50, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    enable_x64()
    numpyro.set_host_device_count(1)

    main(args)
