"""
Copyright 2023 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy

# Find the parent root folder, and add its parent to the path
if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    while True:
        folder, file = os.path.split(file_path)
        if folder == file_path:
            raise RuntimeError("Could not find ror folder")
        elif file == "ror":
            sys.path.append(folder)
            break
        else:
            file_path = folder

    from ror.experiments.h0_vs_h1 import run_experiment

    # Arguments:
    # experiment name, dimensions, name of the output file (with extension)
    exp_name = sys.argv[1]
    exp_dim = int(sys.argv[2])
    output_filename = sys.argv[3]


from ror.experiments import h0_vs_h1 as exp_def


def get_threshold_for_alpha(d_mean, d_std, num_gt, alpha):
    return num_gt**0.5 * d_std * scipy.stats.norm.ppf(alpha)


def get_beta_from_threshold(d_mean, d_std, num_gt, threshold):
    return 1 - scipy.stats.norm.cdf(
        (threshold - num_gt * d_mean) / (num_gt**0.5 * d_std)
    )


def nll_beta(param, gen, dim, num_gt, alpha, num_samples):
    rng = np.random.default_rng(12345)

    pair_diff = gen(dim, param)
    dist_gt, dist_fcst = pair_diff.get_distributions()

    results = []
    targets_mult = dist_gt.sample(num_samples, rng)
    for i in range(num_samples):
        targets = targets_mult[i]
        nll_gt = -dist_gt.log_pdf(targets)
        nll_fcst = -dist_fcst.log_pdf(targets)
        results.append(nll_gt - nll_fcst)
    results = np.array(results)
    d_mean = results.mean()
    d_std = results.std()

    t = get_threshold_for_alpha(d_mean, d_std, num_gt, alpha)
    return get_beta_from_threshold(d_mean, d_std, num_gt, t)


def numerical_beta_diff(param, gen, dim, num_gt, alpha, beta_target, num_samples):
    return nll_beta(param, gen, dim, num_gt, alpha, num_samples) - beta_target


def calibrate_non_gaussian(exp, dim):
    gen = exp_def.EXP_GEN[exp]

    return scipy.optimize.root_scalar(
        numerical_beta_diff,
        args=(gen, dim, exp_def.NUM_DRAWS, exp_def.ALPHA, exp_def.BETA_NLL, 100000),
        bracket=exp_def.EXP_PARAMETER_RANGE[exp],
        xtol=1e-5,
    ).root


if __name__ == "__main__":
    result = calibrate_non_gaussian(exp_name, exp_dim)

    print(f"Result = {exp_dim} : {result}")
    with open(output_filename, "w") as f:
        f.write(f"{exp_dim} : {result},\n")
