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

import numpy as np
from typing import Optional
from ..distributions import Distribution


def gaussian_cross_entropy(
    mu1: np.array, cov1: np.array, mu2: np.array, cov2: np.array
) -> float:
    term1 = cov1.shape[0] * np.log(2 * np.pi)
    term2 = np.log(np.linalg.det(cov2))
    inv_cov_fcst = np.linalg.inv(cov2)
    term3 = (inv_cov_fcst * cov1).sum()
    delta = mu2 - mu1
    term4 = np.dot(delta, np.dot(inv_cov_fcst, delta))

    return 0.5 * (term1 + term2 + term3 + term4)


def gaussian_kl_divergence(
    mu_gt: np.array, cov_gt: np.array, mu_fcst: np.array, cov_fcst: np.array
) -> float:
    return gaussian_cross_entropy(
        mu_gt, cov_gt, mu_fcst, cov_fcst
    ) - gaussian_cross_entropy(mu_gt, cov_gt, mu_gt, cov_gt)


def exponential_cross_entropy(scale1: np.array, scale2: np.array) -> float:
    return (np.log(scale2) + (scale1 / scale2)).sum()


def exponential_kl_divergence(scale_gt: np.array, scale_fcst: np.array) -> float:
    return exponential_cross_entropy(scale_gt, scale_fcst) - exponential_cross_entropy(
        scale_gt, scale_gt
    )


def estimate_cross_entropy(
    dist1: Distribution, dist2: Distribution, num_samples: int, rng: np.random.Generator
) -> float:
    """
    The cross-entropy is -E_dist1(x)[log p_dist2(x)].
    """
    sum = 0.0
    for _ in range(num_samples):
        x = dist1.sample(1, rng)[0]
        sum += -dist2.log_pdf(x)
    return sum / num_samples


def estimate_kl(
    dist1: Distribution,
    dist2: Distribution,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    sum = 0.0
    for _ in range(num_samples):
        x = dist1.sample(1, rng)[0]
        sum += -dist2.log_pdf(x) + dist1.log_pdf(x)
    return sum / num_samples
    # This is equivalent for very large samples, but is much more noisy:
    # return estimate_cross_entropy(
    #     dist1, dist2, num_samples, rng
    # ) - estimate_cross_entropy(dist1, dist1, num_samples, rng)
