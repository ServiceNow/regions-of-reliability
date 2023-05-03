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
import math
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist


def kernel_laplace(
    target: np.array, samples: np.array, gamma: float = 1.0
) -> np.float32:
    """
    The Laplace kernel is:
    k(x, y) = exp(-gamma ||x - y||_1),
    where ||z||_1 is the L_1 norm (the sum of absolute differences).
    """

    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]
    dim = target.shape[0] * target.shape[1]

    flat_target = target.reshape(dim)
    flat_samples = samples.reshape(num_samples, dim)

    target_sample = np.exp(
        -gamma * np.abs(flat_target[None, :] - flat_samples).sum(axis=1)
    ).mean()
    sample_sample = np.triu(
        np.exp(
            -gamma
            * np.abs(flat_samples[:, None, :] - flat_samples[None, :, :]).sum(axis=2)
        ),
        k=1,
    ).sum() / (num_samples * (num_samples - 1) / 2)

    return -target_sample + 0.5 * sample_sample


def kernel_laplace_spherical(
    target: np.array, samples: np.array, gamma: float = 1.0
) -> np.float32:
    """
    A modification of the Laplace kernel where the L_1 norm is replaced by the L_2 (Euclidean) norm.
    """
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]
    dim = target.shape[0] * target.shape[1]

    flat_target = target.reshape(dim)
    flat_samples = samples.reshape(num_samples, dim)

    target_sample = np.exp(
        -gamma * ((flat_target[None, :] - flat_samples) ** 2).sum(axis=1) ** 0.5
    ).mean()
    sample_sample = np.triu(
        np.exp(
            -gamma
            * ((flat_samples[:, None, :] - flat_samples[None, :, :]) ** 2).sum(axis=2)
            ** 0.5
        ),
        k=1,
    ).sum() / (num_samples * (num_samples - 1) / 2)

    return -target_sample + 0.5 * sample_sample


def kernel_matern(
    target: np.array, samples: np.array, nu: float = 1.5, gamma: float = 1.0
) -> np.float32:
    """
    The Matern kernel.
    If nu == 0.5, this is the same as the "spherical" Laplace kernel.
    """
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]
    dim = target.shape[0] * target.shape[1]

    flat_target = target.reshape(dim)
    flat_samples = samples.reshape(num_samples, dim)

    kernel = Matern(length_scale=1.0 / gamma, length_scale_bounds="fixed", nu=nu)

    target_sample = kernel(flat_target[None, :], flat_samples).mean()
    sample_sample = np.triu(kernel(flat_samples, flat_samples), k=1).sum() / (
        num_samples * (num_samples - 1) / 2
    )

    return -target_sample + 0.5 * sample_sample


def custom_matern(X, Y, nu, gamma, metric="cityblock"):
    dists = cdist(gamma * X, gamma * Y, metric=metric)

    if nu == 0.5:
        K = np.exp(-dists)
    elif nu == 1.5:
        K = dists * math.sqrt(3)
        K = (1.0 + K) * np.exp(-K)
    elif nu == 2.5:
        K = dists * math.sqrt(5)
        K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
    else:
        raise ValueError(f"nu ({nu}) must be 0.5, 1.5, or 2.5")
    return K


def kernel_matern_manhattan(
    target: np.array, samples: np.array, nu: float = 1.5, gamma: float = 1.0
) -> np.float32:
    """
    The Matern kernel, where the Euclidean norm is replaced by the L_1 (Manhattan) norm.
    If nu == 0.5, this is the same as the Laplace kernel.
    """
    # Adapted from: https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/gaussian_process/kernels.py#L1660

    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]
    dim = target.shape[0] * target.shape[1]

    flat_target = target.reshape(dim)
    flat_samples = samples.reshape(num_samples, dim)

    target_sample = custom_matern(
        flat_target[None, :], flat_samples, nu, gamma, "cityblock"
    ).mean()
    sample_sample = np.triu(
        custom_matern(flat_samples, flat_samples, nu, gamma, "cityblock"), k=1
    ).sum() / (num_samples * (num_samples - 1) / 2)

    return -target_sample + 0.5 * sample_sample
