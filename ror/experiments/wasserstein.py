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

from typing import List
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import linprog


def gaussian_wasserstein_distance(
    mu1: np.array, cov1: np.array, mu2: np.array, cov2: np.array
) -> np.float64:
    """
    Compute the square Wasserstein-2 distance between two Gaussian distributions.
    """
    return ((mu1 - mu2) ** 2).sum() + np.trace(
        cov1 + cov2 - 2 * np.real(sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1)))
    )


def gaussian_mixture_wasserstein_distance(
    weights1: List[float],
    weights2: List[float],
    mu1: List[np.array],
    cov1: List[np.array],
    mu2: List[np.array],
    cov2: List[np.array],
) -> np.float64:
    """
    Compute the MW_2 distance (see https://arxiv.org/pdf/1907.05254.pdf) between two Gaussian mixture distributions.
    """
    # Objective: minimize the Wasserstein distance between the linked distributions
    c = [
        gaussian_wasserstein_distance(mu1[i], cov1[i], mu2[j], cov2[j])
        for i in range(len(weights1))
        for j in range(len(weights2))
    ]

    # Constraints: all of the components from the first distribution must be associated with the all of the components of the second.
    A_eq = []
    b_eq = []
    for ii in range(len(weights1)):
        A_eq.append(
            [
                1 if i == ii else 0
                for i in range(len(weights1))
                for j in range(len(weights2))
            ]
        )
        b_eq.append(weights1[ii])
    for jj in range(len(weights2)):
        A_eq.append(
            [
                1 if j == jj else 0
                for i in range(len(weights1))
                for j in range(len(weights2))
            ]
        )
        b_eq.append(weights2[jj])

    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq)
    return res.fun
