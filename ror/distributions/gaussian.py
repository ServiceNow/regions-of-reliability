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

from . import Distribution

import numpy as np
import math
from scipy.stats import skewnorm


class GaussianDistribution(Distribution):
    """
    A Gaussian distribution with an arbitrary mean and correlation matrix.
    """

    def __init__(
        self,
        num_time: int,
        num_series: int,
        mean: np.array,
        cov: np.array,
    ):
        assert mean.shape[0] == num_time * num_series
        assert cov.shape[0] == num_time * num_series
        assert cov.shape[1] == num_time * num_series

        self.num_time = num_time
        self.num_series = num_series
        self.cov = cov
        self.mean = mean

        self.det_cov = np.abs(np.linalg.det(self.cov))
        self.inv_cov = np.linalg.inv(self.cov)

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x = rng.multivariate_normal(
            mean=self.mean, cov=self.cov, size=num_samples, check_valid="raise"
        )
        return x.reshape(num_samples, self.num_time, self.num_series)

    def log_pdf(self, x):
        num_elements = self.num_time * self.num_series
        x = x.reshape(num_elements)

        l1 = -(num_elements / 2) * math.log(2 * math.pi)
        l2 = -0.5 * math.log(self.det_cov)
        l3 = -0.5 * (x - self.mean) @ self.inv_cov @ (x - self.mean)

        return l1 + l2 + l3


class IndependentSkewedGaussianDistribution(Distribution):
    """
    Each dimension is an independent skewed normal distribution,
    with mean = 0 and variance = 1.
    Done this way to get access to a faster sampling procedure.
    """

    def __init__(self, num_time: int, num_series: int, alpha: np.array):
        assert alpha.shape[0] == num_time * num_series

        self.num_time = num_time
        self.num_series = num_series

        mean, var = skewnorm.stats(a=alpha)
        self.alpha = alpha
        self.scale = var ** (-0.5)
        self.loc = -mean * self.scale
        self.delta = alpha / np.sqrt(1 + alpha**2)
        self.neg_delta = np.sqrt(1 - self.delta**2)

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x1 = rng.normal(
            loc=0, scale=1, size=(num_samples, self.num_time, self.num_series)
        )
        x2 = rng.normal(
            loc=0, scale=1, size=(num_samples, self.num_time, self.num_series)
        )

        raw_x = self.delta * np.abs(x1) + self.neg_delta * x2
        x = self.loc + self.scale * raw_x
        return x.reshape(num_samples, self.num_time, self.num_series)

    def log_pdf(self, x: np.array) -> float:
        num_elements = self.num_time * self.num_series
        x = x.reshape(num_elements)
        # Shape = [num time steps, num series]
        return skewnorm.logpdf(x, a=self.alpha, loc=self.loc, scale=self.scale).sum()
