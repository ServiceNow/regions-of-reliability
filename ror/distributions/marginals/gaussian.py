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

from .. import Marginal

import numpy as np
from scipy.stats import norm


class N01Marginal(Marginal):
    """
    A normal distribution with mean = 0 and variance = 1.
    """

    def inverse_cdf(self, u: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x = norm.ppf(u, loc=0, scale=1)
        return x

    def cdf(self, x: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        u = norm.cdf(x, loc=0, scale=1)
        return u

    def log_pdf(self, x: np.array) -> float:
        prob = norm.pdf(x)
        return np.log(prob).sum()


class NmuMarginal(Marginal):
    """
    A normal distribution with mean = mu and variance = 1.
    """

    def __init__(self, mu):
        self.mu = mu

    def inverse_cdf(self, u: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x = norm.ppf(u, loc=0, scale=1) + self.mu[None, :, :]
        return x

    def cdf(self, x: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        if len(x.shape) == 3:
            u = norm.cdf(x - self.mu[None, :, :], loc=0, scale=1)
        elif len(x.shape) == 2:
            u = norm.cdf(x - self.mu, loc=0, scale=1)
        return u

    def log_pdf(self, x: np.array) -> float:
        # Shape = [num time steps, num series]
        prob = norm.pdf(x - self.mu)
        return np.log(prob).sum()


class NmusigmaMarginal(Marginal):
    """
    A normal distribution with mean = mu and variance = sigma^2.
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def inverse_cdf(self, u: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x = norm.ppf(u, loc=0, scale=1) * self.sigma[None, :, :] + self.mu[None, :, :]
        return x

    def cdf(self, x: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        if len(x.shape) == 3:
            u = norm.cdf(
                (x - self.mu[None, :, :]) / self.sigma[None, :, :], loc=0, scale=1
            )
        elif len(x.shape) == 2:
            u = norm.cdf((x - self.mu) / self.sigma, loc=0, scale=1)
        return u

    def log_pdf(self, x: np.array) -> float:
        # Shape = [num time steps, num series]
        return norm.logpdf(x, loc=self.mu, scale=self.sigma).sum()
        # prob = norm.pdf((x - self.mu) / self.sigma)
        # return np.log(prob).sum() - np.log(self.sigma).sum()


class PerSeriesGaussianMarginal(Marginal):
    def __init__(
        self,
        num_time: int,
        num_series: int,
        rng: np.random.Generator,
        error_ratio: float = 0.0,
    ):
        self.num_time = num_time
        self.num_series = num_series

        # One value for each series
        assert 0 <= error_ratio <= 1.0
        base_mean = rng.normal(loc=0, scale=1, size=(num_series,))
        error_mean = rng.normal(loc=0, scale=1, size=(num_series,))
        base_std = rng.normal(loc=0, scale=1, size=(num_series,))
        error_std = rng.normal(loc=0, scale=1, size=(num_series,))
        self.mean = (1 - error_ratio) * base_mean + error_ratio * error_mean
        self.std = np.exp((1 - error_ratio) * base_std + error_ratio * error_std)

    def inverse_cdf(self, u: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x = norm.ppf(u, loc=0, scale=1)

        return (x * self.std[None, None, :]) + self.mean[None, None, :]
