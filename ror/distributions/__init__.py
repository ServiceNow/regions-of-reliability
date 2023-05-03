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

from abc import ABC, abstractmethod
import numpy as np


class Distribution(ABC):
    @abstractmethod
    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        """
        The output should have the following shape:
        [num_samples, number of timesteps, number of series].
        The number of timesteps and series must be passed in the class constructor.
        """
        pass

    def log_pdf(self, x: np.array) -> float:
        """
        Logarithm of the PDF for a single sample.
        x must then have the following shape:
        [number of timesteps, number of series]
        """
        raise NotImplementedError(f"log_pdf is not implemented for {type(self)}")


class Copula(ABC):
    @abstractmethod
    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        pass

    def log_pdf(self, x: np.array) -> float:
        raise NotImplementedError(f"log_pdf is not implemented for {type(self)}")


class Marginal(ABC):
    @abstractmethod
    def inverse_cdf(self, u: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        pass

    def cdf(self, x: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        # or shape = [num time steps, num series]
        raise NotImplementedError(f"cdf is not implemented for {type(self)}")

    def log_pdf(self, x: np.array) -> float:
        # Shape = [num time steps, num series]
        raise NotImplementedError(f"log_pdf is not implemented for {type(self)}")


class MixtureDistribution(Distribution):
    def __init__(
        self,
        dist1: Distribution,
        dist2: Distribution,
        ratio: float,
    ):
        self.dist1 = dist1
        self.dist2 = dist2
        self.ratio = ratio

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        # Shape = [num samples, num time steps, num series]
        u1 = self.dist1.sample(num_samples, rng)
        u2 = self.dist2.sample(num_samples, rng)
        q = rng.random(num_samples)[:, None, None]
        return u1 * (q < self.ratio) + u2 * (q >= self.ratio)

    def log_pdf(self, u) -> float:
        ll1 = (np.log(self.ratio) if self.ratio > 0 else -np.inf) + self.dist1.log_pdf(
            u
        )
        ll2 = (
            np.log(1 - self.ratio) if self.ratio < 1 else -np.inf
        ) + self.dist2.log_pdf(u)

        max_ll = max(ll1, ll2)

        return np.log(np.exp(ll1 - max_ll) + np.exp(ll2 - max_ll)) + max_ll


class TrivialCopula(Copula):
    """
    The copula where all variables are independant.
    """

    def __init__(self, num_time: int, num_series: int):
        self.num_time = num_time
        self.num_series = num_series

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        # Shape = [num samples, num time steps, num series]
        return rng.uniform(
            low=0, high=1, size=(num_samples, self.num_time, self.num_series)
        )

    def log_pdf(self, x: np.array) -> float:
        if (x >= 0).all() and (x <= 1).all():
            return 0.0
        else:
            return -np.inf


class MixtureCopula(Copula):
    def __init__(
        self,
        copula1: Copula,
        copula2: Copula,
        ratio: float,
    ):
        self.copula1 = copula1
        self.copula2 = copula2
        self.ratio = ratio

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        # Shape = [num samples, num time steps, num series]
        u1 = self.copula1.sample(num_samples, rng)
        u2 = self.copula2.sample(num_samples, rng)
        q = rng.random(num_samples)[:, None, None]
        return u1 * (q < self.ratio) + u2 * (q >= self.ratio)

    def log_pdf(self, u) -> float:
        ll1 = (
            np.log(self.ratio) if self.ratio > 0 else -np.inf
        ) + self.copula1.log_pdf(u)
        ll2 = (
            np.log(1 - self.ratio) if self.ratio < 1 else -np.inf
        ) + self.copula2.log_pdf(u)

        max_ll = max(ll1, ll2)

        return np.log(np.exp(ll1 - max_ll) + np.exp(ll2 - max_ll)) + max_ll


class TrivialMarginal(Marginal):
    """
    All marginal are Uniform[0,1].
    """

    def inverse_cdf(self, u: np.array) -> np.array:
        return u

    def cdf(self, x: np.array) -> np.array:
        return x.clip(0, 1)

    def log_pdf(self, x: np.array) -> float:
        if (x >= 0).all() and (x <= 1).all():
            return 0.0
        else:
            return -np.inf


class CombinedDistribution(Distribution):
    def __init__(self, copula: Copula, marginal: Marginal):
        self.copula = copula
        self.marginal = marginal

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.array:
        u = self.copula.sample(num_samples, rng)
        return self.marginal.inverse_cdf(u)

    def log_pdf(self, x: np.array) -> float:
        u = self.marginal.cdf(x)
        return self.marginal.log_pdf(x) + self.copula.log_pdf(u)
