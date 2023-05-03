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
from scipy.stats import skewnorm


class SkewedGaussian(Marginal):
    """
    A skewed normal distribution, with mean = 0 and variance = 1.
    """

    def __init__(self, alpha):
        mean, var = skewnorm.stats(a=alpha)
        self.alpha = alpha
        self.scale = var ** (-0.5)
        self.loc = -mean * self.scale

    def inverse_cdf(self, u: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x = (
            skewnorm.ppf(u, a=self.alpha, loc=0, scale=1) * self.scale[None, :, :]
            + self.loc
        )
        return x

    def cdf(self, x: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        if len(x.shape) == 3:
            u = skewnorm.cdf(
                (x - self.loc[None, :, :]) / self.scale[None, :, :],
                a=self.alpha,
                loc=0,
                scale=1,
            )
        elif len(x.shape) == 2:
            u = skewnorm.cdf((x - self.loc) / self.scale, a=self.alpha, loc=0, scale=1)
        return u

    def log_pdf(self, x: np.array) -> float:
        # Shape = [num time steps, num series]
        return skewnorm.logpdf(x, a=self.alpha, loc=self.loc, scale=self.scale).sum()
