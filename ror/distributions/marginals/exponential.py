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
from scipy.stats import expon


class Exponential(Marginal):
    """
    An exponential distribution with a lambda = scale.
    """

    def __init__(self, scale):
        self.scale = scale

    def inverse_cdf(self, u: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        x = expon.ppf(u, loc=0, scale=1) * self.scale[None, :, :]
        return x

    def cdf(self, x: np.array) -> np.array:
        # Shape = [num samples, num time steps, num series]
        if len(x.shape) == 3:
            u = expon.cdf(x / self.scale[None, :, :], loc=0, scale=1)
        elif len(x.shape) == 2:
            u = expon.cdf(x / self.scale, loc=0, scale=1)
        return u

    def log_pdf(self, x: np.array) -> float:
        # Shape = [num time steps, num series]
        return expon.logpdf(x, loc=0, scale=self.scale).sum()
