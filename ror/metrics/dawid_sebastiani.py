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


def dawid_sebastiani_score(target: np.array, samples: np.array) -> np.float32:
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]

    flat_target = target.reshape(-1)
    flat_samples = samples.reshape(samples.shape[0], -1)

    cov = np.cov(flat_samples, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    # This metric cannot handle cases where there are less samples than dimensions in the forecast.
    # In those cases, the determinant of the covariance matrix will naturally be 0,
    # leading to a -infinity metric score.
    sign, logdet = np.linalg.slogdet(cov)
    assert sign == 1, f"sign = {sign}, logdet = {logdet}, det = {np.linalg.det(cov)}"

    diff = flat_target - flat_samples.mean(axis=0)
    # Numpy matmul (@) automatically does the appropriate transpose when multiplying vectors with matrices
    quadratic_form = diff @ inv_cov @ diff

    return logdet + quadratic_form
