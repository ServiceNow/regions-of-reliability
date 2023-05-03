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


EPSILON = 1e-10


def energy_score(target: np.array, samples: np.array, beta: float = 1.0) -> np.float32:
    assert 0 <= beta < 2
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]

    # The Frobenius norm of a matrix is equal to the Euclidean norm of its element:
    # the square root of the sum of the square of its elements
    norm = np.linalg.norm(samples - target[None, :, :], ord="fro", axis=(1, 2))
    if beta > 0:
        first_term = (norm**beta).mean()
    else:
        first_term = np.log(norm.clip(min=EPSILON)).mean()

    # For the second term of the energy score, we need two independant realizations of the distributions.
    # So we do a sum ignoring the i == j terms (which would be zero anyway), and we normalize by the
    # number of pairs of samples we have summed over.
    s = np.float32(0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            norm = np.linalg.norm(samples[i] - samples[j], ord="fro")
            if beta > 0:
                s += norm**beta
            else:
                s += np.log(norm if norm > EPSILON else EPSILON)
    second_term = s / (num_samples * (num_samples - 1) / 2)

    return first_term - 0.5 * second_term


def energy_score_fast(
    target: np.array, samples: np.array, beta: float = 1.0
) -> np.float32:
    assert 0 <= beta < 2
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]

    # The Frobenius norm of a matrix is equal to the Euclidean norm of its element:
    # the square root of the sum of the square of its elements
    norm = np.linalg.norm(samples - target[None, :, :], ord="fro", axis=(1, 2))
    first_term = (norm**beta).mean()

    # For the second term of the energy score, we need two independant realizations of the distributions.
    # So we do a sum ignoring the i == j terms (which would be zero anyway), and we normalize by the
    # number of pairs of samples we have summed over.
    s = np.float32(0)
    for i in range(num_samples // 2):
        j = i + num_samples // 2
        norm = np.linalg.norm(samples[i] - samples[j], ord="fro")
        if beta > 0:
            s += norm**beta
        else:
            s += np.log(norm if norm > EPSILON else EPSILON)
    second_term = s / (num_samples // 2)

    return first_term - 0.5 * second_term
