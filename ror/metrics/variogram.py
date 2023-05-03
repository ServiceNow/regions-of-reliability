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


def variogram_score_tensor(
    target: np.array, samples: np.array, p: float = 1.0
) -> np.float32:
    assert p > 0
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]

    flat_target = target.reshape(-1)
    flat_samples = samples.reshape(samples.shape[0], -1)

    diff_target = flat_target[:, None] - flat_target[None, :]
    value_target = np.abs(diff_target) ** p

    # The first dimension contains the various samples
    diff_samples = flat_samples[:, :, None] - flat_samples[:, None, :]
    value_samples = (np.abs(diff_samples) ** p).mean(axis=0)

    return ((value_target - value_samples) ** 2).sum(axis=(0, 1))


def variogram_score_iter(
    target: np.array, samples: np.array, p: float = 1.0
) -> np.float32:
    assert p > 0
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]

    flat_target = target.reshape(-1)
    flat_samples = samples.reshape(samples.shape[0], -1)

    result = np.float32(0)
    for i in range(0, flat_target.shape[0]):
        for j in range(0, flat_target.shape[0]):
            diff_target = flat_target[i] - flat_target[j]
            value_target = np.abs(diff_target) ** p

            # The first dimension contains the various samples
            diff_samples = flat_samples[:, i] - flat_samples[:, j]
            value_samples = (np.abs(diff_samples) ** p).mean(axis=0)

            result += (value_target - value_samples) ** 2
    return result


def variogram_score_mixed(
    target: np.array, samples: np.array, p: float = 1.0
) -> np.float32:
    assert p >= 0
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]

    flat_target = target.reshape(-1)
    flat_samples = samples.reshape(samples.shape[0], -1)

    result = np.float32(0)
    for i in range(flat_target.shape[0]):
        diff_target = flat_target[i] - flat_target[:]
        if p > 0:
            value_target = np.abs(diff_target) ** p
        else:
            value_target = np.log(np.abs(diff_target).clip(min=EPSILON))

        # The first dimension contains the various samples
        diff_samples = flat_samples[:, i, None] - flat_samples[:, :]
        if p > 0:
            value_samples = (np.abs(diff_samples) ** p).mean(axis=0)
        else:
            value_samples = np.log(np.abs(diff_samples).clip(min=EPSILON)).mean(axis=0)

        result += ((value_target - value_samples) ** 2).sum(axis=0)

    return result
