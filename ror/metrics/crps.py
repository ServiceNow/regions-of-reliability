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


def crps_quantile(
    target: np.array,
    samples: np.array,
    quantiles: np.array = (np.arange(20) / 20.0)[1:],
) -> np.float32:
    # Compute the CRPS using the quantile scores
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]

    sorted_samples = np.sort(samples, axis=0)

    result_sum = np.float32(0)

    for q in quantiles:
        # From 0 to num_samples - 1 so that the 100% quantile is the last sample
        q_idx = int(np.round(q * (num_samples - 1)))
        q_value = sorted_samples[q_idx, :, :]

        # The absolute value is just there in case of numerical inaccuracies
        quantile_score = np.abs(2 * ((target <= q_value) - q) * (q_value - target))

        result_sum += np.nanmean(quantile_score, axis=(0, 1))

    return result_sum / len(quantiles)


def crps_slow(target: np.array, samples: np.array) -> np.float32:
    # Compute the CRPS using the definition:
    # CRPS(y, X) = E |y - X| + 0.5 E |X - X'|, averaged over each dimension
    assert target.shape[0] == samples.shape[1]
    assert target.shape[1] == samples.shape[2]
    num_samples = samples.shape[0]

    first_term = np.abs(samples - target[None, :, :]).mean(axis=0).mean()
    s = np.float32(0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            s += np.abs(samples[i] - samples[j]).mean()
    second_term = s / (num_samples * (num_samples - 1) / 2)

    return first_term - 0.5 * second_term
