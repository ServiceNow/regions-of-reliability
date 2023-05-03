"""
Copyright 2022 ServiceNow
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

import glob
import json
import os
import pickle
from typing import Any, Dict

import pandas as pd

from . import compute_metrics


def compute_metrics_single_experiment(pickle_file: str) -> pd.Series:
    with open(pickle_file, "rb") as f:
        forecasts, targets = pickle.load(f)

    # Extract the part of the target which is relevant to the forecasts
    target_list = []
    samples_list = []
    for forecast, target in zip(forecasts, targets):
        target_data = target.iloc[-forecast.prediction_length :].to_numpy()
        target_list.append(target_data)
        samples_list.append(forecast.samples)

    raw_df = compute_metrics.compute_list(target_list, samples_list)

    # Take the mean over the various inference in the experiment, and the sum over the run times
    metrics_agg = {m: "mean" for m in compute_metrics.get_all_metrics()}
    timers_agg = {t: "sum" for t in compute_metrics.get_all_timers()}
    return raw_df.agg({**metrics_agg, **timers_agg})


def compute_metrics_from_benchmark(
    basefolder: str, filter: Dict[str, Any] = {}
) -> pd.DataFrame:
    # Compute all the metrics for all the experiments in the given benchmark folder.

    results = []

    for pickle_file in glob.glob(
        os.path.join(basefolder, "backtesting", "*", "forecasts_targets.pkl")
    ):
        folder = os.path.dirname(pickle_file)

        with open(os.path.join(folder, "exp_dict.json"), "r") as f:
            exp_dict = json.load(f)

        # Skip any experiment not matching the filter
        skip = False
        for k, v in filter.items():
            if exp_dict[k] != v:
                skip = True

        if not skip:
            metrics = compute_metrics_single_experiment(pickle_file)
            results.append({**exp_dict, **metrics})

    return pd.DataFrame(results)
