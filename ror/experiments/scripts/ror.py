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

import os
import sys

# Find the parent ror folder, and add its parent to the path
if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    while True:
        folder, file = os.path.split(file_path)
        if folder == file_path:
            raise RuntimeError("Could not find ror folder")
        elif file == "ror":
            sys.path.append(folder)
            break
        else:
            file_path = folder

    from ror.experiments.h0_vs_h1 import run_experiment, DRAWS_PER_TRIAL

    # Arguments:
    # experiment name, metric name, trial number, dimensions, number of samples, name of the output file (without extension)
    # if using split trials, trial number is replaced by: {trial}S{start}-{end}
    exp_name = sys.argv[1]
    metric_name = sys.argv[2]
    if "S" in sys.argv[3]:
        trial, trial_range = sys.argv[3].split("S")
        trial_start, trial_end = trial_range.split("-")
        trial = int(trial)
        trial_start = int(trial_start)
        trial_end = int(trial_end)
    else:
        trial = int(sys.argv[3])
        trial_start = 0
        trial_end = DRAWS_PER_TRIAL
    dim = int(sys.argv[4])
    num_samples = int(sys.argv[5])
    output_filename = sys.argv[6]

    # Hack to skip Dawid-Sebastiani when dim >= num_samples
    if metric_name == "dawid_sebastiani" and dim >= num_samples:
        print(
            f"Not running {metric_name} for dim = {dim} and num_samples = {num_samples}"
        )
        # Keep the .running file to not rerun this dummy script
    else:
        run_experiment(
            exp_name=exp_name,
            metric_name=metric_name,
            trial=trial,
            trial_start=trial_start,
            trial_end=trial_end,
            dim=dim,
            num_samples=num_samples,
            output_filename=output_filename,
        )
        if os.path.exists(output_filename + ".running"):
            os.remove(output_filename + ".running")
