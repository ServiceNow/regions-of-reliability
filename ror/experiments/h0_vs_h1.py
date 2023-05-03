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

# This file contains the various parameters for the experiments where
# we compute the False Negative rates for various metrics.
# The experiments are calibrated such that the false negative rate
# for the Negative Log-Likelihood is set to a specific value.

import os
import pickle
import time

import numpy as np
import pandas as pd
from .distributions import (
    MissingCovarianceBlockDiag,
    MissingCovarianceCheckerBoard,
    MissingCovarianceFull,
    MissingMixture,
    MissingSkewAllFast,
    MissingSkewSingleFast,
    WrongExponentialScalingAll,
    WrongExponentialScalingSingle,
    WrongMeanAll,
    WrongMeanSingle,
    WrongStdDevAll,
    WrongStdDevSingle,
)
from ..metrics.crps import crps_quantile, crps_slow
from ..metrics.dawid_sebastiani import dawid_sebastiani_score
from ..metrics.energy import energy_score, energy_score_fast
from ..metrics.variogram import variogram_score_mixed

# False Positive rate for H0 (assuming no signal between the distributions):
ALPHA = 0.05
# False Negative rate for H1 for the Negative Log-Likelihood
BETA_NLL = 0.2
# Number of draws from the ground-truth per statistical tests
NUM_DRAWS = 30

# Dimensions of the problems
DIM_LIST = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# How many samples to pull when computing the metrics
NUM_SAMPLES_LIST = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# How many ground-truth and metric computations to do for each trial
DRAWS_PER_TRIAL = 1000

# The seeds used for each trials
RNG_SEED_GT = [
    12345,
    112345,
    212345,
    312345,
    412345,
    512345,
    612345,
    712345,
    812345,
    912345,
]
RNG_SEED_FCST = [
    23456,
    123456,
    223456,
    323456,
    423456,
    523456,
    623456,
    723456,
    823456,
    923456,
]

# The list of the experiments, which are also the name of their results folders
EXP_LIST = [
    "wrong_mean_single",
    "wrong_mean_all",
    "wrong_std_single_lower",
    "wrong_std_single_higher",
    "wrong_std_all_lower",
    "wrong_std_all_higher",
    "wrong_exponential_single_lower",
    "wrong_exponential_single_higher",
    "wrong_exponential_all_lower",
    "wrong_exponential_all_higher",
    # "missing_skew_single", # NLL cannot reach a beta of 0.2, even when pushing the parameter to its limit
    "missing_skew_all",
    "missing_covariance_full",
    "extra_covariance_full",
    "missing_covariance_checker",
    "extra_covariance_checker",
    "missing_covariance_block",
    "extra_covariance_block",
    "missing_mixture",
    "extra_mixture",
]

# Functions which creates the pair distributions from the dimension and the free parameter
EXP_GEN = {
    "wrong_mean_single": lambda dim, param: WrongMeanSingle(dim, param, no_norm=True),
    "wrong_mean_all": lambda dim, param: WrongMeanAll(dim, param),
    "wrong_std_single_lower": lambda dim, param: WrongStdDevSingle(dim, param),
    "wrong_std_single_higher": lambda dim, param: WrongStdDevSingle(dim, param),
    "wrong_std_all_lower": lambda dim, param: WrongStdDevAll(dim, param),
    "wrong_std_all_higher": lambda dim, param: WrongStdDevAll(dim, param),
    "wrong_exponential_single_lower": lambda dim, param: WrongExponentialScalingSingle(
        dim, param
    ),
    "wrong_exponential_single_higher": lambda dim, param: WrongExponentialScalingSingle(
        dim, param
    ),
    "wrong_exponential_all_lower": lambda dim, param: WrongExponentialScalingAll(
        dim, param
    ),
    "wrong_exponential_all_higher": lambda dim, param: WrongExponentialScalingAll(
        dim, param
    ),
    "missing_skew_single": lambda dim, param: MissingSkewSingleFast(dim, param),
    "missing_skew_all": lambda dim, param: MissingSkewAllFast(dim, param),
    "missing_covariance_full": lambda dim, param: MissingCovarianceFull(dim, param),
    "extra_covariance_full": lambda dim, param: MissingCovarianceFull(
        dim, param, inverse=True
    ),
    "missing_covariance_checker": lambda dim, param: MissingCovarianceCheckerBoard(
        dim, param
    ),
    "extra_covariance_checker": lambda dim, param: MissingCovarianceCheckerBoard(
        dim, param, inverse=True
    ),
    "missing_covariance_block": lambda dim, param: MissingCovarianceBlockDiag(
        dim, param
    ),
    "extra_covariance_block": lambda dim, param: MissingCovarianceBlockDiag(
        dim, param, inverse=True
    ),
    "missing_mixture": lambda dim, param: MissingMixture(dim, param),
    "extra_mixture": lambda dim, param: MissingMixture(dim, param, inverse=True),
}

# Experiments with a different NLL_BETA target can reuse generators
EXP_GEN["extra_covariance_full_beta_10"] = EXP_GEN["extra_covariance_full"]
EXP_GEN["extra_covariance_full_beta_5"] = EXP_GEN["extra_covariance_full"]
EXP_GEN["missing_covariance_full_beta_10"] = EXP_GEN["missing_covariance_full"]
EXP_GEN["missing_covariance_full_beta_5"] = EXP_GEN["missing_covariance_full"]
EXP_GEN["wrong_std_single_higher_beta_10"] = EXP_GEN["wrong_std_single_higher"]
EXP_GEN["wrong_std_single_higher_beta_5"] = EXP_GEN["wrong_std_single_higher"]
EXP_GEN["wrong_std_single_lower_beta_10"] = EXP_GEN["wrong_std_single_lower"]
EXP_GEN["wrong_std_single_lower_beta_5"] = EXP_GEN["wrong_std_single_lower"]
# Experiments with a different ALPHA value can reuse generators
EXP_GEN["extra_covariance_full_alpha_10"] = EXP_GEN["extra_covariance_full"]
EXP_GEN["extra_covariance_full_alpha_20"] = EXP_GEN["extra_covariance_full"]
EXP_GEN["missing_covariance_full_alpha_10"] = EXP_GEN["missing_covariance_full"]
EXP_GEN["missing_covariance_full_alpha_20"] = EXP_GEN["missing_covariance_full"]
EXP_GEN["wrong_std_single_higher_alpha_10"] = EXP_GEN["wrong_std_single_higher"]
EXP_GEN["wrong_std_single_higher_alpha_20"] = EXP_GEN["wrong_std_single_higher"]
EXP_GEN["wrong_std_single_lower_alpha_10"] = EXP_GEN["wrong_std_single_lower"]
EXP_GEN["wrong_std_single_lower_alpha_20"] = EXP_GEN["wrong_std_single_lower"]

# What are the possible parameters values for each experiment
EXP_PARAMETER_RANGE = {
    "wrong_mean_single": (0.0, 10.0),
    "wrong_mean_all": (0.0, 10.0),
    "wrong_std_single_lower": (0.01, 1),
    "wrong_std_single_higher": (1, 10.0),
    "wrong_std_all_lower": (0.01, 1),
    "wrong_std_all_higher": (1.0, 10.0),
    "wrong_exponential_single_lower": (0.01, 1),
    "wrong_exponential_single_higher": (1, 10.0),
    "wrong_exponential_all_lower": (0.01, 1),
    "wrong_exponential_all_higher": (1, 10.0),
    "missing_skew_single": (0, 100.0),
    "missing_skew_all": (0, 100.0),
    "missing_covariance_full": (0, 0.99),
    "extra_covariance_full": (0, 0.99),
    "missing_covariance_checker": (0, 0.99),
    "extra_covariance_checker": (0, 0.99),
    "missing_covariance_block": (0, 0.99),
    "extra_covariance_block": (0, 0.99),
    "missing_mixture": (0.01, 10.0),
    "extra_mixture": (0.01, 10.0),
}

# For each experiment, and for each dimension, the value of the calibration parameter.
# This is chosen such that beta = BETA_NLL for the NLL metric when alpha = ALPHA and
# the number of samples per test is NUM_DRAWS.
EXP_CALIBRATION = {
    "extra_covariance_block": {
        16: 0.32005755247861506,
        32: 0.2268115789612484,
        64: 0.16046952817833035,
        128: 0.11348600709458659,
        256: 0.08024915019654844,
        512: 0.056748060975906615,
        1024: 0.040125028853900964,
        2048: 0.028374110772231533,
        4096: 0.020062528644776274,
    },
    "extra_covariance_checker": {
        16: 0.12684348785786603,
        32: 0.0628763430068676,
        64: 0.03117832890862141,
        128: 0.015510606482954978,
        256: 0.007734375,
        512: 0.0038621874999999984,
        1024: 0.001928593749999999,
        2048: 0.0009617968749999995,
        4096: 0.0004833984375,
    },
    "extra_covariance_full": {
        16: 0.12684348785786603,
        32: 0.0628763430068676,
        64: 0.03117832890862141,
        128: 0.015510606482954978,
        256: 0.007734375,
        512: 0.0038621874999999984,
        1024: 0.001928593749999999,
        2048: 0.0009617968749999995,
        4096: 0.0004833984375,
    },
    "missing_covariance_block": {
        16: 0.3057528263295386,
        32: 0.22137469601305318,
        64: 0.1584772818070364,
        128: 0.11276838440030243,
        256: 0.07999301490898172,
        512: 0.05665224851001018,
        1024: 0.04009279265491415,
        2048: 0.02836272005947263,
        4096: 0.020058492244730084,
    },
    "missing_covariance_checker": {
        16: 0.20549208277013448,
        32: 0.12181485921007722,
        64: 0.06800858148986216,
        128: 0.036301932282043085,
        256: 0.01881914667622033,
        512: 0.009591927975370942,
        1024: 0.004840605237371607,
        2048: 0.0024323546714514043,
        4096: 0.0012192247742747596,
    },
    "missing_covariance_full": {
        16: 0.20549208277013448,
        32: 0.12181485921007722,
        64: 0.06800858148986216,
        128: 0.036301932282043085,
        256: 0.01881914667622033,
        512: 0.009591927975370942,
        1024: 0.004840605237371607,
        2048: 0.0024323546714514043,
        4096: 0.0012192247742747596,
    },
    "wrong_mean_all": {
        16: 0.22698301950502112,
        32: 0.16050126132446094,
        64: 0.11349150975250578,
        128: 0.08024923209719853,
        256: 0.05674575487621992,
        512: 0.040127116048258464,
        1024: 0.028372877436305264,
        2048: 0.020063558025688884,
        4096: 0.01418540414577501,
    },
    "wrong_mean_single": {
        16: 0.9079320780200845,
        32: 0.9079320780200654,
        64: 0.9079320780200463,
        128: 0.9079320780206583,
        256: 0.9079320780195187,
        512: 0.9079320780273185,
        1024: 0.9079320779617684,
        2048: 0.9079320780528087,
        4096: 0.9079320776880028,
    },
    "wrong_std_all_higher": {
        16: 1.185459837852242,
        32: 1.1254011711249432,
        64: 1.0860197765413664,
        128: 1.0595684542946626,
        256: 1.0415175370496852,
        512: 1.0290609194576095,
        1024: 1.0204048400387786,
        2048: 1.0143564348682867,
        4096: 1.010116270997352,
    },
    "wrong_std_all_lower": {
        16: 0.8583539377265302,
        32: 0.8962832171961854,
        64: 0.9247679544763537,
        128: 0.9458120135876311,
        256: 0.961171569990072,
        512: 0.9722841213182117,
        1024: 0.9802646314784013,
        2048: 0.9859782917519458,
        4096: 0.9900518894270346,
    },
    "wrong_std_single_higher": {
        16: 2.451391866778791,
        32: 2.4513918667787955,
        64: 2.4513918667788417,
        128: 2.451391866778783,
        256: 2.4513918667786965,
        512: 2.4513918667794017,
        1024: 2.4513918667780477,
        2048: 2.4513918668015857,
        4096: 2.451391866848331,
    },
    "wrong_std_single_lower": {
        16: 0.579881232833752,
        32: 0.5798812328336749,
        64: 0.5798812328335538,
        128: 0.5798812328337933,
        256: 0.5798812328353367,
        512: 0.579881232828288,
        1024: 0.5798812329075603,
        2048: 0.5798812329295485,
        4096: 0.5798812333671054,
    },
    "wrong_exponential_single_lower": {
        16: 0.4481305228044053,
        32: 0.44866771348841866,
        64: 0.444650767851471,
        128: 0.4480870225347209,
        256: 0.45376444982106096,
        512: 0.4528365086664983,
        1024: 0.44628118211619633,
        2048: 0.4463173131600407,
        4096: 0.449286155437286,
    },
    "wrong_exponential_single_higher": {
        16: 3.0031801612659157,
        32: 3.039486035045125,
        64: 2.9980197834967894,
        128: 3.0000190822562653,
        256: 3.031601471765097,
        512: 3.030336474055931,
        1024: 3.0327118704446216,
        2048: 3.0496701525258736,
        4096: 3.05139203752406,
    },
    "wrong_exponential_all_lower": {
        16: 0.802843771071015,
        32: 0.8539343736914893,
        64: 0.8931535747618627,
        128: 0.9233175542599157,
        256: 0.9450826396725747,
        512: 0.9608706189266522,
        1024: 0.9721071103315705,
        2048: 0.9800019324497683,
        4096: 0.9858770299244973,
    },
    "wrong_exponential_all_higher": {
        16: 1.2666350340804042,
        32: 1.17782467660095,
        64: 1.1209278812947292,
        128: 1.0838286066923042,
        256: 1.0583658069971096,
        512: 1.0410506727849511,
        1024: 1.0289130790952128,
        2048: 1.020241491430306,
        4096: 1.0141970587664084,
    },
    "missing_skew_all": {
        16: 2.3987087621478884,
        32: 1.8089994801160285,
        64: 1.4737702864605207,
        128: 1.2036489108989459,
        256: 1.0149320412078116,
        512: 0.8743857895251127,
        1024: 0.7531846111252345,
        2048: 0.6555390834546319,
        4096: 0.5747938632001282,
    },
    "missing_mixture": {
        16: 0.5905809090463467,
        32: 0.41510040767914025,
        64: 0.29743338756458043,
        128: 0.20832912302141757,
        256: 0.1479542178659061,
        512: 0.10531428066427649,
        1024: 0.0738853032729885,
        2048: 0.051897245342581486,
        4096: 0.036732201507065125,
    },
    "extra_mixture": {
        16: 0.8019927370284049,
        32: 0.5749210307564232,
        64: 0.40515842969286553,
        128: 0.2908668499042934,
        256: 0.20403518681266022,
        512: 0.14559845240812255,
        1024: 0.10322462325866288,
        2048: 0.07266666794435331,
        4096: 0.05159516844497624,
    },
    "extra_covariance_full_beta_10": {
        16: 0.164091873809959,
        32: 0.08117874626426465,
        64: 0.04000875554366879,
        128: 0.019816399554391614,
        256: 0.009855433844802577,
        512: 0.004913920628597884,
        1024: 0.0024558628298703344,
        2048: 0.0012278604900025872,
        4096: 0.0006113090547447927,
    },
    "extra_covariance_full_beta_5": {
        16: 0.2003904769691813,
        32: 0.0992746972443671,
        64: 0.04856790540341611,
        128: 0.023910928217221027,
        256: 0.011851341957882336,
        512: 0.0058965468622062195,
        1024: 0.0029439889557697804,
        2048: 0.0014702851899634712,
        4096: 0.0007347137136413683,
    },
    "missing_covariance_full_beta_10": {
        16: 0.27978512191767557,
        32: 0.18168260674931208,
        64: 0.11044901116181811,
        128: 0.0630984687357803,
        256: 0.0342716020498531,
        512: 0.017966157956155852,
        1024: 0.009216482356151818,
        2048: 0.00467212465642774,
        4096: 0.0023493965433711737,
    },
    "missing_covariance_full_beta_5": {
        16: 0.34683749651658546,
        32: 0.2443846092639489,
        64: 0.16324233450937634,
        128: 0.10277986348371136,
        256: 0.0608804592418805,
        512: 0.03409636501511163,
        1024: 0.01827236656827044,
        2048: 0.009501173431463124,
        4096: 0.004851428009869072,
    },
    "wrong_std_single_higher_beta_10": {
        16: 3.270589678549332,
        32: 3.270589678549338,
        64: 3.2705896785493436,
        128: 3.270589678549356,
        256: 3.2705896785491593,
        512: 3.2705896785505524,
        1024: 3.2705896785448467,
        2048: 3.2705896785461332,
        4096: 3.2705896785461332,
    },
    "wrong_std_single_higher_beta_5": {
        16: 4.616761423663534,
        32: 4.616761423663541,
        64: 4.616761423663526,
        128: 4.6167614236635846,
        256: 4.6167614236631245,
        512: 4.616761423663884,
        1024: 4.616761423667041,
        2048: 4.616761423662848,
        4096: 4.6167614236763335,
    },
    "wrong_std_single_lower_beta_10": {
        16: 0.5338615047078314,
        32: 0.5338615047078178,
        64: 0.5338615047078356,
        128: 0.5338615047074368,
        256: 0.5338615047065064,
        512: 0.533861504697786,
        1024: 0.5338615047247222,
        2048: 0.533861504648622,
        4096: 0.5338615049383538,
    },
    "wrong_std_single_lower_beta_5": {
        16: 0.4996494298548997,
        32: 0.4996494298548443,
        64: 0.4996494298548902,
        128: 0.49964942985498556,
        256: 0.4996494298573545,
        512: 0.49964942986689603,
        1024: 0.49964942988257344,
        2048: 0.49964942987635647,
        4096: 0.4996494303204035,
    },
    "extra_covariance_full_alpha_10": {
        16: 0.10077705259578434,
        32: 0.050035249756167105,
        64: 0.02488471841849668,
        128: 0.012403399838774658,
        256: 0.006189828862704874,
        512: 0.0030947093517854317,
        1024: 0.0015466671323122471,
        2048: 0.0007731602726168352,
        4096: 0.00038800699568960237,
    },
    "extra_covariance_full_alpha_20": {
        16: 0.07369678970096342,
        32: 0.03661094815056821,
        64: 0.018237833206560198,
        128: 0.009097642680479657,
        256: 0.004543955472834699,
        512: 0.0022707396038112986,
        1024: 0.001135057554153522,
        2048: 0.0005674503748997834,
        4096: 0.0002816578544130236,
    },
    "missing_covariance_full_alpha_10": {
        16: 0.1532080829343928,
        32: 0.08555102412272322,
        64: 0.04566307972446072,
        128: 0.023669909154440595,
        256: 0.012062480798907366,
        512: 0.006089365429807871,
        1024: 0.0030623060554673585,
        2048: 0.0015349765610947826,
        4096: 0.0007684495857734572,
    },
    "missing_covariance_full_alpha_20": {
        16: 0.10229734340431439,
        32: 0.05424650847341646,
        64: 0.028016632764008946,
        128: 0.014248708567198195,
        256: 0.007186934025153105,
        512: 0.003607183594524999,
        1024: 0.0018076858777700258,
        2048: 0.0009048715098428608,
        4096: 0.0004526935320240942,
    },
    "wrong_std_single_higher_alpha_10": {
        16: 2.037086292025281,
        32: 2.0370862920252666,
        64: 2.037086292025333,
        128: 2.0370862920252057,
        256: 2.037086292024277,
        512: 2.0370862920255277,
        1024: 2.037086292004683,
        2048: 2.0370862919878294,
        4096: 2.037086291762827,
    },
    "wrong_std_single_higher_alpha_20": {
        16: 1.690056307293806,
        32: 1.69005630729382,
        64: 1.6900563072937524,
        128: 1.6900563072938124,
        256: 1.690056307293152,
        512: 1.6900563072956771,
        1024: 1.6900563072968486,
        2048: 1.690056307307105,
        4096: 1.6900563076039135,
    },
    "wrong_std_single_lower_alpha_10": {
        16: 0.6223133587535926,
        32: 0.6223133587535443,
        64: 0.6223133587533379,
        128: 0.6223133587532617,
        256: 0.6223133587570525,
        512: 0.6223133587653129,
        1024: 0.6223133586754261,
        2048: 0.6223133588886992,
        4096: 0.6223133577183461,
    },
    "wrong_std_single_lower_alpha_20": {
        16: 0.680077201202112,
        32: 0.680077201202045,
        64: 0.6800772012022394,
        128: 0.6800772012017705,
        256: 0.6800772011989753,
        512: 0.6800772012186811,
        1024: 0.6800772011934512,
        2048: 0.6800772010634918,
        4096: 0.680077202002461,
    },
}


def select_params(func, **kwargs):
    # Select the value of the named parameters in the subfunction
    def params_func(target, samples):
        return func(target, samples, **kwargs)

    return params_func


# Give the function which can be called to get each of the metric.
METRIC_FUNCTIONS = {
    "crps_quantile": crps_quantile,
    "crps_slow": crps_slow,
    "energy_0": select_params(energy_score, beta=0),
    "energy_0.5": select_params(energy_score, beta=0.5),
    "energy_1": select_params(energy_score, beta=1),
    "energy_1.5": select_params(energy_score, beta=1.5),
    "energy_fast_0": select_params(energy_score_fast, beta=0),
    "energy_fast_0.5": select_params(energy_score_fast, beta=0.5),
    "energy_fast_1": select_params(energy_score_fast, beta=1),
    "energy_fast_1.5": select_params(energy_score_fast, beta=1.5),
    "variogram_0": select_params(variogram_score_mixed, p=0),
    "variogram_0.5": select_params(variogram_score_mixed, p=0.5),
    "variogram_1": select_params(variogram_score_mixed, p=1),
    "variogram_2": select_params(variogram_score_mixed, p=2),
    "dawid_sebastiani": dawid_sebastiani_score,
}


def run_experiment(
    exp_name: str,
    metric_name: str,
    trial: int,
    trial_start: int,
    trial_end: int,
    dim: int,
    num_samples: int,
    output_filename: str,
):
    param = EXP_CALIBRATION[exp_name][dim]
    dist_gt, dist_fcst = EXP_GEN[exp_name](dim, param).get_distributions()

    # Select an independant PRNG for each experiment, to allow us to easily
    # restart an experiment that was interrupted at any point.
    rng_gt_sequence = np.random.SeedSequence(RNG_SEED_GT[trial - 1])
    rng_fcst_sequence = np.random.SeedSequence(RNG_SEED_FCST[trial - 1])

    rng_gt_list = [
        np.random.default_rng(s) for s in rng_gt_sequence.spawn(DRAWS_PER_TRIAL)
    ]
    rng_fcst_list = [
        np.random.default_rng(s) for s in rng_fcst_sequence.spawn(DRAWS_PER_TRIAL)
    ]

    metric_func = METRIC_FUNCTIONS[metric_name]

    if os.path.exists(output_filename + ".pkl"):
        with open(output_filename + ".pkl", "rb") as f:
            results = pickle.load(f)
        starting_draw = trial_start + len(results)
        remaining_draws = trial_end - starting_draw
        print(f"Restarting from checkpoint {remaining_draws} draws left")
    else:
        results = []
        starting_draw = trial_start
        remaining_draws = trial_end - trial_start
        print(f"Starting from scratch {remaining_draws} draws left")
    print(f"Computing from draw {starting_draw} ({remaining_draws} draws left)")

    last_checkpoint_timer = time.time()

    for draw_id in range(starting_draw, trial_end):
        rng_gt = rng_gt_list[draw_id]
        rng_fcst = rng_fcst_list[draw_id]

        targets = dist_gt.sample(1, rng_gt)[0]

        forecasts = dist_fcst.sample(num_samples, rng_fcst)
        gt_forecasts = dist_gt.sample(num_samples, rng_fcst)

        metric_fcst = metric_func(targets, forecasts)
        metric_gt = metric_func(targets, gt_forecasts)

        results.append(
            {
                "metric": metric_name,
                "experiment": exp_name,
                "dim": dim,
                "num_samples": num_samples,
                "GT": metric_gt,
                "FCST": metric_fcst,
            }
        )

        # Save a checkpoint every hour instead of every X iterations,
        # since there is a lot of variety in run times.
        current_timer = time.time()
        if current_timer > last_checkpoint_timer + 3600:
            with open(output_filename + ".pkl", "wb") as f:
                pickle.dump(results, file=f)
                print(f"Saved {len(results)} results to checkpoint")
                last_checkpoint_timer = current_timer
    df = pd.DataFrame(results)

    df.to_csv(output_filename + ".csv", index=False)

    if os.path.exists(output_filename + ".pkl"):
        os.remove(output_filename + ".pkl")
