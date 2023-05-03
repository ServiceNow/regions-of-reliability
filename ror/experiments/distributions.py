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

# This file contains various pairs of multivariate distributions,
# to be used in the various experiments.

import math
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from ..distributions import (
    CombinedDistribution,
    Distribution,
    MixtureDistribution,
    TrivialCopula,
)
from ..distributions.copulas.gaussian import CustomGaussianCopula
from ..distributions.gaussian import (
    GaussianDistribution,
    IndependentSkewedGaussianDistribution,
)
from ..distributions.marginals.gaussian import (
    N01Marginal,
    NmuMarginal,
    NmusigmaMarginal,
)
from ..distributions.marginals.exponential import Exponential
from ..distributions.marginals.generalized_gaussian import SkewedGaussian
from .kl_divergence import (
    estimate_kl,
    exponential_kl_divergence,
    gaussian_kl_divergence,
)
from .wasserstein import (
    gaussian_mixture_wasserstein_distance,
    gaussian_wasserstein_distance,
)


class DistributionPair(ABC):
    @abstractmethod
    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        """
        Return the ground truth distribution, and the forecasting distribution.
        """
        pass

    def get_kl_divergence(self) -> float:
        """
        Return the exact KL divergence between the ground-truth and the forecasting distributions.
        """
        raise NotImplementedError(
            f"get_kl_divergence() is not implemented for {type(self)}"
        )

    def get_wasserstein_distance(self) -> float:
        """
        Return the Wasserstein distance (or an approximation of it) between both distributions.
        """
        raise NotImplementedError(
            f"get_wasserstein_distance() is not implemented for {type(self)}"
        )


############################################
# Distributions varying by their marginals #
############################################


class WrongMeanSingle(DistributionPair):
    """
    Both distributions are N(mu,1) multivariate gaussians.
    Their means are identical except for one dimension where it differs by distance * sqrt(dim).
    """

    def __init__(self, dim: int, distance: float, no_norm: bool = False) -> None:
        super().__init__()
        self.dim = dim
        if no_norm:
            self.delta = distance
        else:
            # Stay backward-compatible with old experiments
            self.delta = distance * math.sqrt(dim)

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        mu_fcst = np.zeros((1, self.dim))
        mu_fcst[0][0] = self.delta
        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=NmuMarginal(mu=mu_fcst),
        )

        return dist_gt, dist_fcst

    def get_gaussian_parameters(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # GT mu, GT Sigma, FCST mu, FCST Sigma
        mu_gt = np.zeros(self.dim)
        sigma_gt = np.eye(self.dim)
        mu_fcst = np.zeros(self.dim)
        mu_fcst[0] = self.delta
        sigma_fcst = np.eye(self.dim)
        return mu_gt, sigma_gt, mu_fcst, sigma_fcst

    def get_kl_divergence(self) -> float:
        mu_fcst = np.zeros(self.dim)
        mu_fcst[0] = self.delta
        return gaussian_kl_divergence(
            np.zeros(self.dim),
            np.eye(self.dim),
            mu_fcst,
            np.eye(self.dim),
        )

    def get_wasserstein_distance(self) -> float:
        mu_fcst = np.zeros(self.dim)
        mu_fcst[0] = self.delta
        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            np.eye(self.dim),
            mu_fcst,
            np.eye(self.dim),
        )


class WrongMeanAll(DistributionPair):
    """
    Both distributions are N(mu,1) multivariate gaussians.
    Their differs by distance on all dimensions.
    """

    def __init__(self, dim: int, distance: float) -> None:
        super().__init__()
        self.dim = dim
        self.distance = distance

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        mu_fcst = self.distance * np.ones((1, self.dim))
        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=NmuMarginal(mu=mu_fcst),
        )

        return dist_gt, dist_fcst

    def get_gaussian_parameters(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # GT mu, GT Sigma, FCST mu, FCST Sigma
        mu_gt = np.zeros(self.dim)
        sigma_gt = np.eye(self.dim)
        mu_fcst = self.distance * np.ones(self.dim)
        sigma_fcst = np.eye(self.dim)
        return mu_gt, sigma_gt, mu_fcst, sigma_fcst

    def get_kl_divergence(self) -> float:
        mu_fcst = self.distance * np.ones(self.dim)
        return gaussian_kl_divergence(
            np.zeros(self.dim),
            np.eye(self.dim),
            mu_fcst,
            np.eye(self.dim),
        )

    def get_wasserstein_distance(self) -> float:
        mu_fcst = self.distance * np.ones(self.dim)
        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            np.eye(self.dim),
            mu_fcst,
            np.eye(self.dim),
        )


class WrongStdDevSingle(DistributionPair):
    """
    Both distributions are N(0,sigma) multivariate gaussians.
    Their standard deviation are identical except for one dimension where it differs by ratio.
    """

    def __init__(self, dim: int, ratio: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        sigma_gt = np.ones((1, self.dim))
        sigma_gt[0][0] = self.ratio
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=NmusigmaMarginal(mu=np.zeros((1, self.dim)), sigma=sigma_gt),
        )

        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_gaussian_parameters(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # GT mu, GT Sigma, FCST mu, FCST Sigma
        mu_gt = np.zeros(self.dim)
        sigma_diag_gt = np.ones(self.dim)
        sigma_diag_gt[0] = self.ratio
        sigma_gt = np.diag(sigma_diag_gt**2)
        mu_fcst = np.zeros(self.dim)
        sigma_fcst = np.eye(self.dim)
        return mu_gt, sigma_gt, mu_fcst, sigma_fcst

    def get_kl_divergence(self) -> float:
        sigma_gt = np.ones(self.dim)
        sigma_gt[0] = self.ratio
        cov_gt = np.diag(sigma_gt**2)
        cov_fcst = np.eye(self.dim)
        if self.inverse:
            cov_gt, cov_fcst = cov_fcst, cov_gt

        return gaussian_kl_divergence(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )

    def get_wasserstein_distance(self) -> float:
        sigma_gt = np.ones(self.dim)
        sigma_gt[0] = self.ratio
        cov_gt = np.diag(sigma_gt**2)
        cov_fcst = np.eye(self.dim)
        if self.inverse:
            cov_gt, cov_fcst = cov_fcst, cov_gt

        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )


class WrongStdDevAll(DistributionPair):
    """
    Both distributions are N(0,sigma) multivariate gaussians.
    Their standard deviations differ by a constant ratio over all dimensions.
    """

    def __init__(self, dim: int, ratio: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        sigma_gt = self.ratio * np.ones((1, self.dim))
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=NmusigmaMarginal(mu=np.zeros((1, self.dim)), sigma=sigma_gt),
        )

        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_gaussian_parameters(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # GT mu, GT Sigma, FCST mu, FCST Sigma
        mu_gt = np.zeros(self.dim)
        sigma_gt = np.eye(self.dim) * self.ratio**2
        mu_fcst = np.zeros(self.dim)
        sigma_fcst = np.eye(self.dim)
        return mu_gt, sigma_gt, mu_fcst, sigma_fcst

    def get_kl_divergence(self) -> float:
        sigma_gt = self.ratio * np.ones(self.dim)
        cov_gt = np.diag(sigma_gt**2)
        cov_fcst = np.eye(self.dim)
        if self.inverse:
            cov_gt, cov_fcst = cov_fcst, cov_gt

        return gaussian_kl_divergence(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )

    def get_wasserstein_distance(self) -> float:
        sigma_gt = self.ratio * np.ones(self.dim)
        cov_gt = np.diag(sigma_gt**2)
        cov_fcst = np.eye(self.dim)
        if self.inverse:
            cov_gt, cov_fcst = cov_fcst, cov_gt

        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )


class WrongExponentialScalingSingle(DistributionPair):
    """
    Both distributions are made out of multiple independants exponential distributions.
    The scale of each exponential distribution is the same, except for one from the ground-truth.
    """

    def __init__(self, dim: int, ratio: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        scale_gt = np.ones((1, self.dim))
        scale_gt[0][0] = self.ratio
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=Exponential(scale=scale_gt),
        )

        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=Exponential(scale=np.ones((1, self.dim))),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_kl_divergence(self) -> float:
        scale_gt = np.ones((1, self.dim))
        scale_gt[0][0] = self.ratio
        scale_fcst = np.ones((1, self.dim))
        if self.inverse:
            scale_gt, scale_fcst = scale_fcst, scale_gt

        return exponential_kl_divergence(scale_gt, scale_fcst)


class WrongExponentialScalingAll(DistributionPair):
    """
    Both distributions are made out of multiple independants exponential distributions.
    The scale of each exponential distribution is the same inside each multivariate distribution,
    but differs between the ground-truth and the forecast.
    """

    def __init__(self, dim: int, ratio: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        scale_gt = self.ratio * np.ones((1, self.dim))
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=Exponential(scale=scale_gt),
        )

        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=Exponential(scale=np.ones((1, self.dim))),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_kl_divergence(self) -> float:
        scale_gt = self.ratio * np.ones((1, self.dim))
        scale_fcst = np.ones((1, self.dim))
        if self.inverse:
            scale_gt, scale_fcst = scale_fcst, scale_gt

        return exponential_kl_divergence(scale_gt, scale_fcst)


class MissingSkewSingle(DistributionPair):
    """
    The ground-truth contains a single skewed normal distribution,
    and (dim-1) independant gaussian distributions.
    The forecast only has independant gaussian distributions.
    """

    def __init__(self, dim: int, skew: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.skew = skew
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        alpha_gt = np.zeros((1, self.dim))
        alpha_gt[0][0] = self.skew
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=SkewedGaussian(alpha=alpha_gt),
        )

        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_kl_divergence(self, num_values: int = 1000) -> float:
        # This is quite slow, due to the sampling from the Skewed distribution being slow.
        # A Monte-Carlo approximation due to not having a closed-form formula for either cases (normal and inverse).
        dist_gt, dist_fcst = self.get_distributions()
        return estimate_kl(dist_gt, dist_fcst, num_values)


class MissingSkewAll(DistributionPair):
    """
    The ground-truth contains a independent skewed normal distributions.
    The forecast only has independant gaussian distributions.
    """

    def __init__(self, dim: int, skew: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.skew = skew
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        alpha_gt = self.skew * np.ones((1, self.dim))
        dist_gt = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=SkewedGaussian(alpha=alpha_gt),
        )

        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_kl_divergence(self, num_values: int = 1000) -> float:
        # This is quite slow, due to the sampling from the Skewed distribution being slow.
        # A Monte-Carlo approximation due to not having a closed-form formula for either cases (normal and inverse).
        dist_gt, dist_fcst = self.get_distributions()
        return estimate_kl(dist_gt, dist_fcst, num_values)


class MissingSkewSingleFast(DistributionPair):
    """
    The ground-truth contains a single skewed normal distribution,
    and (dim-1) independant gaussian distributions.
    The forecast only has independant gaussian distributions.
    """

    def __init__(self, dim: int, skew: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.skew = skew
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        alpha_gt = np.zeros(self.dim)
        alpha_gt[0] = self.skew
        dist_gt = IndependentSkewedGaussianDistribution(
            num_time=1, num_series=self.dim, alpha=alpha_gt
        )
        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_kl_divergence(self, num_values: int = 1000) -> float:
        # This is quite slow, due to the sampling from the Skewed distribution being slow.
        # A Monte-Carlo approximation due to not having a closed-form formula for either cases (normal and inverse).
        dist_gt, dist_fcst = self.get_distributions()
        return estimate_kl(dist_gt, dist_fcst, num_values)


class MissingSkewAllFast(DistributionPair):
    """
    The ground-truth contains a independent skewed normal distributions.
    The forecast only has independant gaussian distributions.
    """

    def __init__(self, dim: int, skew: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.skew = skew
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        alpha_gt = self.skew * np.ones(self.dim)
        dist_gt = IndependentSkewedGaussianDistribution(
            num_time=1, num_series=self.dim, alpha=alpha_gt
        )
        dist_fcst = CombinedDistribution(
            copula=TrivialCopula(num_time=1, num_series=self.dim),
            marginal=N01Marginal(),
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_kl_divergence(self, num_values: int = 1000) -> float:
        # This is quite slow, due to the sampling from the Skewed distribution being slow.
        # A Monte-Carlo approximation due to not having a closed-form formula for either cases (normal and inverse).
        dist_gt, dist_fcst = self.get_distributions()
        return estimate_kl(dist_gt, dist_fcst, num_values)


######################################################
# Distributions varying by their covariance matrices #
######################################################


class MissingCovarianceFull(DistributionPair):
    """
    Both distributions are N(0,sigma) multivariate gaussians.
    The non-diagonal element of the covariance matrix of the ground truth is a constant.
    The covariance matrix of the forecast is the identity matrix.
    """

    def __init__(self, dim: int, corr: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.corr_gt = corr
        self.corr_fcst = 0
        if inverse:
            self.corr_gt, self.corr_fcst = self.corr_fcst, self.corr_gt

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        cov_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(cov_gt, 1)
        dist_gt = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_gt),
            marginal=N01Marginal(),
        )

        cov_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(cov_fcst, 1)
        dist_fcst = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_fcst),
            marginal=N01Marginal(),
        )

        return dist_gt, dist_fcst

    def get_gaussian_parameters(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # GT mu, GT Sigma, FCST mu, FCST Sigma
        mu_gt = np.zeros(self.dim)
        sigma_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(sigma_gt, 1)
        mu_fcst = np.zeros(self.dim)
        sigma_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(sigma_fcst, 1)
        return mu_gt, sigma_gt, mu_fcst, sigma_fcst

    def get_kl_divergence(self) -> float:
        cov_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(cov_gt, 1)
        cov_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(cov_fcst, 1)
        return gaussian_kl_divergence(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )

    def get_wasserstein_distance(self) -> float:
        cov_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(cov_gt, 1)
        cov_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(cov_fcst, 1)
        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )


class MissingCovarianceCheckerBoard(DistributionPair):
    """
    Both distributions are N(0,sigma) multivariate gaussians.
    The non-diagonal element of the covariance matrix of the ground truth is a constant up to a sign.
    That sign is + if both the row and column have the same parity, and - otherwise.
    The covariance matrix of the forecast is the identity matrix.
    """

    def __init__(self, dim: int, corr: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.corr_gt = corr
        self.corr_fcst = 0
        if inverse:
            self.corr_gt, self.corr_fcst = self.corr_fcst, self.corr_gt

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        cov_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(cov_gt, 1)
        cov_gt[::2, :] = -cov_gt[::2, :]
        cov_gt[:, ::2] = -cov_gt[:, ::2]
        dist_gt = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_gt),
            marginal=N01Marginal(),
        )

        cov_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(cov_fcst, 1)
        cov_fcst[::2, :] = -cov_fcst[::2, :]
        cov_fcst[:, ::2] = -cov_fcst[:, ::2]
        dist_fcst = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_fcst),
            marginal=N01Marginal(),
        )

        return dist_gt, dist_fcst

    def get_gaussian_parameters(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # GT mu, GT Sigma, FCST mu, FCST Sigma
        mu_gt = np.zeros(self.dim)
        sigma_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(sigma_gt, 1)
        sigma_gt[::2, :] = -sigma_gt[::2, :]
        sigma_gt[:, ::2] = -sigma_gt[:, ::2]

        mu_fcst = np.zeros(self.dim)
        sigma_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(sigma_fcst, 1)
        sigma_fcst[::2, :] = -sigma_fcst[::2, :]
        sigma_fcst[:, ::2] = -sigma_fcst[:, ::2]
        return mu_gt, sigma_gt, mu_fcst, sigma_fcst

    def get_kl_divergence(self) -> float:
        cov_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(cov_gt, 1)
        cov_gt[::2, :] = -cov_gt[::2, :]
        cov_gt[:, ::2] = -cov_gt[:, ::2]
        cov_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(cov_fcst, 1)
        cov_fcst[::2, :] = -cov_fcst[::2, :]
        cov_fcst[:, ::2] = -cov_fcst[:, ::2]
        return gaussian_kl_divergence(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )

    def get_wasserstein_distance(self) -> float:
        cov_gt = np.ones((self.dim, self.dim)) * self.corr_gt
        np.fill_diagonal(cov_gt, 1)
        cov_gt[::2, :] = -cov_gt[::2, :]
        cov_gt[:, ::2] = -cov_gt[:, ::2]
        cov_fcst = np.ones((self.dim, self.dim)) * self.corr_fcst
        np.fill_diagonal(cov_fcst, 1)
        cov_fcst[::2, :] = -cov_fcst[::2, :]
        cov_fcst[:, ::2] = -cov_fcst[:, ::2]
        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )


class MissingCovarianceBlockDiag(DistributionPair):
    """
    Both distributions are N(0,sigma) multivariate gaussians.
    The covariance matrix of the ground truth is block diagonal matrix with 2-by-2 blocks.
    The covariance matrix of the forecast is the identity matrix.
    """

    def __init__(self, dim: int, corr: float, inverse: bool = False) -> None:
        super().__init__()
        assert (
            dim % 2 == 0
        ), f"dim ({dim}) for MissingCovarianceBlockDiag must be a multiple of 2"
        self.dim = dim
        self.gt_block = np.array(
            [
                [1, corr],
                [corr, 1],
            ]
        )
        self.fcst_block = np.array(
            [
                [1, 0],
                [0, 1],
            ]
        )
        if inverse:
            self.gt_block, self.fcst_block = self.fcst_block, self.gt_block

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        cov_gt = np.block(
            [
                [
                    self.gt_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )
        dist_gt = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_gt),
            marginal=N01Marginal(),
        )

        cov_fcst = np.block(
            [
                [
                    self.fcst_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )
        dist_fcst = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_fcst),
            marginal=N01Marginal(),
        )

        return dist_gt, dist_fcst

    def get_gaussian_parameters(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # GT mu, GT Sigma, FCST mu, FCST Sigma
        mu_gt = np.zeros(self.dim)
        sigma_gt = np.block(
            [
                [
                    self.gt_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )

        mu_fcst = np.zeros(self.dim)
        sigma_fcst = np.block(
            [
                [
                    self.fcst_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )

        return mu_gt, sigma_gt, mu_fcst, sigma_fcst

    def get_kl_divergence(self) -> float:
        cov_gt = np.block(
            [
                [
                    self.gt_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )
        cov_fcst = np.block(
            [
                [
                    self.fcst_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )
        return gaussian_kl_divergence(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )

    def get_wasserstein_distance(self) -> float:
        cov_gt = np.block(
            [
                [
                    self.gt_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )
        cov_fcst = np.block(
            [
                [
                    self.fcst_block if i == j else np.zeros((2, 2))
                    for i in range(self.dim // 2)
                ]
                for j in range(self.dim // 2)
            ]
        )
        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )


class TooSmallCovBlockDiag(DistributionPair):
    """
    Both distributions are N(0,sigma) multivariate gaussians.
    The covariance is block diagonal for both distribution.
    The ground-truth distribution blocks are 4-by-4, with two correlations values
    (one for the diagonal 2-by-2 blocks, and one for the outer 2-by-2 blocks).
    The forecast distribution only has the diagonal 2-by-2 blocks.
    """

    def __init__(
        self, dim: int, inner_corr: float, outer_corr: float, inverse: bool = False
    ) -> None:
        super().__init__()
        assert (
            dim % 4 == 0
        ), f"dim ({dim}) for TooSmallCovBlockDiag must be a multiple of 4"
        self.dim = dim
        self.gt_block = np.array(
            [
                [1, inner_corr, outer_corr, outer_corr],
                [inner_corr, 1, outer_corr, outer_corr],
                [outer_corr, outer_corr, 1, inner_corr],
                [outer_corr, outer_corr, inner_corr, 1],
            ]
        )
        self.fcst_block = np.array(
            [
                [1, inner_corr, 0, 0],
                [inner_corr, 1, 0, 0],
                [0, 0, 1, inner_corr],
                [0, 0, inner_corr, 1],
            ]
        )
        if inverse:
            self.gt_block, self.fcst_block = self.fcst_block, self.gt_block

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        cov_gt = np.block(
            [
                [
                    self.gt_block if i == j else np.zeros((4, 4))
                    for i in range(self.dim // 4)
                ]
                for j in range(self.dim // 4)
            ]
        )
        dist_gt = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_gt),
            marginal=N01Marginal(),
        )

        cov_fcst = np.block(
            [
                [
                    self.fcst_block if i == j else np.zeros((4, 4))
                    for i in range(self.dim // 4)
                ]
                for j in range(self.dim // 4)
            ]
        )
        dist_fcst = CombinedDistribution(
            copula=CustomGaussianCopula(num_time=1, num_series=self.dim, cov=cov_fcst),
            marginal=N01Marginal(),
        )

        return dist_gt, dist_fcst

    def get_kl_divergence(self) -> float:
        cov_gt = np.block(
            [
                [
                    self.gt_block if i == j else np.zeros((4, 4))
                    for i in range(self.dim // 4)
                ]
                for j in range(self.dim // 4)
            ]
        )
        cov_fcst = np.block(
            [
                [
                    self.fcst_block if i == j else np.zeros((4, 4))
                    for i in range(self.dim // 4)
                ]
                for j in range(self.dim // 4)
            ]
        )
        return gaussian_kl_divergence(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )

    def get_wasserstein_distance(self) -> float:
        cov_gt = np.block(
            [
                [
                    self.gt_block if i == j else np.zeros((4, 4))
                    for i in range(self.dim // 4)
                ]
                for j in range(self.dim // 4)
            ]
        )
        cov_fcst = np.block(
            [
                [
                    self.fcst_block if i == j else np.zeros((4, 4))
                    for i in range(self.dim // 4)
                ]
                for j in range(self.dim // 4)
            ]
        )
        return gaussian_wasserstein_distance(
            np.zeros(self.dim),
            cov_gt,
            np.zeros(self.dim),
            cov_fcst,
        )


###########################################
# Distributions varying by their mixtures #
###########################################


class MissingMixture(DistributionPair):
    """
    The ground-truth is a mixture of a N(mu,1) and a N(-mu,1) multivariate Gaussians,
    with mu being equal in all dimensions.
    The forecast is a N(0,sigma) multivariate Gaussian, with the covariance matrix
    chosen to have the same covariance as the ground-truth distribution.
    """

    def __init__(self, dim: int, distance: float, inverse: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.distance = distance
        self.inverse = inverse

    def get_distributions(self) -> Tuple[Distribution, Distribution]:
        mean_gt_1 = self.distance * np.ones(self.dim)
        mean_gt_2 = -self.distance * np.ones(self.dim)
        cov_gt = np.eye(self.dim)

        dist_gt = MixtureDistribution(
            dist1=GaussianDistribution(
                num_time=1, num_series=self.dim, mean=mean_gt_1, cov=cov_gt
            ),
            dist2=GaussianDistribution(
                num_time=1, num_series=self.dim, mean=mean_gt_2, cov=cov_gt
            ),
            ratio=0.5,
        )

        # A Gaussian with mean 0 and the same correlations as the mixture
        mean_fcst = np.zeros(self.dim)
        cov_fcst = np.eye(self.dim) + self.distance**2 * np.ones((self.dim, self.dim))
        dist_fcst = GaussianDistribution(
            num_time=1, num_series=self.dim, mean=mean_fcst, cov=cov_fcst
        )

        if not self.inverse:
            return dist_gt, dist_fcst
        else:
            return dist_fcst, dist_gt

    def get_kl_divergence(self, num_values: int = 10000) -> float:
        # A Monte-Carlo approximation due to not having a closed-form formula for both cases (normal and inverse).
        dist_gt, dist_fcst = self.get_distributions()
        return estimate_kl(dist_gt, dist_fcst, num_values)

    def get_wasserstein_distance(self) -> float:
        # The Wasserstein distance does not depend on whether it is the inverse or not.
        # Also, this is a rough approximation due to not having a closed-form formula for Gaussian mixtures.
        mean_gt_1 = self.distance * np.ones(self.dim)
        mean_gt_2 = -self.distance * np.ones(self.dim)
        cov_gt = np.eye(self.dim)
        mean_fcst = np.zeros(self.dim)
        cov_fcst = np.eye(self.dim) + self.distance**2 * np.ones((self.dim, self.dim))

        return gaussian_mixture_wasserstein_distance(
            weights1=[0.5, 0.5],
            weights2=[1],
            mu1=[mean_gt_1, mean_gt_2],
            mu2=[mean_fcst],
            cov1=[cov_gt, cov_gt],
            cov2=[cov_fcst],
        )
