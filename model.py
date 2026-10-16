"""Gamma-Poisson empirical-Bayes model, implemented from scratch with NumPy/SciPy.

Why this model: incident counts per location are small, so raw counts are noisy.
We assume each location has a true High/Critical rate  lambda_i ~ Gamma(alpha, beta)
and observe y_i ~ Poisson(lambda_i * exposure_i). The prior is learned from all
locations by maximum marginal likelihood (a negative-binomial likelihood), then each
location's rate is shrunk toward the city average in proportion to how little data it has.

Outputs: calibrated expected counts, predictive ranges, and a measure of how much
real between-location variation exists (alpha small => real differences, alpha large =>
locations are statistically similar and rankings are mostly noise).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom


class GammaPoissonEB:
    def __init__(self, max_alpha=1e4):
        self.max_alpha = max_alpha

    def fit(self, y, exposure):
        y, e = np.asarray(y, float), np.asarray(exposure, float)
        self.rate_ = y.sum() / e.sum()  # pooled rate = prior mean (fixes beta given alpha)

        def nll(log_a):
            a = np.exp(log_a)
            b = a / self.rate_
            return -(gammaln(a + y) - gammaln(a) - gammaln(y + 1)
                     + a * np.log(b / (b + e)) + y * np.log(e / (b + e))).sum()

        res = minimize_scalar(nll, bounds=(-4, np.log(self.max_alpha)), method="bounded")
        self.alpha_ = float(np.exp(res.x))
        self.beta_ = self.alpha_ / self.rate_
        return self

    def rate(self, y, e_obs):
        """Posterior mean rate per unit exposure."""
        return (self.alpha_ + np.asarray(y, float)) / (self.beta_ + np.asarray(e_obs, float))

    def predict(self, y, e_obs, e_future):
        return self.rate(y, e_obs) * e_future

    def data_weight(self, e_obs):
        """Share of the estimate that comes from the location's own data (0..1)."""
        e = np.asarray(e_obs, float)
        return e / (self.beta_ + e)

    def interval(self, y, e_obs, e_future, level=0.8):
        """Predictive range for the COUNT in the future window (negative binomial)."""
        n = self.alpha_ + np.asarray(y, float)
        p = (self.beta_ + e_obs) / (self.beta_ + e_obs + e_future)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return nbinom.ppf(lo, n, p), nbinom.ppf(hi, n, p)

    def to_dict(self):
        return {"alpha": self.alpha_, "beta": self.beta_, "prior_rate_per_day": self.rate_}
