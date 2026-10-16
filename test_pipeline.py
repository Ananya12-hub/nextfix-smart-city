import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nextfix import data, features, scoring  # noqa: E402
from nextfix.model import GammaPoissonEB  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return data.load()


def test_join_key_and_sizes(tables):
    L, I, C = tables
    assert (len(L), len(I), len(C)) == (1000, 10000, 8000)


def test_validate_rejects_orphan_ids(tables):
    L, I, C = tables
    bad = I.copy(); bad.loc[0, "location_id"] = "NOPE"
    with pytest.raises(AssertionError):
        data.validate(L, bad, C)


def test_feature_tensor_matches_raw_counts(tables):
    L, I, C = tables
    f, tensor, types = features.location_frame(L, I, C)
    assert len(f) == len(L) and tensor.shape == (1000, len(types), len(features.METRICS))
    assert tensor[:, :, 0].sum() == len(I)                      # every incident counted once
    assert f.cn.sum() == len(C)                                  # every complaint counted once
    assert tensor[:, :, 1].sum() == I.severity.isin(["High", "Critical"]).sum()


def test_model_frame_has_no_time_leakage(tables):
    L, I, C = tables
    cut = pd.Timestamp("2025-10-01")
    f, y, e_obs, e_fut = features.model_frame(L, I, C, cut)
    assert f.n.sum() == (I.timestamp < cut).sum()                # features use only pre-cut incidents
    assert y.sum() == ((I.timestamp >= cut) & I.severity.isin(["High", "Critical"])).sum()


def test_priority_scores_bounded_and_ordered(tables):
    L, I, C = tables
    f, _, _ = features.location_frame(L, I, C)
    score, c = scoring.priority(f, np.zeros(len(f)))
    assert score.between(0, 100).all() and c.max().max() <= 1.0
    rl = scoring.review_list(f, np.zeros(len(f)), top=10)
    assert rl.priority.is_monotonic_decreasing and len(rl) == 10


def test_eb_recovers_known_prior():
    rng = np.random.default_rng(0)
    alpha, beta, e = 4.0, 400.0, 500.0                           # true prior mean rate = 0.01/day
    lam = rng.gamma(alpha, 1 / beta, 5000)
    y = rng.poisson(lam * e)
    m = GammaPoissonEB().fit(y, np.full(5000, e))
    assert abs(m.alpha_ - alpha) / alpha < 0.25


def test_eb_shrinks_more_with_less_data():
    m = GammaPoissonEB(); m.alpha_, m.beta_, m.rate_ = 4.0, 400.0, 0.01
    assert m.data_weight(10) < m.data_weight(1000)
    assert m.rate(5, 10) < m.rate(500, 1000) * 5                 # small sample pulled toward the mean
