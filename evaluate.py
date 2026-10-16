"""Honest model evaluation: repeated K-fold over locations on a time-based split."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURES
from .model import GammaPoissonEB

EPS = 1e-3


def _metrics(y, p):
    p = np.clip(p, EPS, None)
    top = np.argsort(-p)[: max(1, int(0.2 * len(y)))]
    return {
        "mae": mean_absolute_error(y, p),
        "poisson_deviance": mean_poisson_deviance(y, p),
        "spearman": spearmanr(p, y)[0] if np.ptp(p) > 0 else 0.0,
        "top20_lift": (y[top].sum() / max(y.sum(), 1)) / 0.2,  # 1.0 = no better than random
    }


def cv_compare(frame, y, e_obs, e_fut, n_splits=5, repeats=3, seed=42):
    """Compare baselines and models. Each fold fits on train locations, scores on unseen ones."""
    X, yv = frame[MODEL_FEATURES].to_numpy(float), np.asarray(y, float)
    hc_before = frame.hc.to_numpy(float)
    scale = e_fut / e_obs
    rows = {k: [] for k in ["global_mean", "repeat_last_period", "gamma_poisson_eb", "poisson_glm", "grad_boosting"]}
    for tr, te in RepeatedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed).split(X):
        rows["global_mean"].append(_metrics(yv[te], np.full(len(te), yv[tr].mean())))
        rows["repeat_last_period"].append(_metrics(yv[te], hc_before[te] * scale))
        eb = GammaPoissonEB().fit(hc_before[tr], np.full(len(tr), e_obs))
        rows["gamma_poisson_eb"].append(_metrics(yv[te], eb.predict(hc_before[te], e_obs, e_fut)))
        sc = StandardScaler().fit(X[tr])
        glm = PoissonRegressor(alpha=1.0, max_iter=500).fit(sc.transform(X[tr]), yv[tr])
        rows["poisson_glm"].append(_metrics(yv[te], glm.predict(sc.transform(X[te]))))
        gb = HistGradientBoostingRegressor(loss="poisson", max_depth=3, max_iter=80, learning_rate=0.05, random_state=seed)
        rows["grad_boosting"].append(_metrics(yv[te], gb.fit(X[tr], yv[tr]).predict(X[te])))
    out = {}
    base = np.array([m["poisson_deviance"] for m in rows["global_mean"]])
    for name, ms in rows.items():
        arr = {k: np.array([m[k] for m in ms]) for k in ms[0]}
        out[name] = {k: {"mean": float(v.mean()), "sd": float(v.std())} for k, v in arr.items()}
        out[name]["folds_better_than_global_mean"] = float((arr["poisson_deviance"] < base).mean())
    return out


def split_half_reliability(I, L):
    """Spearman correlation of per-location High/Critical counts between the two halves of the year.
    ~0 means a location's past does not predict its future (rankings are mostly noise)."""
    mid = I.timestamp.min() + (I.timestamp.max() - I.timestamp.min()) / 2
    hc = I[I.severity.isin(["High", "Critical"])]
    a = hc[hc.timestamp < mid].groupby("location_id").size().reindex(L.location_id, fill_value=0)
    b = hc[hc.timestamp >= mid].groupby("location_id").size().reindex(L.location_id, fill_value=0)
    return float(spearmanr(a, b)[0])
