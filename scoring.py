"""Transparent priority score, reasons and confidence (mirrors app/app.js)."""
import numpy as np
import pandas as pd

COMPONENTS = [("sev", "Incident severity", 25), ("inj", "Injuries", 20), ("back", "Unresolved backlog", 20),
              ("comp", "Complaint pressure", 15), ("vul", "Vulnerability", 15),
              ("rec", "Recent activity", 5), ("fc", "Forecast (next 90 days)", 10)]
DEFAULT_WEIGHTS = {k: w for k, _, w in COMPONENTS}


def components(f, fc):
    raw = pd.DataFrame({"sev": f.sw, "inj": f.inj, "back": f.un + f.oc, "comp": f.cw,
                        "vul": f.vul, "rec": f.rec, "fc": fc}, index=f.index).astype(float)
    return raw / raw.max().replace(0, 1)  # each component scaled to 0..1


def priority(f, fc, weights=None):
    w = pd.Series(weights or DEFAULT_WEIGHTS)
    c = components(f, fc)
    return 100 * (c * w[c.columns]).sum(axis=1) / w.sum(), c


def confidence(f):
    evidence = f.n + f.cn
    return np.where(evidence >= 22, "High", np.where(evidence >= 15, "Medium", "Low"))


def reasons(f, c, weights=None, top=3):
    w = pd.Series(weights or DEFAULT_WEIGHTS)
    contrib = (c * w[c.columns]).drop(columns="fc")  # forecast is shown separately, not as a reason
    out = []
    for i in range(len(f)):
        r = f.iloc[i]
        text = {
            "sev": f"{int(r.hc)} High/Critical incidents", "inj": f"{int(r.inj)} injuries",
            "back": f"{int(r.un)} unresolved incidents, {int(r.oc)} open complaints",
            "comp": f"{int(r.cn)} complaints ({int(r.hp)} high priority)",
            "rec": f"{int(r.rec)} incidents in last 90 days", "fc": f"{r.fc:.1f} High/Critical expected next 90 days",
            "vul": ", ".join(s for s, ok in [("near a school", r.sc), ("near a hospital", r.ho),
                                             ("high pedestrian activity", r.pa == 1), ("poor road condition", r.rc >= 0.67)] if ok) or "average surroundings",
        }
        keys = contrib.iloc[i].sort_values(ascending=False).index[:top]
        out.append("; ".join(text[k] for k in keys if contrib.iloc[i][k] > 0))
    return out


def review_list(f, fc, top=50, weights=None):
    f = f.assign(fc=fc)
    score, c = priority(f, fc, weights)
    f = f.assign(priority=score.round(1), confidence=confidence(f))
    order = score.sort_values(ascending=False).index[:top]
    f, c = f.loc[order], c.loc[order]
    f["reasons"] = reasons(f, c, weights)
    f.insert(0, "rank", range(1, len(f) + 1))
    return f[["rank", "location_id", "zone", "priority", "confidence", "reasons", "fc"]].rename(columns={"fc": "forecast_hc_90d"})
