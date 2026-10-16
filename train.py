"""Train, evaluate and export NextFix end to end.   Usage:  python train.py

Pipeline (all vectorised; whole run takes a few seconds):
  1. load + validate the three dataset CSVs (src/nextfix/data.py)
  2. build one feature row per location (src/nextfix/features.py)
  3. evaluate baselines vs. models on a leak-free time split (src/nextfix/evaluate.py)
  4. fit the final Gamma-Poisson empirical-Bayes model (src/nextfix/model.py, from scratch)
  5. score locations, write reports/, models/ and the web app data (app/data.js)
No external APIs, hosted models or pretrained weights are used.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from nextfix import data, evaluate, features, model, scoring  # noqa: E402

CUT = "2025-10-01"      # time split: learn from Jan-Sep, predict Oct-Dec
HORIZON_DAYS = 90       # forecast window shown in the app
timings = {}


def tick(name, t0):
    timings[name] = round(time.perf_counter() - t0, 3)


def main():
    T = time.perf_counter()
    t0 = time.perf_counter(); L, I, C = data.load(); tick("load_validate", t0)
    t0 = time.perf_counter(); frame, tensor, types = features.location_frame(L, I, C); tick("features", t0)

    # ---- evaluation on a leak-free time split --------------------------------------------
    t0 = time.perf_counter()
    mf, y, e_obs, e_fut = features.model_frame(L, I, C, CUT)
    cv = evaluate.cv_compare(mf, y, e_obs, e_fut)
    reliability = evaluate.split_half_reliability(I, L)
    tick("cross_validation", t0)

    # ---- final model: fit on the whole year, forecast the next 90 days ---------------------
    t0 = time.perf_counter()
    days = float((I.timestamp.max().normalize() + pd.Timedelta(days=1) - I.timestamp.min().normalize()).days)
    hc = frame.hc.to_numpy(float)
    eb = model.GammaPoissonEB().fit(hc, np.full(len(hc), days))
    frame["fc"] = eb.predict(hc, days, HORIZON_DAYS)
    lo, hi = eb.interval(hc, days, HORIZON_DAYS)
    frame["fl"], frame["fh"] = lo, hi
    data_weight = float(eb.data_weight(days))
    tick("final_fit", t0)

    # ---- reports ---------------------------------------------------------------------------
    disp = float(frame.n.var() / frame.n.mean())
    similar = reliability < 0.1 or data_weight < 0.2 or eb.alpha_ > 500
    msg = ("The model finds High/Critical rates statistically similar across locations "
           f"(split-half rank correlation {reliability:.2f}; incident dispersion {disp:.2f}). "
           "Treat the ranking as a screening aid and verify on site.") if similar else \
          ("The model finds real differences between locations "
           f"(split-half rank correlation {reliability:.2f}); the forecast is informative.")
    metrics = {
        "data": {"locations": len(L), "incidents": len(I), "complaints": len(C), "days": days},
        "split": {"cut": CUT, "train_days": e_obs, "test_days": e_fut, "folds": "5-fold x 3 repeats over locations"},
        "noise_diagnostics": {"incident_dispersion_index": disp, "split_half_spearman_hc": reliability,
                              "eb_alpha": eb.alpha_, "eb_weight_on_own_data": data_weight},
        "cv": cv, "final_model": eb.to_dict(), "interpretation": msg, "runtime_seconds": timings,
    }
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "models" / "eb_model.json").write_text(json.dumps(eb.to_dict(), indent=2))
    weights = {**scoring.DEFAULT_WEIGHTS, "fc": 0 if similar else scoring.DEFAULT_WEIGHTS["fc"]}  # no weight on a signal-free forecast
    review = scoring.review_list(frame, frame.fc, top=50, weights=weights)
    review.to_csv(ROOT / "reports" / "review_list_top50.csv", index=False)
    write_results_md(metrics)

    # ---- web app data ----------------------------------------------------------------------
    t0 = time.perf_counter(); export_app(L, I, frame, tensor, types, msg, weights['fc']); tick("export_app", t0)
    timings["total"] = round(time.perf_counter() - T, 3)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done in {timings['total']}s")
    print(msg)


def write_results_md(m):
    names = {"global_mean": "Baseline: global mean", "repeat_last_period": "Baseline: repeat last period",
             "gamma_poisson_eb": "Gamma-Poisson EB (ours, from scratch)", "poisson_glm": "Poisson regression (scikit-learn)",
             "grad_boosting": "Gradient boosting, Poisson loss (scikit-learn)"}
    lines = ["| Model | MAE (lower better) | Poisson deviance (lower better) | Spearman | Top-20% lift (1.0 = random) |",
             "|---|---|---|---|---|"]
    for k, v in m["cv"].items():
        f = lambda s: f"{v[s]['mean']:.3f} ± {v[s]['sd']:.3f}"
        lines.append(f"| {names[k]} | {f('mae')} | {f('poisson_deviance')} | {f('spearman')} | {f('top20_lift')} |")
    nd = m["noise_diagnostics"]
    text = ("# Model results (auto-generated by train.py)\n\n"
            f"Target: High/Critical incidents per location, {m['split']['cut']} to year end. "
            f"Features and priors use data before the cut. {m['split']['folds']}.\n\n" + "\n".join(lines) +
            f"\n\nNoise diagnostics: incident dispersion index {nd['incident_dispersion_index']:.2f} (1.0 = pure Poisson), "
            f"split-half Spearman of High/Critical counts {nd['split_half_spearman_hc']:.3f}, "
            f"EB weight on a location's own data {nd['eb_weight_on_own_data']:.2f}.\n\n"
            f"**Interpretation:** {m['interpretation']}\n")
    (ROOT / "reports" / "RESULTS.md").write_text(text)


def export_app(L, I, f, tensor, types, msg, fcw):
    rows = []
    for i, r in enumerate(f.itertuples()):
        rows.append(dict(id=r.location_id, z=r.zone.replace("ZONE_", "Z"), rt=r.road_type, rc=r.road_condition,
                         pd=int(r.pd), sc=int(r.sc), ho=int(r.ho), pa=r.pedestrian_activity,
                         la=round(r.latitude, 5), lo=round(r.longitude, 5), t=tensor[i].tolist(), ni=int(r.n),
                         cn=int(r.cn), cw=float(r.cw), hp=int(r.hp), oc=int(r.oc),
                         fc=round(float(r.fc), 3), fl=int(r.fl), fh=int(r.fh)))
    zone = I.merge(L[["location_id", "zone"]], on="location_id")
    zone["z"] = zone.zone.str.replace("ZONE_", "Z"); zone["m"] = zone.timestamp.dt.month
    zm = zone.groupby(["z", "m"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    ZM = {z: row.tolist() for z, row in zm.iterrows()}
    ZM["ALL"] = zm.sum().tolist()
    js = ("const TYPES=" + json.dumps(types) + ";\nconst META=" + json.dumps({"msg": msg, "fcw": fcw}) + ";\nconst ZM=" +
          json.dumps(ZM, separators=(",", ":")) + ";\nconst DATA=" + json.dumps(rows, separators=(",", ":")) + ";\n")
    app = ROOT / "app"
    (app / "data.js").write_text(js)
    html = (app / "index.html").read_text()  # single-file copy that works even if files are moved
    html = html.replace('<link rel="stylesheet" href="style.css">', "<style>\n" + (app / "style.css").read_text() + "\n</style>")
    html = html.replace('<script src="data.js"></script><script src="app.js"></script>',
                        "<script>\n" + js + (app / "app.js").read_text() + "\n</script>")
    (app / "NextFix_standalone.html").write_text(html)


if __name__ == "__main__":
    main()
