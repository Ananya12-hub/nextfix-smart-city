# NextFix: Where Should We Fix First?

**Team Data Duo · DATATHON 2026 · Smart City track · Round 2 (complete prototype)**

NextFix turns a city's incident and complaint history into a ranked weekly work queue in which **every rank explains itself**, shows how far to trust it, and can be compared against alternative interventions.

## Problem and target user

- **Problem:** cities receive thousands of reports (potholes, flooding, streetlights, drains, signals, collisions) but crews and budgets are limited. Priorities are usually set by the loudest complaint or gut feeling, with no reason a manager can check.
- **Target user:** a municipal operations coordinator who has to decide which locations get crews this week. Also useful to road/traffic departments and civic-service coordinators.
- **The one workflow we fix:** "Which locations should be reviewed first, based on what evidence, and what information is still missing?"

## Quick start (about 10 seconds)

```bash
pip install -r requirements.txt
python train.py                 # loads data, evaluates models, fits the final model, rewrites app/data.js
python -m pytest -q             # 7 tests: data integrity, no leakage, scoring bounds, model correctness
# then open app/index.html in a browser (or app/NextFix_standalone.html, a single file with everything inline)
```

`app/data.js` is already committed, so you can open `app/index.html` without running anything. No server, no API keys.

## What is in the repo

```
train.py                    <- TRAINING CODE: end-to-end pipeline (load -> features -> evaluate -> fit -> export)
src/nextfix/data.py         loading + integrity checks (unique IDs, no orphan location_id)
src/nextfix/features.py     one feature row per location (single groupby per source table, vectorised)
src/nextfix/model.py        Gamma-Poisson empirical-Bayes model, written from scratch (NumPy/SciPy)
src/nextfix/evaluate.py     leak-free time split, repeated K-fold, baselines
src/nextfix/scoring.py      transparent priority score, reasons, confidence
app/                        the product: HTML + CSS + JavaScript (index.html, style.css, app.js, data.js)
data/                       the three provided dataset files (locations, incidents, public_complaints)
models/eb_model.json        fitted model parameters
reports/                    metrics.json, RESULTS.md, review_list_top50.csv
tests/                      pytest suite
docs/screenshots/           screenshots of the product
```

## Dataset files used

| File | Rows | Used for |
|---|---|---|
| `data/locations.csv` | 1,000 | context: zone, road type/condition, density, school/hospital, pedestrian activity, coordinates |
| `data/incidents.csv` | 10,000 | severity-weighted incidents, injuries, unresolved items, monthly trend, model target |
| `data/public_complaints.csv` | 8,000 | complaint pressure (priority-weighted), open backlog |

All three are joined on `location_id`. Checks in `data.py`: unique IDs, no orphan IDs, no missing values. `resolution_hours` and `days_to_resolution` are **not used** because unresolved/open records also carry values, which makes them unreliable. No extra datasets were used.

## The model

- **What it predicts:** the number of High/Critical incidents a location will have in a future window (the app shows the next 90 days with a likely range).
- **Algorithm:** a **Gamma-Poisson empirical-Bayes model, implemented from scratch** in `src/nextfix/model.py`. Each location has an unknown true rate; the prior is learned across all locations by maximum marginal likelihood (negative-binomial likelihood), and each location's estimate is shrunk toward the city average in proportion to how little data it has. It also gives predictive ranges and a measure of how much real variation exists between locations.
- **Trained from scratch:** yes, on the provided data only. **No API, hosted model or pretrained weights.** scikit-learn is used only for the comparison baselines (Poisson regression, gradient boosting) and metrics.
- **Validation:** time-based split (features from before 1 Oct 2025, target after), 5-fold x 3 repeats over locations. Baselines: global mean and repeat-last-period.

| Model | MAE (lower better) | Poisson deviance (lower better) | Spearman | Top-20% lift (1.0 = random) |
|---|---|---|---|---|
| Baseline: global mean | 0.674 ± 0.011 | 1.084 ± 0.027 | 0.000 ± 0.000 | 1.033 ± 0.161 |
| Baseline: repeat last period | 0.709 ± 0.021 | 2.173 ± 0.270 | 0.006 ± 0.055 | 0.975 ± 0.141 |
| Gamma-Poisson EB (ours, from scratch) | 0.675 ± 0.012 | 1.082 ± 0.027 | 0.006 ± 0.055 | 0.975 ± 0.141 |
| Poisson regression (scikit-learn) | 0.673 ± 0.010 | 1.084 ± 0.027 | 0.030 ± 0.049 | 1.111 ± 0.124 |
| Gradient boosting, Poisson loss (scikit-learn) | 0.669 ± 0.012 | 1.099 ± 0.038 | 0.047 ± 0.064 | 1.085 ± 0.157 |

Noise diagnostics: incident dispersion index 1.06 (1.0 means pure Poisson noise); split-half Spearman correlation of High/Critical counts -0.040; fitted prior is at its upper bound (alpha about 10000), so the model puts 0% weight on a location's own history.

### Honest reading of the results

The provided data is synthetic and, in our analysis, **contains no learnable location-level pattern**: location context (road condition, school, density) does not change incident counts, and a location's past High/Critical counts do not predict its future (split-half correlation about 0). Consequently **no model, including ours, beats the simple baselines**, and we do not claim otherwise. What the from-scratch model does correctly is *detect* this: it shrinks every location to the city average, reports wide predictive ranges, and the app shows a "Model check" banner telling users the ranking is a screening aid to verify on site. On data with real structure the same code will learn a small alpha, weight local history more, and the forecast will enter the ranking (the forecast weight is set to 0 only when the model finds no signal). A model that claimed accuracy here would be fitting noise.

## Priority score (transparent, adjustable)

`score = 100 * sum(weight_k * component_k) / sum(weight_k)`, each component scaled 0 to 1.

| Component | Default weight | Raw value |
|---|---|---|
| Incident severity | 25 | Low=1, Medium=2, High=4, Critical=8, summed |
| Injuries | 20 | total injuries |
| Unresolved backlog | 20 | unresolved incidents + open/assigned/in-progress complaints |
| Complaint pressure | 15 | High=2, Normal=1, Low=0.5, summed |
| Vulnerability | 15 | school, hospital, pedestrian activity, density, road condition |
| Recent activity | 5 | severity-weighted incidents in the last 90 days |
| Forecast (next 90 days) | 0 or 10 | model forecast; 0 when the model finds no signal |

Every rank shows its top 3 drivers, a confidence level (from the number of records), and information gaps. Sliders change the weights live.

## Product features (all in `app/`)

- Ranked priority queue with reasons, confidence and information gaps; map; location card with score breakdown and forecast range
- Filters by zone and issue type; adjustable weights
- **Incident trend monitor** per zone with spike detection (months above mean + 2 standard deviations)
- **Fairness check:** each zone's share of the top 100 compared with its share of city population density
- **Intervention comparison:** for the top N locations, which issue type to fix addresses the most High/Critical incidents, injuries and backlog
- **CSV export** of the weekly review list
- Model-check banner (see above)

## Efficiency

Everything is vectorised pandas/NumPy with one groupby pass per source table. Full pipeline (load, validate, features, 15-fold evaluation of 5 models, final fit, export) runs in **0.941 s** on a laptop CPU (cross-validation 0.806 s, final fit 0.004 s). The app is static files, computes rankings in the browser in milliseconds, and needs no server.

## APIs, pretrained models and keys

- **APIs used: none.** No LLM or hosted model. No pretrained model or extra dataset.
- **No API keys exist in this repo**; `.env` files are git-ignored.

## Screenshots

![Dashboard](docs/screenshots/01_dashboard.png)
![Priority queue and trend monitor](docs/screenshots/02_priority_queue_trend.png)
![Fairness check and interventions](docs/screenshots/03_fairness_interventions.png)

## Limitations and responsible use

Synthetic data and simulated coordinates; not for real public-safety decisions. NextFix is a decision aid: weights are visible, a human approves the queue, and the fairness view helps spot zones that are over- or under-represented. Population density is used as a proxy because no population column is provided.
