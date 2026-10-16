"""Feature engineering: one row per location, built with a single groupby per source table."""
import numpy as np
import pandas as pd

SEV_W = {"Low": 1, "Medium": 2, "High": 4, "Critical": 8}
PRI_W = {"High": 2.0, "Normal": 1.0, "Low": 0.5}
OPEN_STATUS = ["Open", "Assigned", "In progress"]
RC = {"Good": 0.0, "Fair": 0.33, "Poor": 0.67, "Critical": 1.0}
PA = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
RECENT_DAYS = 90
METRICS = ["n", "hc", "un", "inj", "sw", "rec"]  # per-type columns exported to the app
MODEL_FEATURES = ["n", "hc", "sw", "inj", "un", "cn", "hp", "vul", "rc", "pa", "sc", "ho", "pd"]


def location_frame(L, I, C, as_of=None):
    """Aggregate incidents/complaints per location.

    Returns (frame, tensor, types). `tensor` has shape (locations, types, len(METRICS)).
    """
    I = I.assign(
        w=I.severity.map(SEV_W),
        hc=I.severity.isin(["High", "Critical"]).astype(int),
        un=1 - I.resolved,
    )
    if as_of is None:
        as_of = I.timestamp.max().normalize() + pd.Timedelta(days=1)
    I["rec"] = (I.timestamp >= as_of - pd.Timedelta(days=RECENT_DAYS)).astype(int)
    types = sorted(I.incident_type.unique())

    g = I.groupby(["location_id", "incident_type"]).agg(
        n=("w", "size"), hc=("hc", "sum"), un=("un", "sum"),
        inj=("injuries", "sum"), sw=("w", "sum"), rec=("rec", "sum"),
    ).unstack("incident_type", fill_value=0).reindex(L.location_id, fill_value=0)
    tensor = np.stack([g[m].reindex(columns=types, fill_value=0).to_numpy() for m in METRICS], axis=2).astype(int)

    c = C.assign(
        cw=C.priority_reported.map(PRI_W),
        hp=(C.priority_reported == "High").astype(int),
        oc=C.status.isin(OPEN_STATUS).astype(int),
    ).groupby("location_id").agg(cn=("cw", "size"), cw=("cw", "sum"), hp=("hp", "sum"), oc=("oc", "sum"))

    f = L.set_index("location_id").copy()
    tot = tensor.sum(axis=1)
    for k, m in enumerate(METRICS):
        f[m] = tot[:, k]
    f = f.join(c).fillna({"cn": 0, "cw": 0, "hp": 0, "oc": 0})
    f["rc"] = f.road_condition.map(RC)
    f["pa"] = f.pedestrian_activity.map(PA)
    f["sc"], f["ho"] = f.near_school, f.near_hospital
    f["pd"] = f.population_density_per_km2
    f["vul"] = (0.30 * f.sc + 0.20 * f.ho + 0.25 * f.pa
                + 0.15 * np.minimum(f.pd / 25000, 1) + 0.10 * f.rc)
    return f.reset_index(), tensor, types


def model_frame(L, I, C, cut):
    """Leak-free modelling table: features from data BEFORE `cut`, target AFTER it.

    target y = High/Critical incidents per location after `cut`.
    """
    cut = pd.Timestamp(cut)
    start = I.timestamp.min().normalize()
    end = I.timestamp.max().normalize() + pd.Timedelta(days=1)
    before = I[I.timestamp < cut]
    f, _, _ = location_frame(L, before, C[C.created_at < cut], as_of=cut)
    after = I[(I.timestamp >= cut) & I.severity.isin(["High", "Critical"])]
    y = after.groupby("location_id").size().reindex(f.location_id, fill_value=0).to_numpy()
    return f, y, float((cut - start).days), float((end - cut).days)
