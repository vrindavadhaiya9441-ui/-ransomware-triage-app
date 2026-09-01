"""
core.py — Model, explainability and triage logic for the
Explainable, Human-in-the-Loop Ransomware Forensic Triage demo.

Loading priority (first that succeeds wins):
  1. REAL artefacts   — your trained dedup_XGBoost model, saved SHAP
                        TreeExplainer, operating_point.json and the
                        deduplicated hold-out test set. Shows your real
                        dissertation model, numbers and explanations.
  2. Train from CSV    — fallback: train from an MLRan feature CSV in data/.

Nothing here executes malware. MLRan ships pre-extracted BEHAVIOURAL
FEATURES only, so the whole pipeline is safe on any laptop.
"""
from __future__ import annotations
import os, glob, json, pickle, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_DIR   = os.path.join(HERE, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DECISION_COLOR = {"AUTO-CLEAR": "#1a7f37", "ESCALATE": "#9a6700", "AUTO-FLAG": "#b42318"}


def triage_decision(p: float, low: float, high: float) -> str:
    if p < low:
        return "AUTO-CLEAR"
    if p >= high:
        return "AUTO-FLAG"
    return "ESCALATE"


# Positional adapter — the XGBoost booster stored non-matching feature names,
# so we always predict on positional arrays (column order is the training order).
class PositionalModel:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X):
        arr = X.values if hasattr(X, "values") else np.asarray(X)
        return self.model.predict_proba(arr.astype(float))


def build_explainer(base_model, X_background):
    import shap
    name = type(base_model).__name__.lower()
    if any(k in name for k in ("forest", "xgb", "boosting", "tree")):
        return shap.TreeExplainer(base_model)
    bg = shap.utils.sample(X_background, min(100, len(X_background)), random_state=0)
    return shap.LinearExplainer(base_model, bg)


def shap_top_contributions(explainer, x_row, feature_names, top_k: int = 8):
    arr = np.asarray(x_row, dtype=float)
    sv = np.asarray(explainer.shap_values(arr))
    if sv.ndim == 3:
        vals = sv[0, :, -1]
    elif sv.ndim == 2:
        vals = sv[0]
    else:
        vals = sv
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    return [{
        "feature": feature_names[i],
        "value": float(arr[0, i]),
        "contribution": float(vals[i]),
        "direction": "ransomware" if vals[i] > 0 else "goodware",
    } for i in order]


# --------------------------------------------------------------------------- #
# MODE 1 — real artefacts
# --------------------------------------------------------------------------- #
REAL = {"xgb": "dedup_XGBoost.joblib", "shap": "shap_explainer.joblib",
        "op": "operating_point.json", "summary": "FINAL_results_summary.json"}


def _find(fn):
    for d in (MODELS_DIR, DATA_DIR, HERE):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    return None


def real_artifacts_present() -> bool:
    return (all(_find(REAL[k]) for k in ("xgb", "shap", "op"))
            and _find("X_test_dd.pkl") is not None
            and _find("y_test_dd.pkl") is not None)


def load_real_bundle() -> dict:
    xgb = joblib.load(_find(REAL["xgb"]))
    explainer = joblib.load(_find(REAL["shap"]))
    with open(_find(REAL["op"])) as f:
        op = json.load(f)
    low, high = float(op["low"]), float(op["high"])

    X = pickle.load(open(_find("X_test_dd.pkl"), "rb"))
    y = pickle.load(open(_find("y_test_dd.pkl"), "rb"))
    if hasattr(y, "values"):
        y = y.values
    y = np.asarray(y).astype(int)
    feature_names = list(X.columns)

    model = PositionalModel(xgb)
    p = model.predict_proba(X)[:, 1]

    bank = X.copy(); bank["_true"] = y; bank["_prob"] = p

    preds = (p >= 0.5).astype(int)
    dec = np.array([triage_decision(pp, low, high) for pp in p])
    live = {"n": int(len(y)),
            "accuracy": round(float((preds == y).mean()), 4),
            "auto_handled": round(float(np.mean(dec != "ESCALATE")), 4),
            "escalation": round(float(np.mean(dec == "ESCALATE")), 4),
            "missed": int(np.sum((dec == "AUTO-CLEAR") & (y == 1)))}

    reported = {}
    sp = _find(REAL["summary"])
    if sp:
        with open(sp) as f:
            reported = json.load(f)

    meta = {"mode": "real", "best_model": "XGBoost (deduplicated)",
            "reported": reported, "operating_point": {"low": low, "high": high},
            "live_test": live, "n_features": len(feature_names)}

    return {"model": model, "base_model": xgb, "explainer": explainer,
            "feature_names": feature_names, "thresholds": (low, high),
            "sample_bank": bank.reset_index(drop=True), "meta": meta}


# --------------------------------------------------------------------------- #
# MODE 2 — train from CSV (fallback)
# --------------------------------------------------------------------------- #
NON_FEATURE_COLS = {"sample_type", "label", "class", "target", "y", "sample",
                    "sample_id", "id", "hash", "sha256", "md5", "family",
                    "ransomware_family", "name", "filename", "split", "index",
                    "unnamed: 0"}
TARGET_CANDIDATES = ["sample_type", "label", "class", "target", "y"]


def find_dataset(data_dir=DATA_DIR):
    c = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    return c[0] if c else None


def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    target = next((c for c in df.columns if c.lower() in TARGET_CANDIDATES), None)
    if target is None:
        raise ValueError(f"No target column found. Columns: {list(df.columns)[:8]}")
    y = df[target]
    if y.dtype == object:
        y = (y.astype(str).str.lower()
               .map(lambda v: 1 if any(k in v for k in ("ransom", "malicious", "1")) else 0))
    y = y.astype(int)
    drop = [c for c in df.columns if c.lower() in NON_FEATURE_COLS or c == target]
    X = df.drop(columns=drop, errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]
    return X, y, list(X.columns)


def derive_thresholds(p, y, grid=None):
    p = np.asarray(p, float); y = np.asarray(y, int)
    if grid is None:
        grid = np.round(np.linspace(0.01, 0.99, 99), 3)
    low = 0.0
    for t in grid:
        if not np.any((y == 1) & (p < t)): low = float(t)
        else: break
    high = 1.0
    for t in grid[::-1]:
        if not np.any((y == 0) & (p >= t)): high = float(t)
        else: break
    if low >= high: low, high = 0.20, 0.80
    return round(low, 3), round(high, 3)


def train_from_dataframe(X, y, feature_names):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    try:
        from xgboost import XGBClassifier
        have_xgb = True
    except Exception:
        have_xgb = False
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    models = {"Logistic Regression": LogisticRegression(max_iter=2000),
              "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)}
    if have_xgb:
        models["XGBoost"] = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                          eval_metric="logloss", n_jobs=-1, random_state=42)
    best, best_f1, fitted = None, -1, {}
    for n, m in models.items():
        m.fit(X_tr, y_tr); fitted[n] = m
        f1 = f1_score(y_te, m.predict(X_te))
        if f1 > best_f1: best, best_f1 = n, f1
    model = fitted[best]
    p = model.predict_proba(X_te)[:, 1]
    low, high = derive_thresholds(p, y_te.values)
    bank = X_te.copy(); bank["_true"] = y_te.values; bank["_prob"] = p
    parts = [bank[bank["_true"] == c].sample(min(int((bank["_true"] == c).sum()), 40), random_state=1)
             for c in (0, 1)]
    bank = pd.concat(parts).sample(frac=1, random_state=2)
    meta = {"mode": "trained", "best_model": best,
            "live_test": {"n": int(len(y_te)),
                          "accuracy": round(float((model.predict(X_te) == y_te).mean()), 4),
                          "roc_auc": round(float(roc_auc_score(y_te, p)), 4)},
            "n_features": len(feature_names)}
    return {"model": model, "base_model": model, "explainer": None,
            "feature_names": feature_names, "thresholds": (low, high),
            "sample_bank": bank.reset_index(drop=True), "meta": meta}


def get_bundle(force_train=False):
    if real_artifacts_present() and not force_train:
        return load_real_bundle(), "real"
    ds = find_dataset()
    if ds is None:
        raise FileNotFoundError(
            "No real artefacts and no dataset CSV found. Put your model files "
            "(dedup_XGBoost.joblib, shap_explainer.joblib, operating_point.json) "
            "and X_test_dd.pkl / y_test_dd.pkl in models/, or an MLRan CSV in data/.")
    X, y, feats = load_dataset(ds)
    return train_from_dataframe(X, y, feats), "trained"
