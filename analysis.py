"""Label-transition auditing (LightGBM + conformal threshold) vs. cosine-outlier baseline
for label-flipping federated clients. Stand-in data: sklearn digits. Seeds 42/43/44."""
import json
import os
import sys
import time

import lightgbm as lgb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, train_test_split  # noqa: E402

SEED = 42
SEEDS = [42, 43, 44]                      # each seed re-simulates federations and LightGBM
N_CLIENTS, ATTACK_FRAC, CLIENT_SIZE, DIR_ALPHA = 20, 0.3, 80, 0.3
N_FED = {"train": 30, "cal": 20, "test": 30}  # federations per split
FPR_TARGET = 0.10                         # split-conformal false-alarm rate
Z_FLAG = -2.5                             # baseline: flag if robust z of cosine < Z_FLAG
GRID = [dict(num_leaves=nl, n_estimators=ne) for nl in (7, 15) for ne in (100, 300)]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def grad_update(Xc, yc, ref):
    """One local step: gradient of the server reference model's CE loss on client data."""
    P = ref.predict_proba(Xc)
    G = (P - np.eye(10)[yc]).T @ np.hstack([Xc, np.ones((len(Xc), 1))]) / len(Xc)
    return G.ravel()


def make_federation(pool_X, pool_y, rng):
    """Dirichlet label-skewed clients; ~30% of clients flip one source class to a target class."""
    by_cls = [np.where(pool_y == c)[0] for c in range(10)]
    bad = {int(v) for v in rng.choice(N_CLIENTS, int(round(ATTACK_FRAC * N_CLIENTS)), replace=False)}
    clients = []
    for i in range(N_CLIENTS):
        counts = rng.multinomial(CLIENT_SIZE, rng.dirichlet(DIR_ALPHA * np.ones(10)))
        idx = np.concatenate([rng.choice(by_cls[c], k) for c, k in enumerate(counts) if k > 0])
        Xc, yc = pool_X[idx], pool_y[idx].copy()
        if i in bad:
            cnt = np.bincount(yc, minlength=10)
            present = np.where(cnt >= 5)[0]
            if len(present) == 0:
                present = np.array([np.argmax(cnt)])
            s = rng.choice(present)
            t = rng.choice([c for c in range(10) if c != s])
            yc[yc == s] = t
        clients.append((Xc, yc, int(i in bad)))
    return clients


def federation_features(pool_X, pool_y, ref, rng):
    """Per-client features: 10x10 transition matrix + update-cosine features."""
    clients = make_federation(pool_X, pool_y, rng)
    G = np.array([grad_update(Xc, yc, ref) for Xc, yc, _ in clients])
    med = np.median(G, axis=0)
    norms = np.linalg.norm(G, axis=1)
    cos = G @ med / (norms * np.linalg.norm(med) + 1e-12)
    # robust z-score of cosine to median update (baseline statistic)
    mad = 1.4826 * np.median(np.abs(cos - np.median(cos))) + 1e-12
    z = (cos - np.median(cos)) / mad
    log_norm = np.log(norms / (np.linalg.norm(med) + 1e-12) + 1e-12)
    F = []
    for j, (Xc, yc, _) in enumerate(clients):
        pred = ref.predict(Xc)
        T = np.zeros((10, 10))
        np.add.at(T, (pred, yc), 1)          # rows: reference prediction, cols: client label
        T /= len(yc)
        F.append(np.concatenate([T.ravel(), [cos[j], z[j], log_norm[j]]]))
    return np.array(F), np.array([b for _, _, b in clients]), z


def build(pool, n_fed, rng, ref):
    out = [federation_features(pool[0], pool[1], ref, rng) for _ in range(n_fed)]
    F = np.vstack([o[0] for o in out])
    y = np.concatenate([o[1] for o in out])
    groups = np.repeat(np.arange(n_fed), N_CLIENTS)
    z = np.concatenate([o[2] for o in out])
    return F, y, groups, z


def metrics(y, score, flag):
    """Client-level ROC-AUC, F1 of flagged clients, and false-positive rate on clean clients."""
    return dict(auc=roc_auc_score(y, score), f1=f1_score(y, flag), fpr=float(flag[y == 0].mean()))


def run_seed(seed, ref, client_pool, test_pool):
    rng = np.random.default_rng(seed)
    Ftr, ytr, gtr, _ = build(client_pool, N_FED["train"], rng, ref)
    Fca, yca, _, _ = build(client_pool, N_FED["cal"], rng, ref)
    Fte, yte, _, zte = build(test_pool, N_FED["test"], rng, ref)
    log(f"seed {seed}: train {len(ytr)} clients ({ytr.sum()} flipped), "
        f"cal {len(yca)}, test {len(yte)} ({yte.sum()} flipped)")

    # BASELINE: cosine-to-median robust z-score outlier detector (no training)
    base = metrics(yte, -zte, zte < Z_FLAG)

    # IMPROVED: LightGBM; small grid chosen by grouped CV on training federations only
    def make(p):
        return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, min_child_samples=10,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, verbose=-1, **p)
    cv = GroupKFold(n_splits=3)
    cv_auc = []
    for p in GRID:
        aucs = []
        for tr, va in cv.split(Ftr, ytr, gtr):
            m = make(p).fit(Ftr[tr], ytr[tr])
            aucs.append(roc_auc_score(ytr[va], m.predict_proba(Ftr[va])[:, 1]))
        cv_auc.append(np.mean(aucs))
    best = GRID[int(np.argmax(cv_auc))]
    model = make(best).fit(Ftr, ytr)
    log(f"seed {seed}: best grid {best} (CV AUC {max(cv_auc):.3f})")

    # split-conformal threshold: (1-alpha) quantile of clean calibration scores, finite-sample corrected
    s_cal = np.sort(model.predict_proba(Fca)[yca == 0, 1])
    n0 = len(s_cal)
    k = int(np.ceil((n0 + 1) * (1 - FPR_TARGET)))
    thr = s_cal[k - 1] if k <= n0 else np.inf
    imp_score = model.predict_proba(Fte)[:, 1]
    imp = metrics(yte, imp_score, imp_score > thr)
    log(f"seed {seed}: baseline AUC {base['auc']:.3f} | improved AUC {imp['auc']:.3f}")
    return base, imp


def main():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0                                     # pixels to [0, 1]
    # split ONCE (stratified, seed 42): 30% test pool; remainder is train pool
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    # server-held clean reference split and client pool, both from the train pool
    X_ref, X_cli, y_ref, y_cli = train_test_split(X_tr, y_tr, test_size=0.7, stratify=y_tr,
                                                  random_state=SEED)
    ref = LogisticRegression(max_iter=2000).fit(X_ref, y_ref)
    log(f"reference model trained on {len(y_ref)} clean images; "
        f"accuracy on held-out test pool {ref.score(X_te, y_te):.3f}")

    runs = [run_seed(s, ref, (X_cli, y_cli), (X_te, y_te)) for s in SEEDS]
    b = np.array([r[0]["auc"] for r in runs])
    i = np.array([r[1]["auc"] for r in runs])
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(i))):
        raise RuntimeError("non-finite ROC-AUC; cannot produce meaningful metrics")

    def agg(key, which):
        return float(np.mean([r[which][key] for r in runs]))

    metrics_out = {
        "baseline_score": float(b.mean()), "improved_score": float(i.mean()),
        "improvement": float(i.mean() - b.mean()),
        "baseline_score_std": float(b.std()), "improved_score_std": float(i.std()),
        "baseline_f1": agg("f1", 0), "improved_f1": agg("f1", 1),
        "baseline_fpr": agg("fpr", 0), "improved_fpr": agg("fpr", 1),
    }
    log(f"mean test ROC-AUC: baseline {metrics_out['baseline_score']:.3f}, "
        f"improved {metrics_out['improved_score']:.3f}")

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["cosine baseline", "label-transition"], [b.mean(), i.mean()],
           yerr=[b.std(), i.std()], color=["grey", "tab:blue"], capsize=4)
    ax.set_ylabel("test client ROC-AUC (mean ± sd, 3 seeds)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("figures/auc_bar.png", dpi=120)

    results = {
        "question": "Can label-transition features from a reference model, scored by a gradient-boosted "
                    "classifier with a conformally calibrated threshold, detect label-flipping federated "
                    "clients better than a standard update-cosine outlier detector?",
        "method": "Simulated federations (20 Dirichlet-skewed clients, 30% flip one source class to a target) "
                  "on sklearn digits; server reference logistic regression; baseline = robust z-score of "
                  "update cosine to median; improved = class-balanced LightGBM on 10x10 transition matrix "
                  "plus cosine features, threshold by split-conformal at 10% FPR on calibration federations.",
        "dataset": {
            "name": "scikit-learn digits (load_digits, bundled)",
            "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
            "n_rows": int(len(X)), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "client-level ROC-AUC on test federations",
        "higher_is_better": True,
        "baseline": "Standard client-update outlier detector: cosine of each client's one-step update to the "
                    "median update, flagged by a robust z-score threshold (no learning).",
        "improved": "Class-balanced LightGBM on the client's 10x10 label-transition matrix (reference-model "
                    "predictions vs. client labels) plus update-cosine features; flag threshold from "
                    "split-conformal calibration at 10% false-alarm rate.",
        "metrics": metrics_out,
        "notes": "Stand-in for a non-public federated setting: digits is small (federations of 20 clients "
                 "with 80 images each), so variance is high; results are over seeds 42-44. n_train/n_test are "
                 "image-level counts of the single stratified split (70/30); the client pool is a further split "
                 "of the train part (reference vs. client), and test federations are drawn only from the test "
                 "part. Calibration and training federations come from the client pool. Hyper-parameter grid "
                 "chosen by grouped CV on training federations only. Conformal guarantee is marginal and "
                 "assumes exchangeable clients. LightGBM source: https://github.com/microsoft/LightGBM.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("wrote results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # non-zero exit if metrics cannot be produced
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
