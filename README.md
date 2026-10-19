# Label-Transition Auditing vs. Cosine Outliers for Label-Flipping Clients

Label-flipping attacks are a known threat in federated learning, and trust-aware systems need client-level detectors that come with controlled false-alarm rates. The paper "Conformal Label-Transition Auditing Against Label-Flipping Attacks in Trust-Aware Federated Learning" (2026) proposes auditing clients through label-transition statistics against a reference model, with conformal calibration of the flag threshold. This experiment checks that idea on a small, public stand-in setting.

**Question:** Can label-transition features from a reference model, scored by a gradient-boosted classifier with a conformally calibrated threshold, detect label-flipping federated clients better than a standard update-cosine outlier detector?

**Dataset:** scikit-learn's bundled digits dataset (1,797 images, 10 classes, loaded without download). It is *not* the paper's own data; the paper's federated setting is not public. Federations are simulated from it with Dirichlet label skew and 30% label-flipping clients.

**Baseline vs. improved:**
- *Baseline:* cosine similarity of each client's one-step update to the median update, flagged by a robust z-score threshold.
- *Improved:* class-balanced LightGBM on the client's 10x10 label-transition matrix (reference-model predictions vs. client labels) plus update-cosine features, with its flag threshold set by split-conformal calibration at a 10% false-alarm rate.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json` (and a bar chart to `figures/`).

## Results

Results are written to results.json.

## Reference

This project is motivated by the following paper:

> CW Wu, WC Hsu, YA Lien, SH Huang (2026). *Conformal Label-Transition Auditing Against Label-Flipping Attacks in Trust-Aware Federated Learning*. 2026 8th International. https://ieeexplore.ieee.org/abstract/document/11666546/

Paper author: Prof. Li-Der Chou, National Central University (cld@csie.ncu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
