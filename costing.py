"""
Turn per-image scores into decisions and euros.

Shared by train.py (which saves the sweep and the chosen operating point) and
app.py (which recomputes everything live when the sliders move), so both use
exactly the same arithmetic.

The decision is made per image (one camera frame): does this frame show at
least one worker without a safety vest? Each image gets one score, the highest
`no_safety_vest` confidence the model found in it (0 if it found none).

Three outcomes, set by two thresholds t_low <= t_high:

    score >= t_high          -> FAIL    alert sent straight to the supervisor
    t_low <= score < t_high  -> REVIEW  a person looks at the frame first
    score <  t_low           -> PASS    nothing happens

With t_low == t_high there is no review band and the model decides alone.

Assumption built in here (say it in REPORT.md): a reviewer judges a frame
correctly. A reviewed violation ends up handled like a true positive, a
reviewed clean frame like a true negative; both also pay for the review.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COSTS_FILE = ROOT / "costs.json"


def load_costs(path: Path = COSTS_FILE) -> dict:
    costs = json.loads(path.read_text())
    return {k: v for k, v in costs.items() if not k.startswith("_")}


def review_cost(costs: dict) -> float:
    """Euros of reviewer time spent on one frame."""
    return costs["review_seconds"] / 3600 * costs["reviewer_cost_per_hour"]


def outcome_rates(y: np.ndarray, score: np.ndarray, t_low: float, t_high: float) -> dict:
    """Share of positives and of negatives that land in each outcome.

    Rates rather than counts, so they can be re-weighted to the assumed base
    rate instead of the test set's own share of violations.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    fail = score >= t_high
    passed = score < t_low
    review = ~fail & ~passed
    pos, neg = y, ~y

    def share(mask, group):
        return float(mask[group].mean()) if group.any() else 0.0

    return {
        "tp": share(fail, pos), "review_pos": share(review, pos), "fn": share(passed, pos),
        "fp": share(fail, neg), "review_neg": share(review, neg), "tn": share(passed, neg),
    }


def outcome_counts(y: np.ndarray, score: np.ndarray, t_low: float, t_high: float) -> pd.DataFrame:
    """Raw test-set confusion table: true class x outcome."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    fail = score >= t_high
    passed = score < t_low
    review = ~fail & ~passed
    rows = {}
    for label, group in (("violation (no vest)", y), ("no violation", ~y)):
        rows[label] = {
            "FAIL (alert)": int((fail & group).sum()),
            "REVIEW (person)": int((review & group).sum()),
            "PASS": int((passed & group).sum()),
        }
    return pd.DataFrame(rows).T


def cost_per_case(rates: dict, costs: dict) -> float:
    """Expected euros per frame at the assumed base rate."""
    pi = costs["base_rate"]
    rc = review_cost(costs)
    on_positive = (
        rates["tp"] * costs["cost_tp"]
        + rates["fn"] * costs["cost_fn"]
        + rates["review_pos"] * (rc + costs["cost_tp"])
    )
    on_negative = (
        rates["fp"] * costs["cost_fp"]
        + rates["tn"] * costs["cost_tn"]
        + rates["review_neg"] * (rc + costs["cost_tn"])
    )
    return pi * on_positive + (1 - pi) * on_negative


def review_share(rates: dict, costs: dict) -> float:
    """Share of all frames sent to a person, at the assumed base rate."""
    pi = costs["base_rate"]
    return pi * rates["review_pos"] + (1 - pi) * rates["review_neg"]


def staffing(share: float, costs: dict) -> dict:
    """What a review share means in frames, hours and people per day."""
    frames = share * costs["cases_per_day"]
    hours = frames * costs["review_seconds"] / 3600
    return {
        "frames_per_day": frames,
        "review_hours_per_day": hours,
        "reviewers_needed": hours / costs["reviewer_hours_per_day"],
    }


def evaluate(y, score, t_low, t_high, costs) -> dict:
    """Everything the dashboard shows for one pair of thresholds."""
    rates = outcome_rates(y, score, t_low, t_high)
    share = review_share(rates, costs)
    return {
        "t_low": t_low,
        "t_high": t_high,
        **rates,
        "cost_per_1000": 1000 * cost_per_case(rates, costs),
        "review_share": share,
        **staffing(share, costs),
    }


THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)


def sweep(y, score, costs, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Single threshold (no review band): cost at every threshold."""
    return pd.DataFrame([evaluate(y, score, t, t, costs) for t in thresholds])


def best_band(y, score, costs, grid=np.round(np.linspace(0, 1, 51), 2)) -> dict:
    """Cheapest pair t_low <= t_high. Ties go to the pair reviewing less."""
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            result = evaluate(y, score, lo, hi, costs)
            key = (round(result["cost_per_1000"], 6), result["review_share"])
            if best is None or key < best[0]:
                best = (key, result)
    return best[1]


def alternatives(y, score, costs) -> dict:
    """The operating point priced against the three alternatives in the brief.

    - do nothing: every frame passes, every violation is missed
    - person checks everything: every frame reviewed, nothing missed
    - model alone: cheapest single threshold, no review band
    - model + review: cheapest two-threshold band
    """
    pi = costs["base_rate"]
    rc = review_cost(costs)
    do_nothing = pi * costs["cost_fn"] + (1 - pi) * costs["cost_tn"]
    check_all = rc + pi * costs["cost_tp"] + (1 - pi) * costs["cost_tn"]
    single = sweep(y, score, costs)
    model_alone = single.loc[single["cost_per_1000"].idxmin()].to_dict()
    band = best_band(y, score, costs)
    return {
        "do_nothing": {"cost_per_1000": 1000 * do_nothing, "review_share": 0.0,
                       **staffing(0.0, costs)},
        "person_checks_everything": {"cost_per_1000": 1000 * check_all, "review_share": 1.0,
                                     **staffing(1.0, costs)},
        "model_alone": model_alone,
        "model_plus_review": band,
    }
