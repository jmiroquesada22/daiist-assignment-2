"""
Assignment 2: Gradio dashboard for the safety-vest checker.

    uv run python main.py app

Loads the committed weights (weights/best.pt) and what train.py saved in
artifacts/. Never trains anything. Costs are read from costs.json each time
the app starts, so changing a cost figure only needs an app restart.

Views:
- Thresholds & cost: two sliders split frames into PASS / REVIEW / FAIL; the
  test-set confusion table, cost per 1,000 frames, review share and staffing
  update live, priced against the alternatives.
- Cost curve: cost against a single threshold, chosen operating point marked.
- Try an image: upload a frame, see boxes and the decision at the thresholds.
- Per-class results: per-class AP/precision/recall for the finetuned model
  and both baselines, plus the detection confusion matrix.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import costing

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"
POSITIVE = "no_safety_vest"

model = YOLO(str(WEIGHTS))
scores = pd.read_csv(ARTIFACTS / "test_scores.csv")
detection = json.loads((ARTIFACTS / "detection_metrics.json").read_text())
confusion = json.loads((ARTIFACTS / "detection_confusion.json").read_text())
operating = json.loads((ARTIFACTS / "operating_point.json").read_text())
train_info = json.loads((ARTIFACTS / "train_info.json").read_text())
raw_costs = json.loads(costing.COSTS_FILE.read_text())
costs = costing.load_costs()

Y = scores["y"].to_numpy()
SCORE = scores["finetuned"].to_numpy()
POS_ID = [i for i, n in model.names.items() if n == POSITIVE][0]
SINGLE = costing.sweep(Y, SCORE, costs)
BASELINE_SWEEPS = {m: costing.sweep(Y, scores[m].to_numpy(), costs)
                   for m in ("coco_zero_shot", "predict_nothing")}
CHOSEN_SINGLE = float(SINGLE.loc[SINGLE["cost_per_1000"].idxmin(), "t_low"])
BAND = costing.best_band(Y, SCORE, costs)
ALTERNATIVES = costing.alternatives(Y, SCORE, costs)


def euros(x: float) -> str:
    return f"€{x:,.2f}"


# --------------------------------------------------------------------------- #
# Thresholds & cost
# --------------------------------------------------------------------------- #
def ordered(t_low: float, t_high: float) -> tuple[float, float]:
    return (t_low, t_high) if t_low <= t_high else (t_high, t_low)


def comparison_table(current: dict) -> pd.DataFrame:
    rows = [
        ("Do nothing (all pass)", ALTERNATIVES["do_nothing"]),
        ("Person checks every frame", ALTERNATIVES["person_checks_everything"]),
        (f"Model alone (t = {ALTERNATIVES['model_alone']['t_low']:.2f})",
         ALTERNATIVES["model_alone"]),
        (f"Model + review, cheapest band ({BAND['t_low']:.2f} / {BAND['t_high']:.2f})", BAND),
        (f"Your sliders ({current['t_low']:.2f} / {current['t_high']:.2f})", current),
    ]
    return pd.DataFrame([{
        "Option": name,
        "Cost per 1,000 frames": euros(r["cost_per_1000"]),
        "Sent to review": f"{r['review_share']:.1%}",
        "Frames reviewed / day": f"{r['frames_per_day']:,.0f}",
        "Reviewers needed": f"{r['reviewers_needed']:.2f}",
    } for name, r in rows])


def cost_curve(t_low: float, t_high: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=SINGLE["t_low"], y=SINGLE["cost_per_1000"],
                             name="Finetuned model", line=dict(width=3)))
    labels = {"coco_zero_shot": "COCO zero-shot", "predict_nothing": "Predict nothing"}
    for m, s in BASELINE_SWEEPS.items():
        fig.add_trace(go.Scatter(x=s["t_low"], y=s["cost_per_1000"], name=labels[m],
                                 line=dict(dash="dot", width=1.5)))
    best = SINGLE.loc[SINGLE["cost_per_1000"].idxmin()]
    fig.add_trace(go.Scatter(
        x=[best["t_low"]], y=[best["cost_per_1000"]], mode="markers+text",
        marker=dict(size=13, symbol="star"), name="Chosen threshold",
        text=[f"chosen {best['t_low']:.2f}"], textposition="top center"))
    for t, label in ((t_low, "review from"), (t_high, "fail from")):
        fig.add_vline(x=t, line_dash="dash", line_color="grey",
                      annotation_text=f"{label} {t:.2f}", annotation_position="top")
    fig.update_layout(
        title="Expected cost per 1,000 frames vs. confidence threshold (no review band)",
        xaxis_title="Threshold on the image's highest no_safety_vest confidence",
        yaxis_title="€ per 1,000 frames", height=430, margin=dict(t=60, b=40),
        legend=dict(orientation="h", y=-0.2))
    return fig


def update(t_low: float, t_high: float):
    t_low, t_high = ordered(t_low, t_high)
    current = costing.evaluate(Y, SCORE, t_low, t_high, costs)
    counts = costing.outcome_counts(Y, SCORE, t_low, t_high).reset_index(
        names="True class (test set)")
    summary = (
        f"### {euros(current['cost_per_1000'])} per 1,000 frames\n"
        f"**{current['review_share']:.1%}** of frames go to a person "
        f"({current['frames_per_day']:,.0f} a day at {costs['cases_per_day']:,} frames/day, "
        f"{current['review_hours_per_day']:.1f} hours of checking, "
        f"**{current['reviewers_needed']:.2f} reviewers** at "
        f"{costs['reviewer_hours_per_day']} h/day).\n\n"
        f"Of real violations: {current['tp']:.1%} alerted, {current['review_pos']:.1%} reviewed, "
        f"**{current['fn']:.1%} missed**. Of clean frames: **{current['fp']:.1%} false alarms**, "
        f"{current['review_neg']:.1%} reviewed, {current['tn']:.1%} passed.\n\n"
        f"Cost is re-weighted to an assumed base rate of {costs['base_rate']:.1%} violations; "
        f"the test set has {Y.mean():.1%}."
    )
    return counts, summary, comparison_table(current), cost_curve(t_low, t_high)


# --------------------------------------------------------------------------- #
# Try an image
# --------------------------------------------------------------------------- #
def predict_image(image: np.ndarray, t_low: float, t_high: float):
    if image is None:
        return None, "Upload an image."
    t_low, t_high = ordered(t_low, t_high)
    result = model.predict(image, conf=0.001, verbose=False)[0]
    boxes = result.boxes
    pos_conf = boxes.conf[boxes.cls == POS_ID]
    score = float(pos_conf.max()) if len(pos_conf) else 0.0
    shown = result[boxes.conf >= max(t_low, 0.05)]  # draw what could matter
    annotated = shown.plot()[:, :, ::-1]  # BGR -> RGB

    if score >= t_high:
        decision = "❌ **FAIL**: alert the supervisor"
    elif score >= t_low:
        decision = "🔍 **REVIEW**: a person checks this frame"
    else:
        decision = "✅ **PASS**"
    counts = {model.names[int(c)]: int((boxes.cls[boxes.conf >= t_low] == c).sum())
              for c in boxes.cls.unique()}
    text = (f"{decision}\n\nHighest `{POSITIVE}` confidence: **{score:.3f}** "
            f"(review from {t_low:.2f}, fail from {t_high:.2f}).\n\n"
            f"Boxes at or above {t_low:.2f}: {counts or 'none'}")
    return annotated, text


# --------------------------------------------------------------------------- #
# Per-class results
# --------------------------------------------------------------------------- #
def per_class_table() -> pd.DataFrame:
    labels = {"finetuned": "Finetuned yolo11n", "coco_zero_shot": "COCO yolo11n zero-shot",
              "predict_nothing": "Predict nothing"}
    rows = []
    for key, label in labels.items():
        rep = detection[key]
        for cname, m in rep["per_class"].items():
            rows.append({
                "Model": label,
                "Class": cname + (" (positive)" if cname == POSITIVE else ""),
                "Test instances": m["n_gt"],
                "AP50": round(m["ap50"], 3),
                "AP50-95": round(m["ap50_95"], 3),
                f"Precision @ conf {detection['_notes']['precision_recall_at_conf']}": round(m["precision"], 3),
                f"Recall @ conf {detection['_notes']['precision_recall_at_conf']}": round(m["recall"], 3),
            })
        rows.append({"Model": label, "Class": "ALL (mean)", "Test instances": None,
                     "AP50": round(rep["map50"], 3), "AP50-95": round(rep["map50_95"], 3)})
    return pd.DataFrame(rows)


def confusion_figure() -> go.Figure:
    m = np.array(confusion["matrix"])
    names = confusion["names"]
    fig = go.Figure(go.Heatmap(z=m, x=names, y=names, text=m, texttemplate="%{text}",
                               colorscale="Blues", showscale=False))
    fig.update_layout(
        title=f"Finetuned model, test split (conf ≥ {confusion['conf']}, IoU ≥ {confusion['iou']})",
        xaxis_title="True", yaxis_title="Predicted", yaxis_autorange="reversed",
        height=420, margin=dict(t=60))
    return fig


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
ds = train_info["dataset"]
header = (
    "# Safety-vest checker: pass, review or fail\n"
    f"Each camera frame gets the model's highest `{POSITIVE}` confidence. "
    "Below the review threshold it passes, at or above the fail threshold an alert goes "
    "to the supervisor, and in between a person looks first.\n\n"
    f"Model `weights/best.pt` (yolo11n finetuned) · dataset "
    f"`{ds['workspace']}/{ds['project']}/{ds['version']}` · "
    f"test split: {len(Y)} images, {int(Y.sum())} with a violation."
)
if "_status" in raw_costs:
    header += ("\n\n> ⚠️ **costs.json still holds placeholder figures.** "
               "Every euro amount on this page is meaningless until they are replaced.")
if train_info.get("smoke_run"):
    header += "\n\n> ⚠️ These artifacts come from a SMOKE test run, not the real training."

with gr.Blocks(title="Safety-vest checker") as demo:
    gr.Markdown(header)
    with gr.Row():
        t_low = gr.Slider(0, 1, value=BAND["t_low"], step=0.01,
                          label="Review threshold (below this: PASS)")
        t_high = gr.Slider(0, 1, value=BAND["t_high"], step=0.01,
                           label="Fail threshold (at or above this: FAIL / alert)")
    gr.Markdown(f"Sliders start at the cheapest band found by the sweep "
                f"({BAND['t_low']:.2f} / {BAND['t_high']:.2f}). Set them equal for no review band.")

    with gr.Tab("Thresholds & cost"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Test-set outcomes** (raw image counts)")
                counts_out = gr.Dataframe(interactive=False)
            summary_out = gr.Markdown()
        gr.Markdown("**Priced against the alternatives**")
        compare_out = gr.Dataframe(interactive=False)
        curve_out = gr.Plot()

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload a frame")
            image_out = gr.Image(label="Detections")
        decision_out = gr.Markdown()
        image_in.change(predict_image, [image_in, t_low, t_high], [image_out, decision_out])

    with gr.Tab("Per-class results"):
        gr.Markdown(f"Positive class: **{POSITIVE}**. All three models scored by the same "
                    "code on the same test split.")
        gr.Dataframe(per_class_table(), interactive=False)
        gr.Plot(confusion_figure())

    outputs = [counts_out, summary_out, compare_out, curve_out]
    for slider in (t_low, t_high):
        slider.change(update, [t_low, t_high], outputs)
        slider.change(predict_image, [image_in, t_low, t_high], [image_out, decision_out])
    demo.load(update, [t_low, t_high], outputs)


if __name__ == "__main__":
    demo.launch()
