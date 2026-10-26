"""
Assignment 2: training and evaluation pipeline (Safety Vests).

    uv run python main.py train

Start to finish, no manual steps:

1. Download roboflow-universe-projects/safety-vests (version in dataset.py).
2. Finetune yolo11n from the COCO-pretrained checkpoint; copy the best weights
   to weights/best.pt.
3. Evaluate on the test split, with the same code for all three models:
   - finetuned yolo11n
   - pretrained COCO yolo11n zero-shot (COCO has no vest class, so every
     "person" box is scored as a candidate for each class)
   - trivial baseline that predicts nothing
   Detection: per-class AP50, AP50-95, precision and recall, plus the
   finetuned model's detection confusion matrix.
   Decision (per image): each image's score is its highest no_safety_vest
   confidence.
4. Cost the per-image decisions with costs.json, sweep the threshold, find the
   cheapest single threshold and the cheapest pass/review/fail band, and price
   them against doing nothing and having a person check everything.
5. Save all of it to artifacts/ for app.py, which never trains.

Settings through environment variables:
    EPOCHS=30       training epochs (default 30)
    SKIP_TRAIN=1    reuse weights/best.pt, only re-run evaluation and costing
    SMOKE=1         tiny run (1 epoch, 2% of train, 320px) to test the pipeline
"""

import json
import os
import platform
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import compute_ap

import costing
from dataset import DATA_DIR, PROJECT, VERSION, WORKSPACE, class_names, download_dataset

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"

POSITIVE = "no_safety_vest"  # the class whose presence is a violation
BASE_MODEL = "yolo11n.pt"     # COCO-pretrained; also the zero-shot baseline
COCO_PERSON = 0

SMOKE = os.environ.get("SMOKE") == "1"
SKIP_TRAIN = os.environ.get("SKIP_TRAIN") == "1"
EPOCHS = int(os.environ.get("EPOCHS", 1 if SMOKE else 30))
IMGSZ = 320 if SMOKE else 640
SEED = 0

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # COCO mAP50-95
REPORT_CONF = 0.25  # confidence at which per-class precision/recall are quoted
MATRIX_IOU = 0.45   # IoU for the detection confusion matrix (Ultralytics default)


def device() -> str | int:
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(dev) -> str:
    if dev == 0:
        return torch.cuda.get_device_name(0)
    return "Apple MPS" if dev == "mps" else f"CPU ({platform.processor() or platform.machine()})"


# --------------------------------------------------------------------------- #
# 1-2. Data and finetuning
# --------------------------------------------------------------------------- #
def finetune(data_yaml: Path, dev) -> dict:
    model = YOLO(BASE_MODEL)
    started = time.time()
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=32 if dev == 0 else 16,
        patience=10,
        seed=SEED,
        deterministic=True,
        device=dev,
        fraction=0.02 if SMOKE else 1.0,
        project=str(ROOT / "runs"),
        name="finetune",
        exist_ok=True,
        plots=True,
    )
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)
    return {
        "epochs_requested": EPOCHS,
        "epochs_run": int(model.trainer.epoch) + 1,
        "imgsz": IMGSZ,
        "batch": 32 if dev == 0 else 16,
        "patience": 10,
        "seed": SEED,
        "train_fraction": 0.02 if SMOKE else 1.0,
        "train_minutes": round((time.time() - started) / 60, 1),
    }


# --------------------------------------------------------------------------- #
# 3. Evaluation on the test split
# --------------------------------------------------------------------------- #
def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b.T
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def load_ground_truth(names: list[str]) -> dict[str, dict[int, np.ndarray]]:
    """Test labels per image: {image path: {class id: boxes xyxy normalised}}."""
    gt = {}
    for image in sorted((DATA_DIR / "test" / "images").iterdir()):
        label = DATA_DIR / "test" / "labels" / f"{image.stem}.txt"
        rows = []
        if label.exists():
            rows = [r.split() for r in label.read_text().splitlines() if r.strip()]
        arr = np.array([[float(v) for v in r[:5]] for r in rows]).reshape(-1, 5)
        gt[str(image)] = {
            c: xywh_to_xyxy(arr[arr[:, 0] == c, 1:5]) for c in range(len(names))
        }
    return gt


def predict(model: YOLO, images: list[str], classes=None) -> dict[str, tuple]:
    """{image: (boxes xyxy normalised, confidences, class ids)} at near-zero conf,
    so every threshold can be applied afterwards."""
    out = {}
    for result in model.predict(images, conf=0.001, iou=0.7, imgsz=IMGSZ,
                                classes=classes, stream=True, verbose=False):
        b = result.boxes
        out[result.path] = (b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(),
                            b.cls.cpu().numpy().astype(int))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred_boxes, pred_conf, gt_boxes, iou_thr) -> np.ndarray:
    """Greedy one-to-one matching in descending confidence. True where a
    prediction hits a not-yet-claimed ground truth at IoU >= iou_thr."""
    hit = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return hit
    ious = iou_matrix(pred_boxes, gt_boxes)
    claimed = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-pred_conf):
        candidates = np.where(~claimed & (ious[i] >= iou_thr))[0]
        if len(candidates):
            j = candidates[np.argmax(ious[i, candidates])]
            claimed[j] = True
            hit[i] = True
    return hit


def class_metrics(per_image: dict[str, tuple], gt: dict, cls: int) -> dict:
    """AP50, AP50-95, and precision/recall at REPORT_CONF for one class.

    per_image[image] = (boxes, conf) of predictions counted as this class.
    """
    confs, hits = [], []
    n_gt = 0
    for image, gts in gt.items():
        g = gts[cls]
        n_gt += len(g)
        boxes, conf = per_image.get(image, (np.zeros((0, 4)), np.zeros(0)))
        if len(conf) == 0:
            continue
        confs.append(conf)
        hits.append(np.stack([match(boxes, conf, g, t) for t in IOU_THRESHOLDS], axis=1))
    if not confs or n_gt == 0:
        return {"n_gt": n_gt, "ap50": 0.0, "ap50_95": 0.0, "precision": 0.0,
                "recall": 0.0, "n_pred_at_conf": 0}
    conf = np.concatenate(confs)
    hit = np.concatenate(hits)
    order = np.argsort(-conf)
    conf, hit = conf[order], hit[order]
    aps = []
    for k in range(len(IOU_THRESHOLDS)):
        tp = np.cumsum(hit[:, k])
        fp = np.cumsum(~hit[:, k])
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-9)
        aps.append(compute_ap(recall, precision)[0])
    kept = conf >= REPORT_CONF
    tp50 = hit[kept, 0].sum()
    return {
        "n_gt": int(n_gt),
        "ap50": float(aps[0]),
        "ap50_95": float(np.mean(aps)),
        "precision": float(tp50 / kept.sum()) if kept.sum() else 0.0,
        "recall": float(tp50 / n_gt),
        "n_pred_at_conf": int(kept.sum()),
    }


def detection_report(preds_by_class: dict[int, dict], gt: dict, names: list[str]) -> dict:
    per_class = {names[c]: class_metrics(preds_by_class.get(c, {}), gt, c)
                 for c in range(len(names))}
    return {
        "per_class": per_class,
        "map50": float(np.mean([m["ap50"] for m in per_class.values()])),
        "map50_95": float(np.mean([m["ap50_95"] for m in per_class.values()])),
    }


def split_by_class(preds: dict, nc: int) -> dict[int, dict]:
    out = {c: {} for c in range(nc)}
    for image, (boxes, conf, cls) in preds.items():
        for c in range(nc):
            keep = cls == c
            out[c][image] = (boxes[keep], conf[keep])
    return out


def coco_as_each_class(preds: dict, nc: int) -> dict[int, dict]:
    """Zero-shot COCO: every person box is offered as a candidate for each class."""
    return {c: {img: (b, conf) for img, (b, conf, _) in preds.items()} for c in range(nc)}


def detection_confusion(preds: dict, gt: dict, nc: int) -> list[list[int]]:
    """[predicted, true] matrix with a trailing background row/column, at
    REPORT_CONF and MATRIX_IOU, class-agnostic matching so cross-class
    confusions show up (same layout as Ultralytics)."""
    m = np.zeros((nc + 1, nc + 1), dtype=int)
    for image, gts in gt.items():
        g_boxes = np.concatenate([gts[c] for c in range(nc)])
        g_cls = np.concatenate([np.full(len(gts[c]), c) for c in range(nc)])
        boxes, conf, cls = preds.get(image, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        keep = conf >= REPORT_CONF
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        claimed = np.zeros(len(g_boxes), dtype=bool)
        ious = iou_matrix(boxes, g_boxes)
        for i in np.argsort(-conf):
            candidates = np.where(~claimed & (ious[i] >= MATRIX_IOU))[0] if len(g_boxes) else []
            if len(candidates):
                j = candidates[np.argmax(ious[i, candidates])]
                claimed[j] = True
                m[cls[i], g_cls[j]] += 1
            else:
                m[cls[i], nc] += 1  # false positive (background)
        for j in np.where(~claimed)[0]:
            m[nc, g_cls[j]] += 1    # missed (predicted background)
    return m.tolist()


def image_scores(preds: dict, images: list[str], cls: int | None) -> np.ndarray:
    """Per-image score: highest confidence among boxes of `cls` (any class if None)."""
    scores = []
    for image in images:
        _, conf, c = preds.get(image, (None, np.zeros(0), np.zeros(0, int)))
        if cls is not None:
            conf = conf[c == cls]
        scores.append(float(conf.max()) if len(conf) else 0.0)
    return np.array(scores)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    data_yaml = download_dataset()
    names = class_names(data_yaml)
    pos = names.index(POSITIVE)
    dev = device()
    print(f"\nClasses {names}, positive class: {POSITIVE}. Device: {device_name(dev)}")

    # ---- finetune ----
    if SKIP_TRAIN and WEIGHTS.exists():
        print(f"SKIP_TRAIN=1: reusing {WEIGHTS.relative_to(ROOT)}")
        info_path = ARTIFACTS / "train_info.json"
        train_info = json.loads(info_path.read_text()).get("training", {}) if info_path.exists() else {}
    else:
        train_info = finetune(data_yaml, dev)
        train_info["device"] = device_name(dev)

    # ---- predictions on the test split ----
    gt = load_ground_truth(names)
    images = list(gt)
    print(f"\nEvaluating on {len(images)} test images ...")
    finetuned = YOLO(str(WEIGHTS))
    coco = YOLO(BASE_MODEL)
    ft_preds = predict(finetuned, images)
    coco_preds = predict(coco, images, classes=[COCO_PERSON])

    # ---- detection metrics, same code for all three ----
    detection = {
        "finetuned": detection_report(split_by_class(ft_preds, len(names)), gt, names),
        "coco_zero_shot": detection_report(coco_as_each_class(coco_preds, len(names)), gt, names),
        "predict_nothing": detection_report({}, gt, names),
    }
    detection["_notes"] = {
        "positive_class": POSITIVE,
        "precision_recall_at_conf": REPORT_CONF,
        "iou_for_precision_recall": 0.5,
        "coco_zero_shot": "COCO has no vest classes; every person box is scored as a candidate for each class.",
    }
    (ARTIFACTS / "detection_metrics.json").write_text(json.dumps(detection, indent=2))

    confusion = {
        "names": names + ["background"],
        "layout": "rows = predicted, columns = true",
        "conf": REPORT_CONF,
        "iou": MATRIX_IOU,
        "matrix": detection_confusion(ft_preds, gt, len(names)),
    }
    (ARTIFACTS / "detection_confusion.json").write_text(json.dumps(confusion, indent=2))

    # Cross-check our evaluator against Ultralytics' own validator.
    official = finetuned.val(data=str(data_yaml), split="test", imgsz=IMGSZ,
                             plots=False, verbose=False, project=str(ROOT / "runs"),
                             name="test_eval", exist_ok=True)
    detection["finetuned"]["ultralytics_val"] = {
        "map50": float(official.box.map50), "map50_95": float(official.box.map),
    }
    (ARTIFACTS / "detection_metrics.json").write_text(json.dumps(detection, indent=2))

    # ---- per-image decision scores ----
    y = np.array([len(gt[img][pos]) > 0 for img in images], dtype=int)
    scores = pd.DataFrame({
        "image": [Path(img).name for img in images],
        "y": y,
        "finetuned": image_scores(ft_preds, images, pos),
        "coco_zero_shot": image_scores(coco_preds, images, None),
        "predict_nothing": 0.0,
    })
    scores.to_csv(ARTIFACTS / "test_scores.csv", index=False)

    # ---- costing ----
    raw_costs = json.loads(costing.COSTS_FILE.read_text())
    costs = costing.load_costs()
    sweeps = []
    for model in ("finetuned", "coco_zero_shot", "predict_nothing"):
        s = costing.sweep(scores["y"], scores[model], costs)
        s.insert(0, "model", model)
        sweeps.append(s)
    pd.concat(sweeps).to_csv(ARTIFACTS / "sweep.csv", index=False)

    alt = costing.alternatives(scores["y"], scores["finetuned"], costs)
    baseline_alone = {
        m: float(pd.concat(sweeps).query("model == @m")["cost_per_1000"].min())
        for m in ("coco_zero_shot", "predict_nothing")
    }
    operating = {
        "costs_used": costs,
        "costs_are_placeholders": "_status" in raw_costs,
        "test_images": int(len(y)),
        "test_positive_images": int(y.sum()),
        "chosen_single_threshold": alt["model_alone"]["t_low"],
        "chosen_band": {"t_low": alt["model_plus_review"]["t_low"],
                        "t_high": alt["model_plus_review"]["t_high"]},
        "alternatives": alt,
        "baselines_cheapest_cost_per_1000": baseline_alone,
    }
    (ARTIFACTS / "operating_point.json").write_text(json.dumps(operating, indent=2, default=float))

    (ARTIFACTS / "train_info.json").write_text(json.dumps({
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "base_model": BASE_MODEL,
        "training": train_info,
        "smoke_run": SMOKE,
    }, indent=2))

    # ---- summary ----
    print("\n=== Detection on the test split ===")
    for model, rep in detection.items():
        if model.startswith("_"):
            continue
        print(f"{model:<16} mAP50 {rep['map50']:.3f}  mAP50-95 {rep['map50_95']:.3f}")
        for cname, m in rep["per_class"].items():
            print(f"    {cname:<16} AP50 {m['ap50']:.3f}  P {m['precision']:.3f}  "
                  f"R {m['recall']:.3f}  (n={m['n_gt']})")
    print(f"Ultralytics val cross-check (finetuned): {detection['finetuned']['ultralytics_val']}")

    print("\n=== Cost per 1,000 frames ===")
    for name, r in alt.items():
        extra = ""
        if "t_low" in r:
            extra = f"  thresholds {r['t_low']:.2f}/{r['t_high']:.2f}"
        print(f"{name:<26} EUR {r['cost_per_1000']:>10.2f}   review {r['review_share']:6.1%}{extra}")
    if "_status" in raw_costs:
        print("\nWARNING: costs.json still holds PLACEHOLDER numbers. "
              "Put your own figures in it, then re-run with SKIP_TRAIN=1.")
    print(f"\nWeights: {WEIGHTS.relative_to(ROOT)}   Artifacts: {ARTIFACTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
