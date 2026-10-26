# Assignment 2 Report

- **Name**: Juan Miro-Quesada
- **Student ID**: 19858
- **Email**: jmiroquesada.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Business problem, and the cost of being wrong

Decision. On a construction site, cameras capture frames of the work area. For each frame the system decides whether it shows at least one worker without a high-visibility safety vest. The site's safety technician acts on it: an alert means they go and get the worker to put a vest on.

Unit of decision. One camera frame. Positive = the frame contains at least one no_safety_vest worker.

Two kinds of mistake.

Missed violation (false negative): a worker stays unprotected and less visible to vehicles and machinery, and the company is exposed to a labour inspection fine.
False alarm (false positive): the technician walks over to a worker who is already wearing a vest, which costs a few minutes of their time.

Asymmetry. A miss costs about €15 and a false alarm about €2, roughly 7× (figures justified in Cost matrix below). A miss is more severe than a false alarm because the potential consequence of a missing vest outweights the small cost of checking. An infraction, violation, or injury can cost tens of thousand whereas the few minutes it takes to check is significantly cheaper.

## Dataset

Roboflow Universe: workspace roboflow-universe-projects, project safety-vests, version 14: https://universe.roboflow.com/roboflow-universe-projects/safety-vests/14
Used as published, not forked. Licence CC BY 4.0.
Task: object detection.
One image: a photo of one or more workers, mostly on construction or industrial sites, with a box around each person labelled by whether they wear a vest.
Classes (2): no_safety_vest (positive) and safety_vest.
There is some Resemblence because the right objects appear in the images. However, it is more “Serving” because the labels and images match what the model will actually face and decide.

## What the dataset's composition implies

Train: 13,640 images; 7,167 no_safety_vest boxes and 22,319 safety_vest boxes; 3,390 images contain at least one no_safety_vest.
Valid: 779 images; 361 no_safety_vest boxes and 1,287 safety_vest boxes; 189 images contain at least one no_safety_vest.
Test: 390 images; 226 no_safety_vest boxes and 662 safety_vest boxes; 108 images contain at least one no_safety_vest.
Augmentation is in train only. The project has about 3,900 original images. After taking out valid and test, about 2,700 originals remain for training, and train has 13,640 images, so about 5 augmented copies of each. Valid and test are original images, so the test numbers are not inflated by near-duplicates of training images.
Class balance: safety_vest outnumbers no_safety_vest about 3 : 1 in every split.
Test share of positives: 108 of 390 test images (27.7%) contain a violation.
Object size: median box area is 6.2% of the image for no_safety_vest and 5.6% for safety_vest. About 7.5% of boxes cover less than 1% of the image (distant workers).
Does 27.7% look like production?

Almost certainly not. A real site where more than a quarter of camera frames showed someone without a vest would be an extreme safety failure, and a labour inspector would likely stop the work. A plausible real share is much lower; you assumed 10% in your costs.

Why the test set has so many: dataset builders collect images because they contain the thing to be detected, so violations are deliberately over-represented. That’s useful for training, since the model needs plenty of examples of the positive class, but it makes the test set unrepresentative of real frequency.

Why it matters

Precision depends on the base rate. At a lower real violation rate, most frames are clean, so the same model produces proportionally more false alarms among its alerts. Your measured no_safety_vest precision of 0.77 is therefore optimistic for a real site.
Recall doesn’t depend on the base rate. The share of real violations caught is a property of how the model sees each violation, not of how common they are. Recall carries over better than precision.
That’s why you re-weighted the costs to a 10% base rate instead of using the test set’s share. Without that, the cost of every option would be computed for a site that doesn’t exist.

What you can’t conclude from these results

How the model performs on your actual site’s cameras. If the images differ from fixed, high, wide-angle site footage (as you checked for the dataset question), real performance will likely be lower.
The real false-alarm rate per day. It depends on the true base rate, which you assumed, not measured.
Precise per-class numbers. With only 108 positive test images, each image moves image-level recall by about 1 point, so treat 0.88 as “roughly 0.85–0.9”, not exact.
That the chosen thresholds are optimal for real use. They’re the cheapest given your assumed costs and base rate. If those are wrong, the best thresholds move.
Behaviour over time. Each test image is independent, but on a real camera the same worker appears in many consecutive frames, and the test set says nothing about that.

## Finetuning and baselines

Setup:

Model: yolo11n (detection), finetuned from the COCO-pretrained checkpoint.
Training: Google Colab, NVIDIA T4 GPU. 8 epochs, image size 640, batch 32, seed 0, early-stopping patience 10, Ultralytics default optimiser (AdamW, chosen automatically) and default augmentation. 8 epochs was chosen to keep training around half an hour.
Weights: weights/best.pt.

All three models were scored by the same evaluation code on the same 390 test images:

Finetuned yolo11n: mAP50 0.879, mAP50-95 0.530; no_safety_vest AP50 0.868, safety_vest AP50 0.890.
COCO yolo11n zero-shot (every "person" box offered as each class): mAP50 0.035, mAP50-95 0.008; no_safety_vest AP50 0.015, safety_vest AP50 0.054.
Trivial, predict nothing: mAP50 0.000, mAP50-95 0.000; AP50 0.000 for both classes.

Cross-check: Ultralytics' own validator gives the finetuned model mAP50 0.888 and mAP50-95 0.538 on the same split, which is consistent.

## Per-class evaluation and failure modes

Precision and recall are at confidence 0.25 and IoU 0.5. The positive class is no_safety_vest.

no_safety_vest (positive): 226 test boxes; AP50 0.868; AP50-95 [fill from dashboard]; precision 0.768; recall 0.881.
safety_vest: 662 test boxes; AP50 0.890; AP50-95 [fill from dashboard]; precision 0.846; recall 0.840.

At this setting, about 27 of the 226 no-vest workers are missed, and about 60 of the roughly 260 no_safety_vest predictions are wrong.

## Cost matrix and operating point

All figures are per camera frame:

Missed violation (FN): €15. About €10 expected fine + about €5 expected accident cost (below).
False alarm (FP): €2. About 7 minutes of technician and worker time at €17/h.
Acting on a caught violation (TP): €2. The same visit.
Correct pass (TN): €0. Nothing happens.
Review: €0.047 per frame. 10 s at €17/h.
Base rate: 10% of frames. About 3.5% of workers without a vest × about 3 workers per frame: 1 − 0.965³ ≈ 0.10.

Where the figures come from:

Hourly cost: a Spanish prevention technician earns about €23,400 gross a year; with about 31% employer social security that is about €30,650, over about 1,760 hours a year, ≈ €17/h.
Fine: failing to ensure protective equipment is used, when that creates a serious risk, is a serious ("grave") breach under LISOS art. 12; the lowest fine is about €2,451. I assume about a 1-in-250 chance per day that an inspector sees an uncorrected violation: €2,451 / 250 ≈ €10.
Accident: the EU per-case cost of a work injury ranges from about €37,860 to €73,410 (study of five EU countries). I use about €40,000 and an extra accident probability of about 1 in 8,000 per uncorrected violation: ≈ €5.
Persistence assumption: a miss is assumed to leave the violation uncorrected for a while, because the model tends to miss the same worker in consecutive frames.

Sweep. The threshold on each frame's highest no_safety_vest confidence was swept from 0 to 1 in steps of 0.01, with cost re-weighted to the 10% base rate (not the test set's 27.7%):

Cheapest single threshold (model alone): 0.20, at €361.58 per 1,000 frames.
Cost curve with the chosen point marked: dashboard, "Thresholds & cost" tab.

Sensitivity: [fill in from your re-runs, for example: "With cost_fn = €5 the thresholds become … and … ; with €50 … . Model + review stops beating a person checking everything when cost_fn drops below €…."]

## What a person still has to check

Thresholds: below 0.02 the frame passes; at or above 0.80 an alert is sent; in between, a person reviews the frame. This is the cheapest pair from a sweep of both thresholds.

At 1,200 frames a day:

20.7% of frames are sent for review, about 248 frames a day.
At 10 s each that is about 41 minutes a day, or 0.35 reviewers at 2 review hours per person per day.

Assumption: a reviewer judges a frame correctly.

Do nothing (all pass): €1,500.00 per 1,000 frames (€1,800 a day); 0% sent to a person; no review time.
Person checks every frame: €247.22 per 1,000 frames (€297 a day); 100% sent to a person; 3.3 h of review a day (1.7 reviewers).
Model alone (threshold 0.20): €361.58 per 1,000 frames (€434 a day); 0% sent to a person; no review time.
Model + review (0.02 / 0.80): €221.83 per 1,000 frames (€266 a day); 20.7% sent to a person; 41 min of review a day (0.35 reviewers).

## Limitations & next steps

The test split is small for the positive class (108 images).
Most images probably don't look like fixed site cameras.
Costs assume one frame is independent of the next.
The reviewer is assumed to be always right.
Small or distant workers are about 7.5% of boxes.
Training was only 8 epochs.

## Generative AI use disclosure

Code: Claude helped with the code and the Colab training cells.
Cost figures: Claude researched reference figures (Spanish LISOS fines, EU per-case injury costs, technician salaries) and suggested starting values for every figure in costs.json. 
This report: Claude filled in the numbers and tables from my pipeline's output.

