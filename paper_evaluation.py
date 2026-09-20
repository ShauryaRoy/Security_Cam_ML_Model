#!/usr/bin/env python3
"""
Reproducible paper-evaluation utility for ICACRS camera-ready revision.

It uses the surviving best.pt checkpoint plus the original Construction Site
Safety dataset restored from Git history. It deliberately does not invent
face-recognition, Firebase, ESP32, or security measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
]
COMPLIANCE_CLASS_IDS = {0, 1, 2, 3, 4, 5, 7}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--weights", type=Path, default=Path("best.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("datasets/css-data"))
    p.add_argument("--out", type=Path, default=Path("paper_eval"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def jsafe(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsafe(v) for v in x]
    try:
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return str(x)


def dump_json(path, obj):
    Path(path).write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def dump_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and not fieldnames:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def images_in(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in exts)


def reconstruct_test(repo, data_dir, out):
    src_i = data_dir / "test" / "images"
    src_l = data_dir / "test" / "labels"
    dst = out / "reconstructed_test"
    di, dl = dst / "images", dst / "labels"
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)

    local = {p.stem: p for p in images_in(src_i)}
    root = {p.stem: p for p in images_in(repo)}
    missing = []

    for label in sorted(src_l.glob("*.txt")):
        src = local.get(label.stem) or root.get(label.stem)
        if src is None:
            missing.append(label.stem)
            continue
        shutil.copy2(src, di / src.name)
        shutil.copy2(label, dl / label.name)

    if missing:
        raise RuntimeError("Missing test images for labels: " + ", ".join(missing))
    return dst


def make_yaml(data_dir, test_dir, out):
    p = out / "paper_eval_data.yaml"
    txt = (
        f"path: {data_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"test: {test_dir.resolve().as_posix()}/images\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    p.write_text(txt, encoding="utf-8")
    return p


def labels_for(path, w, h):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        s = line.split()
        if len(s) < 5:
            continue
        cls = int(float(s[0]))
        xc, yc, bw, bh = map(float, s[1:5])
        out.append((cls, np.array([
            (xc - bw / 2) * w, (yc - bh / 2) * h,
            (xc + bw / 2) * w, (yc + bh / 2) * h,
        ], dtype=float)))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw * ih
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    den = aa + ab - inter
    return inter / den if den else 0.0


def match(preds, gts, threshold, match_iou, allowed=None):
    if allowed is not None:
        preds = [p for p in preds if p[0] in allowed]
        gts = [g for g in gts if g[0] in allowed]
    preds = sorted([p for p in preds if p[1] >= threshold],
                   key=lambda x: x[1], reverse=True)
    used = set()
    tp = fp = 0

    for cls, conf, box in preds:
        cand = [(iou(box, gb), gi) for gi, (gc, gb) in enumerate(gts)
                if gc == cls and gi not in used]
        biou, bgi = max(cand, default=(0.0, None))
        if bgi is not None and biou >= match_iou:
            used.add(bgi)
            tp += 1
        else:
            fp += 1

    fn = len(gts) - len(used)
    return tp, fp, fn


def prf(tp, fp, fn, threshold):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return {"threshold": threshold, "TP": tp, "FP": fp, "FN": fn,
            "precision": p, "recall": r, "F1": f1}


def extract_metrics(metrics, model):
    names = getattr(metrics, "names", None) or getattr(model, "names", {})
    box = metrics.box
    p = np.asarray(getattr(box, "p", []), float)
    r = np.asarray(getattr(box, "r", []), float)
    ap50 = np.asarray(getattr(box, "ap50", []), float)
    ap = np.asarray(getattr(box, "ap", []), float)

    rows = []
    n = max(len(p), len(r), len(ap50), len(ap))
    for i in range(n):
        pi = float(p[i]) if i < len(p) else math.nan
        ri = float(r[i]) if i < len(r) else math.nan
        nm = names.get(i, CLASS_NAMES[i]) if isinstance(names, dict) else CLASS_NAMES[i]
        rows.append({
            "class_id": i, "class": nm,
            "precision": pi, "recall": ri,
            "F1": 2*pi*ri/(pi+ri) if pi+ri else 0.0,
            "AP50": float(ap50[i]) if i < len(ap50) else math.nan,
            "AP50_95": float(ap[i]) if i < len(ap) else math.nan,
        })

    mp = float(np.nanmean(p)) if len(p) else math.nan
    mr = float(np.nanmean(r)) if len(r) else math.nan
    agg = {
        "precision_mean": mp,
        "recall_mean": mr,
        "F1_from_mean_P_R": 2*mp*mr/(mp+mr) if mp+mr else math.nan,
        "mAP50": float(np.nanmean(ap50)) if len(ap50) else math.nan,
        "mAP50_95": float(np.nanmean(ap)) if len(ap) else math.nan,
    }
    return rows, agg


def annotation_counts(data_dir, test_dir):
    rows = []
    for split, folder in [
        ("train", data_dir/"train"/"labels"),
        ("val", data_dir/"valid"/"labels"),
        ("test", test_dir/"labels"),
    ]:
        c = Counter()
        files = list(folder.glob("*.txt"))
        empty = 0
        for p in files:
            ls = [x for x in p.read_text(encoding="utf-8", errors="ignore").splitlines()
                  if x.strip()]
            if not ls:
                empty += 1
            for line in ls:
                try:
                    c[int(float(line.split()[0]))] += 1
                except Exception:
                    pass
        for i, name in enumerate(CLASS_NAMES):
            rows.append({
                "split": split, "class_id": i, "class": name,
                "annotation_instances": c[i],
                "label_files_in_split": len(files),
                "empty_label_files_in_split": empty,
            })
    return rows



def transform_rotated(arr, gts, angle_deg):
    import cv2
    h, w = arr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    out = []
    for cls, b in gts:
        x1, y1, x2, y2 = map(float, b)
        pts = np.array([[x1, y1, 1.0], [x2, y1, 1.0],
                        [x2, y2, 1.0], [x1, y2, 1.0]], dtype=float)
        t = pts @ M.T
        nx1 = max(0.0, float(t[:, 0].min()))
        ny1 = max(0.0, float(t[:, 1].min()))
        nx2 = min(float(w), float(t[:, 0].max()))
        ny2 = min(float(h), float(t[:, 1].max()))
        if nx2 > nx1 and ny2 > ny1:
            out.append((cls, np.array([nx1, ny1, nx2, ny2], dtype=float)))
    return rotated, out


def robustness_evaluation(model, test_dir, imgsz, nms_iou, match_iou, device,
                          confidence=0.40):
    import cv2

    conditions = {
        "normal": [],
        "low_illumination_50pct": [],
        "partial_head_occlusion": [],
        "pose_rotation_10deg": [],
        "crowded_ge_3_persons": [],
    }

    imgs = images_in(test_dir / "images")
    for idx, img in enumerate(imgs):
        arr = cv2.imread(str(img))
        if arr is None:
            continue
        h, w = arr.shape[:2]
        gts = labels_for(test_dir / "labels" / f"{img.stem}.txt", w, h)
        person_boxes = [b for c, b in gts if c == 5]

        conditions["normal"].append((arr, gts))

        low = np.clip(arr.astype(np.float32) * 0.50, 0, 255).astype(np.uint8)
        conditions["low_illumination_50pct"].append((low, gts))

        occ = arr.copy()
        for pb in person_boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in pb]
            y_occ2 = y1 + max(1, int(round((y2 - y1) * 0.35)))
            x1c, x2c = max(0, x1), min(w - 1, x2)
            y1c, y2c = max(0, y1), min(h - 1, y_occ2)
            if x2c > x1c and y2c > y1c:
                cv2.rectangle(occ, (x1c, y1c), (x2c, y2c), (0, 0, 0), -1)
        conditions["partial_head_occlusion"].append((occ, gts))

        angle = 10.0 if idx % 2 == 0 else -10.0
        rot, rgts = transform_rotated(arr, gts, angle)
        conditions["pose_rotation_10deg"].append((rot, rgts))

        if len(person_boxes) >= 3:
            conditions["crowded_ge_3_persons"].append((arr, gts))

    rows = []
    for name, samples in conditions.items():
        TP = FP = FN = 0
        for arr, gts in samples:
            res = model.predict(source=arr, imgsz=imgsz, conf=confidence,
                                iou=nms_iou, device=device, verbose=False)[0]
            preds = []
            if res.boxes is not None:
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                confs = res.boxes.conf.detach().cpu().numpy()
                clss = res.boxes.cls.detach().cpu().numpy().astype(int)
                preds = [(int(c), float(cf), b.astype(float))
                         for b, cf, c in zip(xyxy, confs, clss)]
            tp, fp, fn = match(preds, gts, confidence, match_iou,
                               COMPLIANCE_CLASS_IDS)
            TP += tp
            FP += fp
            FN += fn
        row = prf(TP, FP, FN, confidence)
        row = {"condition": name, "images": len(samples), **row}
        rows.append(row)
    return rows


def main():
    a = args()
    import cv2
    import psutil
    import torch
    import ultralytics
    from ultralytics import YOLO

    repo = a.repo.resolve()
    weights = (repo/a.weights).resolve() if not a.weights.is_absolute() else a.weights
    data_dir = (repo/a.data_dir).resolve() if not a.data_dir.is_absolute() else a.data_dir
    out = (repo/a.out).resolve() if not a.out.is_absolute() else a.out
    out.mkdir(parents=True, exist_ok=True)

    test_dir = reconstruct_test(repo, data_dir, out)
    yaml = make_yaml(data_dir, test_dir, out)
    model = YOLO(str(weights))

    ckpt = getattr(model, "ckpt", None) or {}
    dump_json(out/"checkpoint_metadata.json", {
        "weights_bytes": weights.stat().st_size,
        "task": getattr(model, "task", None),
        "names": getattr(model, "names", None),
        "checkpoint_date": ckpt.get("date") if isinstance(ckpt, dict) else None,
        "checkpoint_version": ckpt.get("version") if isinstance(ckpt, dict) else None,
        "train_args": ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {},
        "model_yaml": getattr(getattr(model, "model", None), "yaml", None),
    })

    dump_json(out/"environment.json", {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })

    dump_csv(out/"class_annotation_counts.csv", annotation_counts(data_dir, test_dir))

    aggregate = []
    for split, filename in [("val", "val_per_class.csv"),
                            ("test", "test_per_class.csv")]:
        metrics = model.val(
            data=str(yaml), split=split, imgsz=a.imgsz,
            conf=0.001, iou=a.nms_iou, device=a.device,
            plots=True, project=str(out/"ultralytics"),
            name=split, exist_ok=True, verbose=False,
        )
        rows, agg = extract_metrics(metrics, model)
        dump_csv(out/filename, rows)
        aggregate.append({"split": split, **agg})
    dump_csv(out/"aggregate_metrics.csv", aggregate)

    # Cache test predictions once at the minimum threshold.
    cached = []
    for img in images_in(test_dir/"images"):
        arr = cv2.imread(str(img))
        if arr is None:
            continue
        h, w = arr.shape[:2]
        gts = labels_for(test_dir/"labels"/f"{img.stem}.txt", w, h)
        res = model.predict(
            source=arr, imgsz=a.imgsz, conf=min(a.thresholds),
            iou=a.nms_iou, device=a.device, verbose=False,
        )[0]
        preds = []
        if res.boxes is not None:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            clss = res.boxes.cls.detach().cpu().numpy().astype(int)
            preds = [(int(c), float(cf), b.astype(float))
                     for b, cf, c in zip(xyxy, confs, clss)]
        cached.append((preds, gts))

    all_rows, compliance_rows = [], []
    for t in a.thresholds:
        for allowed, target in [(None, all_rows),
                                (COMPLIANCE_CLASS_IDS, compliance_rows)]:
            TP = FP = FN = 0
            for preds, gts in cached:
                tp, fp, fn = match(preds, gts, t, a.match_iou, allowed)
                TP += tp; FP += fp; FN += fn
            target.append(prf(TP, FP, FN, t))
    dump_csv(out/"threshold_sweep_all_classes.csv", all_rows)
    dump_csv(out/"threshold_sweep_compliance_classes.csv", compliance_rows)

    robust_rows = robustness_evaluation(
        model, test_dir, a.imgsz, a.nms_iou, a.match_iou, a.device, confidence=0.40
    )
    dump_csv(out/"robustness_metrics.csv", robust_rows)


    # Reviewer robustness table: make a manifest, but do not invent semantic
    # labels such as occlusion or pose. Those require manual visual assignment.
    manifest = []
    for img in images_in(test_dir/"images"):
        arr = cv2.imread(str(img))
        if arr is None:
            continue
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        h, w = arr.shape[:2]
        gts = labels_for(test_dir/"labels"/f"{img.stem}.txt", w, h)
        manifest.append({
            "image": img.name,
            "mean_gray": round(float(gray.mean()), 2),
            "gray_std": round(float(gray.std()), 2),
            "gt_person_count": sum(1 for c, _ in gts if c == 5),
            "condition": "",
            "notes": "",
        })
    dump_csv(out/"robustness_manifest_template.csv", manifest,
             ["image", "mean_gray", "gray_std", "gt_person_count",
              "condition", "notes"])

    print("Paper evaluation complete:", out)


if __name__ == "__main__":
    main()
