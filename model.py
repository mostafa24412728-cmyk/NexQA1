"""
╔══════════════════════════════════════════════════════════════════════╗
║         Wood Surface Defect Detection — YOLOv8 Pipeline             ║
║         Dataset: 4000 images | 8 defect classes | YOLO format       ║
║         ✅ يعرض اسم العيب بوضوح على كل bounding box                ║
╚══════════════════════════════════════════════════════════════════════╝

Classes:
  0: Quartzity      1: Live_Knot     2: Marrow        3: resin
  4: Dead_Knot      5: knot_with_crack  6: Knot_missing  7: Crack

────────────────────────────────────────────────────────────────────────
 HOW TO PREPARE YOUR DATASET FROM THE ARCHIVE
────────────────────────────────────────────────────────────────────────
 After extracting your archive, run prepare_dataset() first.
 Expected archive structure:
   Images/          ← your raw images (.jpg / .png)
   Bounding Boxes - YOLO Format/   ← matching .txt label files

 The script will auto-split into train/val/test and build the
 correct folder structure that YOLOv8 expects.
────────────────────────────────────────────────────────────────────────
"""

# pip install ultralytics opencv-python matplotlib seaborn pyyaml tqdm

import os
import shutil
import random
import yaml
import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────
    "raw_images_dir":  "Images - 1\\Images - 1",                        # ← your raw images folder
    "raw_labels_dir":  "Bounding Boxes - YOLO Format - 1\\Bounding Boxes - YOLO Format - 1",  # ← your labels folder
    "dataset_root":    "dataset",                       # prepared dataset (auto-built)
    "output_dir":      "runs/wood_defects",

    # ── Classes  (ORDER MUST MATCH YOUR LABEL FILES) ───────────────────
    "class_names": [
        "Quartzity",        # 0
        "Live_Knot",        # 1
        "Marrow",           # 2
        "resin",            # 3
        "Dead_Knot",        # 4
        "knot_with_crack",  # 5
        "Knot_missing",     # 6
        "Crack",            # 7
    ],
    "class_names_ar": [
        "كوارتزيت (Quartzity)",
        "عقدة حية (Live Knot)",
        "نخاع (Marrow)",
        "راتنج (Resin)",
        "عقدة ميتة (Dead Knot)",
        "عقدة مشققة (Knot with Crack)",
        "عقدة مفقودة (Knot missing)",
        "شرخ (Crack)",
    ],

    # ── Split ratios ───────────────────────────────────────────────────
    "train_ratio": 0.70,
    "val_ratio":   0.20,
    "test_ratio":  0.10,

    # ── Model ──────────────────────────────────────────────────────────
    "model_size": "runs/wood_defects/train/weights/best.pt", # Continue from Round 2
    "img_size":   416,         # Increased for better detail detection
    "pretrained": True,

    # ── Training (Round 3: Fine-Tuning) ────────────────────────────────
    "epochs":       40,        # Extended for maximum refinement
    "batch_size":   8,         # Adjusted for 416 size on CPU
    "patience":     30,        # More patience for early stopping
    "lr0":          0.002,     # Lowered for stable fine-tuning
    "lrf":          0.01,
    "weight_decay": 0.0005,
    "warmup_epochs":5,         # Longer warmup
    "workers":      2,         # Reduced for CPU

    # ── Augmentation (Aggressive for real-world mobile photos) ─────────
    "augment": {
        "hsv_h": 0.015, "hsv_s": 0.7,  "hsv_v": 0.4,
        "degrees": 15.0,      # More rotation for handheld mobile photos
        "translate": 0.1, 
        "scale": 0.6,         # Better scaling detection
        "shear": 2.0,         # Angle distortion
        "perspective": 0.001, # 3D perspective
        "flipud": 0.5,        # Vertical flip for wood textures
        "fliplr": 0.5,
        "mosaic": 1.0, 
        "mixup": 0.2,         # Better handling of overlapping defects
        "copy_paste": 0.1,    # Instance segmentation trick for detection
    },

    # ── Inference ──────────────────────────────────────────────────────
    "conf_threshold": 0.25,
    "iou_threshold":  0.45,

    # ── Misc ───────────────────────────────────────────────────────────
    "seed": 42,
}

# One distinct colour per class  (RGB)
CLASS_COLORS_RGB = [
    (230, 57,  70),   # 0  Quartzity       — red
    (69,  123, 157),  # 1  Live_Knot        — steel blue
    (80,  200, 120),  # 2  Marrow           — green
    (241, 196, 15),   # 3  resin            — amber
    (142, 68,  173),  # 4  Dead_Knot        — purple
    (231, 76,  60),   # 5  knot_with_crack  — coral
    (26,  188, 156),  # 6  Knot_missing     — teal
    (243, 156, 18),   # 7  Crack            — orange
]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def log(msg, symbol="►"):
    print(f"\n{symbol}  {msg}")


# ─────────────────────────────────────────────
#  STEP 0 ▸ PREPARE DATASET FROM YOUR ARCHIVE
# ─────────────────────────────────────────────

def prepare_dataset(cfg: dict):
    """
    Takes your raw Images/ and Labels/ folders and builds the
    standard YOLO directory tree:

        dataset/
          images/train|val|test/
          labels/train|val|test/

    Safe to re-run — skips if dataset/ already exists.
    """
    log("Preparing dataset …", "📂")

    out = Path(cfg["dataset_root"])
    
    # Force re-preparation if folders are empty
    train_img_path = out / "images" / "train"
    if train_img_path.exists() and any(train_img_path.iterdir()):
        print("  ✅  Dataset already prepared and populated — skipping.")
        return
    elif out.exists():
        print("  ⚠  Dataset folder exists but is empty or incomplete. Cleaning up...")
        shutil.rmtree(out)

    img_dir = Path(cfg["raw_images_dir"])
    lbl_dir = Path(cfg["raw_labels_dir"])

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Images folder not found: {img_dir.resolve()}\n"
            "Please extract your archive and set 'raw_images_dir' in CONFIG."
        )
    if not lbl_dir.exists():
        raise FileNotFoundError(
            f"Labels folder not found: {lbl_dir.resolve()}\n"
            "Please extract your archive and set 'raw_labels_dir' in CONFIG."
        )

    # Collect all images that have a matching label
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    all_imgs = []
    for ext in exts:
        all_imgs.extend(img_dir.glob(ext))

    paired = []
    for img in all_imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            paired.append((img, lbl))
        else:
            print(f"  ⚠  No label for {img.name} — skipped")

    random.shuffle(paired)
    n       = len(paired)
    n_train = int(n * cfg["train_ratio"])
    n_val   = int(n * cfg["val_ratio"])

    splits = {
        "train": paired[:n_train],
        "val":   paired[n_train : n_train + n_val],
        "test":  paired[n_train + n_val :],
    }

    for split, items in splits.items():
        img_out = ensure_dir(out / "images" / split)
        lbl_out = ensure_dir(out / "labels" / split)
        for img_path, lbl_path in tqdm(items, desc=f"  Copying {split}"):
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    print(f"\n  ✅  Dataset ready:")
    for split, items in splits.items():
        print(f"      {split:5s} → {len(items)} pairs")


# ─────────────────────────────────────────────
#  STEP 1 ▸ VALIDATE DATASET
# ─────────────────────────────────────────────

def validate_dataset(cfg: dict) -> dict:
    log("Validating dataset …", "🔍")
    root      = Path(cfg["dataset_root"])
    n_classes = len(cfg["class_names"])
    stats     = {}

    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        class_counts = [0] * n_classes
        bad_files    = []
        total_boxes  = 0

        for lbl_path in tqdm(labels, desc=f"  {split}", leave=False):
            try:
                with open(lbl_path) as f:
                    lines = f.read().strip().splitlines()
                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        bad_files.append(str(lbl_path)); break
                    cls = int(parts[0])
                    if 0 <= cls < n_classes:
                        class_counts[cls] += 1
                        total_boxes += 1
                    else:
                        bad_files.append(str(lbl_path))
            except Exception as e:
                bad_files.append(f"{lbl_path}: {e}")

        stats[split] = {
            "images": len(images), "labels": len(labels),
            "total_boxes": total_boxes,
            "class_counts": class_counts, "bad_files": len(bad_files),
        }
        print(f"  [{split}]  images={len(images)}  labels={len(labels)}"
              f"  boxes={total_boxes}  bad={len(bad_files)}")

    # Print per-class summary
    print("\n  Class annotation summary (train):")
    if "train" in stats:
        for i, (name, count) in enumerate(
                zip(cfg["class_names"], stats["train"]["class_counts"])):
            bar = "█" * min(40, count // 10) if count else ""
            print(f"    {i}  {name:<20}  {count:>5}  {bar}")

    return stats


# ─────────────────────────────────────────────
#  STEP 2 ▸ CREATE YAML
# ─────────────────────────────────────────────

def create_yaml(cfg: dict, out_dir: Path) -> Path:
    log("Creating dataset YAML …", "📄")
    root = Path(cfg["dataset_root"]).resolve()

    data = {
        "path":  str(root),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(cfg["class_names"]),
        "names": cfg["class_names"],
    }
    if (root / "images" / "test").exists():
        data["test"] = "images/test"

    yaml_path = out_dir / "wood_defects.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved → {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────
#  STEP 3 ▸ TRAIN
# ─────────────────────────────────────────────

def train(cfg: dict, yaml_path: Path, out_dir: Path) -> YOLO:
    log("Starting YOLOv8 training …", "🚀")
    set_seed(cfg["seed"])

    # If model_size is a path, load it directly, else add .pt
    model_path = cfg['model_size']
    if not os.path.exists(model_path) and not model_path.endswith(".pt"):
        model_path = f"{model_path}.pt"
        
    model = YOLO(model_path)
    aug   = cfg["augment"]

    model.train(
        data=str(yaml_path), epochs=cfg["epochs"],
        imgsz=cfg["img_size"], batch=cfg["batch_size"],
        patience=cfg["patience"], lr0=cfg["lr0"], lrf=cfg["lrf"],
        weight_decay=cfg["weight_decay"], warmup_epochs=cfg["warmup_epochs"],
        workers=cfg["workers"], device="cpu",
        project=str(out_dir), name="train", exist_ok=True,
        seed=cfg["seed"],
        hsv_h=aug["hsv_h"], hsv_s=aug["hsv_s"], hsv_v=aug["hsv_v"],
        degrees=aug["degrees"], translate=aug["translate"],
        scale=aug["scale"], flipud=aug["flipud"], fliplr=aug["fliplr"],
        mosaic=aug["mosaic"], mixup=aug["mixup"],
        save=True, save_period=10, plots=True, verbose=True,
        # freeze=10,           # Unfreezing for Round 2 to refine accuracy
    )

    best = out_dir / "train" / "weights" / "best.pt"
    log(f"Training complete. Best weights → {best}", "✅")
    return YOLO(str(best))


# ─────────────────────────────────────────────
#  STEP 4 ▸ EVALUATE
# ─────────────────────────────────────────────

def evaluate(model: YOLO, cfg: dict, yaml_path: Path, out_dir: Path) -> dict:
    log("Evaluating on validation set …", "📊")
    metrics = model.val(
        data=str(yaml_path), imgsz=cfg["img_size"], batch=cfg["batch_size"],
        conf=cfg["conf_threshold"], iou=cfg["iou_threshold"],
        device="cpu", project=str(out_dir), name="val",
        exist_ok=True, plots=True, verbose=True,
    )

    print("\n" + "─" * 50)
    print(f"  mAP@50    : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95 : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print("─" * 50)

    return {
        "mAP50": metrics.box.map50, "mAP50_95": metrics.box.map,
        "precision": metrics.box.mp, "recall": metrics.box.mr,
    }


# ─────────────────────────────────────────────
#  STEP 5 ▸ INFERENCE  ← الجزء المهم
# ─────────────────────────────────────────────

def draw_predictions(image: np.ndarray, boxes, cfg: dict) -> np.ndarray:
    """
    ✅ يرسم bounding box ويكتب اسم العيب بوضوح
       مثال:  Dead_Knot  87%
    """
    img = image.copy()

    for box in boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── اسم العيب + نسبة الثقة ──────────────────────────────────
        defect_name = cfg["class_names_ar"][cls_id]       # ← اسم العيب بالعربي والإنجليزي
        label       = f"{defect_name} {confidence:.0%}"   # e.g. "Dead_Knot  87%"

        # لون مخصص لكل نوع عيب
        color_rgb = CLASS_COLORS_RGB[cls_id]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB → BGR

        # ── رسم المربع ──────────────────────────────────────────────
        thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

        # ── خلفية النص ──────────────────────────────────────────────
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thick = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        # مربع ملون خلف النص
        cv2.rectangle(img,
                      (x1, y1 - th - baseline - 8),
                      (x1 + tw + 8, y1),
                      color_bgr, -1)  # -1 = filled

        # ── كتابة اسم العيب ─────────────────────────────────────────
        cv2.putText(img, label,
                    (x1 + 4, y1 - baseline - 4),
                    font, font_scale,
                    (255, 255, 255),   # أبيض
                    font_thick, cv2.LINE_AA)

    return img


def predict_single_image(image_path: str, cfg: dict,
                          model: YOLO, save: bool = True) -> dict:
    """
    يحلل صورة واحدة ويرجع قاموس فيه أسماء العيوب المكتشفة.

    Returns:
        {
          "defects_found": ["Dead_Knot", "Crack"],
          "details": [
              {"defect": "Dead_Knot", "confidence": 0.87,
               "bbox": [x1, y1, x2, y2]},
              ...
          ]
        }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    preds = model.predict(
        source=img,
        imgsz=cfg["img_size"],
        conf=cfg["conf_threshold"],
        iou=cfg["iou_threshold"],
        device="cpu",
        verbose=False,
    )

    result = {"defects_found": [], "details": []}

    if preds[0].boxes is not None:
        for box in preds[0].boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox       = list(map(int, box.xyxy[0]))
            defect     = cfg["class_names"][cls_id]

            result["details"].append({
                "defect":     defect,
                "confidence": round(confidence, 3),
                "bbox":       bbox,
            })
            if defect not in result["defects_found"]:
                result["defects_found"].append(defect)

    # ── طباعة النتيجة بوضوح ─────────────────────────────────────────
    print("\n" + "═" * 45)
    print(f"  📸  Image : {Path(image_path).name}")
    print("═" * 45)

    if result["defects_found"]:
        print(f"  🪵  العيوب المكتشفة ({len(result['details'])} مكان):")
        for d in result["details"]:
            print(f"      ✦  {d['defect']:<20}  ثقة: {d['confidence']:.0%}"
                  f"   bbox: {d['bbox']}")
    else:
        print("  ✅  لم يُكتشف أي عيب (أو confidence أقل من threshold)")

    print("═" * 45)

    # ── حفظ الصورة المحللة ──────────────────────────────────────────
    if save:
        drawn      = draw_predictions(img, preds[0].boxes, cfg)
        out_path   = Path(image_path).stem + "_detected.jpg"
        cv2.imwrite(out_path, drawn)
        print(f"  💾  Saved → {out_path}")

    return result


def run_inference_batch(model: YOLO, cfg: dict, out_dir: Path,
                        source: str = None, max_vis: int = 6):
    """
    يشتغل على مجموعة صور ويحفظ نتائج مع أسماء العيوب.
    """
    log("Running inference …", "🎯")
    inf_dir = ensure_dir(out_dir / "inference")

    # اختيار الصور
    if source is None:
        val_dir = Path(cfg["dataset_root"]) / "images" / "val"
        imgs = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
        if not imgs:
            print("  ⚠  No val images found.")
            return
        source_list = random.sample(imgs, min(max_vis, len(imgs)))
    else:
        p = Path(source)
        source_list = list(p.glob("*.*")) if p.is_dir() else [p]

    annotated       = []
    all_defect_names = []

    print(f"\n  {'Image':<30}  Defects Found")
    print("  " + "─" * 60)

    for img_path in source_list[:max_vis]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        preds = model.predict(
            source=img, imgsz=cfg["img_size"],
            conf=cfg["conf_threshold"], iou=cfg["iou_threshold"],
            device="cpu", verbose=False,
        )

        # ── اسم العيب لكل detection ──────────────────────────────────
        detected = []
        if preds[0].boxes is not None:
            for box in preds[0].boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = cfg["class_names"][cls_id]
                detected.append(f"{name}({conf:.0%})")
                all_defect_names.append(name)

        defect_str = ", ".join(detected) if detected else "— none —"
        print(f"  {img_path.name:<30}  {defect_str}")

        drawn = draw_predictions(img, preds[0].boxes, cfg)
        cv2.imwrite(str(inf_dir / img_path.name), drawn)
        annotated.append(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))

    # ── ملخص الكل ─────────────────────────────────────────────────
    if all_defect_names:
        from collections import Counter
        freq = Counter(all_defect_names).most_common()
        print("\n  📊  Defect frequency summary:")
        for name, count in freq:
            bar = "█" * count
            print(f"      {name:<20}  {count:>3}  {bar}")

    if annotated:
        _save_grid(annotated, inf_dir / "prediction_grid.png", cfg)

    print(f"\n  ✅  Saved {len(annotated)} images → {inf_dir}")


def _save_grid(images, save_path, cfg):
    n    = len(images)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    fig.patch.set_facecolor("#0f1117")

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, img in zip(axes_flat, images):
        ax.imshow(img); ax.axis("off")
    for ax in axes_flat[len(images):]:
        ax.set_visible(False)

    handles = [
        patches.Patch(color=np.array(c) / 255, label=name)
        for c, name in zip(CLASS_COLORS_RGB, cfg["class_names"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               facecolor="#1a1d27", edgecolor="#3a3f55",
               labelcolor="#e0e0e0", fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Wood Surface Defect Detection — أسماء العيوب المكتشفة",
                 fontsize=13, color="#ffffff", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor="#0f1117")
    plt.close()


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    set_seed(CONFIG["seed"])
    out_dir = ensure_dir(CONFIG["output_dir"])

    print("\n" + "═" * 60)
    print("  🪵  Wood Surface Defect Detection  —  YOLOv8")
    print("═" * 60)

    # 0. Prepare dataset from your archive
    prepare_dataset(CONFIG)

    # 1. Validate
    stats = validate_dataset(CONFIG)

    # 2. YAML
    yaml_path = create_yaml(CONFIG, out_dir)

    # 3. Train
    model = train(CONFIG, yaml_path, out_dir)

    # 4. Evaluate
    evaluate(model, CONFIG, yaml_path, out_dir)

    # 5. Inference — يطبع اسم العيب لكل صورة
    run_inference_batch(model, CONFIG, out_dir, source=None, max_vis=6)

    print("\n" + "═" * 60)
    print("  🏁  Pipeline complete!")
    print(f"  📁  Outputs: {Path(CONFIG['output_dir']).resolve()}")
    print("═" * 60)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
#  QUICK PREDICT — صورة واحدة
# ─────────────────────────────────────────────
# from wood_defect_detection import CONFIG, predict_single_image
# from ultralytics import YOLO
#
# model  = YOLO("runs/wood_defects/train/weights/best.pt")
# result = predict_single_image("my_wood.jpg", CONFIG, model)
#
# # النتيجة:
# # ═════════════════════════════════════════════
# #   📸  Image : my_wood.jpg
# # ═════════════════════════════════════════════
# #   🪵  العيوب المكتشفة (2 مكان):
# #       ✦  Dead_Knot             ثقة: 87%   bbox: [...]
# #       ✦  Crack                 ثقة: 72%   bbox: [...]
# # ═════════════════════════════════════════════