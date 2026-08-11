# ===== Р”РћРћР‘РЈР§Р•РќРР• РњРћР”Р•Р›Р РћР РЈР–РРЇ: 16 РљР›РђРЎРЎРћР’ (СЃС‚Р°СЂС‹Рµ + gun + pistol + knife) =====
# РџРµСЂРµРґ Р·Р°РїСѓСЃРєРѕРј: РїРµСЂРµС‚Р°С‰Рё СЃРІРѕСЋ С‚РµРєСѓС‰СѓСЋ РјРѕРґРµР»СЊ weapon_detection.pt -> /content/weapon_detection.pt
# Runtime -> Change runtime type -> T4 GPU -> Ctrl+F9
# Р РµР·СѓР»СЊС‚Р°С‚: /content/weights/weapon_detection_16cls.pt

import os
import glob
import shutil
import random
import yaml

# ---------- 1. РЈСЃС‚Р°РЅРѕРІРєР° ----------
!pip install -q ultralytics
!pip install -q roboflow
from ultralytics import YOLO

# ---------- 2. GPU ----------
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("РќР•Рў GPU. Runtime -> Change runtime type -> T4 GPU")

# ---------- 3. РњРѕРґРµР»СЊ (РёС‰РµРј Р›Р®Р‘РћР™ .pt РІ Files, РєР°РєРѕРµ Р±С‹ РёРјСЏ РЅРё Р±С‹Р»Рѕ) ----------
pt_files = [p for p in glob.glob("/content/*.pt") if "yolo" not in os.path.basename(p).lower()]
pt_files += glob.glob("/content/drive/MyDrive/*.pt")
if not pt_files:
    print("РќРµ РЅР°С€С‘Р» .pt. РџСЂРѕРІРµСЂСЊ РїР°РЅРµР»СЊ Files. РР»Рё РІРїРёС€Рё РїСѓС‚СЊ СЃР°Рј:")
    CKPT = input("РџСѓС‚СЊ Рє РјРѕРґРµР»Рё (РЅР°РїСЂРёРјРµСЂ /content/my_model.pt): ").strip()
    assert os.path.exists(CKPT), f"Р¤Р°Р№Р» {CKPT} РЅРµ РЅР°Р№РґРµРЅ!"
else:
    CKPT = pt_files[0]
print("РњРѕРґРµР»СЊ:", CKPT)

# ---------- 4. РўСЂРё РґР°С‚Р°СЃРµС‚Р° РїРѕ API ----------
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_KEY")

rf.workspace("buildx").project("weapon-detection-7kro8").version(2).download(
    "yolov8", location="/content/old_data")          # 14 РєР»Р°СЃСЃРѕРІ
rf.workspace("object-detect-xejgt").project("knife-detect").version(1).download(
    "yolov8", location="/content/knife_data")        # С‚РѕР»СЊРєРѕ РЅРѕР¶Рё
rf.workspace("niksha-zml5f").project("gun-rpy2m").version(1).download(
    "yolov8", location="/content/gun_data")          # gun, pistol, knife, person

def find_base(root, tag):
    c = glob.glob(f"{root}/**/data.yaml", recursive=True)
    if not c:
        raise SystemExit(f"data.yaml РЅРµ РЅР°Р№РґРµРЅ РІ {root}")
    return os.path.dirname(c[0])

def resolve_split(base):
    """Р РµРєСѓСЂСЃРёРІРЅРѕ РёС‰РµС‚ РїР°РїРєРё train/valid/test СЃ images РїРѕРґ base."""
    res = {}
    for key in ("train", "valid", "test"):
        found = glob.glob(f"{base}/**/{key}/images", recursive=True)
        found += glob.glob(f"{base}/**/{key}*/images", recursive=True)
        # РёСЃРєР»СЋС‡Р°РµРј СѓР¶Рµ СЃРѕР·РґР°РЅРЅС‹Р№ MERGE, Р±РµСЂС‘Рј РїРµСЂРІС‹Р№ РїРѕРґС…РѕРґСЏС‰РёР№
        for p in found:
            if "/merged/" not in p and os.path.isdir(p):
                res[key] = p
                break
        if key not in res:
            print(f"  Р’РќРРњРђРќРР•: РЅРµ РЅР°С€С‘Р» {key}/images РїРѕРґ {base}")
    return res

old_base = find_base("/content/old_data", "old")
knife_base = find_base("/content/knife_data", "knife")
gun_base = find_base("/content/gun_data", "gun")
print("old:", old_base)
print("knife:", knife_base)
print("gun:", gun_base)
splitz = [("old", old_base, None), ("knf", knife_base, None), ("gun", gun_base, GUN_MAP)]

# ---------- 5. РЎР±РѕСЂРєР° РѕР±СЉРµРґРёРЅС‘РЅРЅРѕРіРѕ РґР°С‚Р°СЃРµС‚Р° ----------
MERGE = "/content/merged"
!rm -rf {MERGE}
for split in ("train", "valid", "test"):
    os.makedirs(f"{MERGE}/{split}/images", exist_ok=True)
    os.makedirs(f"{MERGE}/{split}/labels", exist_ok=True)

# РС‚РѕРіРѕРІС‹Рµ 16 РєР»Р°СЃСЃРѕРІ (РїРѕСЂСЏРґРѕРє = РёРЅРґРµРєСЃС‹)
CLASS_NAMES = ['Knife', 'ak', 'ax', 'cleaver', 'cutter', 'eto', 'long sword',
               'm16', 'revolver', 'rifle', 'semi automatic', 'short sword',
               'shotgun', 'spear', 'gun', 'pistol']
# РњР°РїРїРёРЅРі РєР»Р°СЃСЃР° РёР· gun-РґР°С‚Р°СЃРµС‚Р° -> РёРЅРґРµРєСЃ РІ CLASS_NAMES
GUN_MAP = {'gun': 14, 'knife': 0, 'pistol': 15}   # person - РѕС‚Р±СЂР°СЃС‹РІР°РµРј

# РћРіСЂР°РЅРёС‡РµРЅРёРµ СЂР°Р·РјРµСЂР° gun-РґР°С‚Р°СЃРµС‚Р° (РІРµСЃСЊ = 32Рє РєР°РґСЂРѕРІ, СЌС‚Рѕ ~5 С‡Р°СЃРѕРІ).
# РЈРІРµР»РёС‡СЊ, РµСЃР»Рё РіРѕС‚РѕРІ Р¶РґР°С‚СЊ РґРѕР»СЊС€Рµ.
MAX_GUN_TRAIN = 6000
MAX_GUN_VAL = 1000
MAX_GUN_TEST = 300

def copy_split(name, base, class_map=None):
    """РљРѕРїРёСЂСѓРµС‚ images+labels РёР· СЃРїР»РёС‚РѕРІ Р±Р°Р·С‹ РІ MERGE.
    class_map: {СЃС‚Р°СЂРѕРµ_РёРјСЏ_РєР»Р°СЃСЃР°: РЅРѕРІС‹Р№_РёРЅРґРµРєСЃ}."""
    dirs = resolve_split(base)
    for split, simg in dirs.items():
        if not simg:
            continue
        bases = [os.path.splitext(f)[0] for f in os.listdir(simg)]
        maxn = {"train": MAX_GUN_TRAIN, "valid": MAX_GUN_VAL, "test": MAX_GUN_TEST}.get(split)
        if class_map and maxn:
            random.seed(42)
            bases = random.sample(bases, min(len(bases), maxn))
        copies = 0
        for b in bases:
            img_copied = False
            for ext, dir_key in ((".jpg", "images"), (".jpeg", "images"),
                                 (".png", "images"), (".txt", "labels")):
                src = os.path.join(os.path.dirname(simg), dir_key, b + ext)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(MERGE, split, dir_key, f"{name}_{b}{ext}")
                if ext == ".txt":
                    lines = []
                    with open(src) as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            if class_map is None:
                                new_cls = parts[0]
                            else:
                                old_name = {0: 'gun', 1: 'knife', 2: 'person', 3: 'pistol'}[int(parts[0])]
                                if old_name == 'person':
                                    continue
                                new_cls = class_map[old_name]
                            parts[0] = str(new_cls)
                            lines.append(" ".join(parts))
                    with open(dst, "w") as fh:
                        fh.write("\n".join(lines))
                else:
                    shutil.copy2(src, dst)
                    img_copied = True
            if img_copied:
                copies += 1
        print(f"  {name}:{split}: {copies} РєР°РґСЂРѕРІ СЃРєРѕРїРёСЂРѕРІР°РЅРѕ")

for name, base, cmap in splitz:
    copy_split(name, base, cmap)

# data.yaml
merged_yaml = os.path.join(MERGE, "data.yaml")
with open(merged_yaml, "w") as f:
    yaml.dump({
        "path": MERGE,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }, f, default_flow_style=False, allow_unicode=True)

print("РћР±СЉРµРґРёРЅС‘РЅРЅС‹Р№ РґР°С‚Р°СЃРµС‚:")
for split in ("train", "valid", "test"):
    n = len(os.listdir(os.path.join(MERGE, split, "images")))
    print(f"  {split}: {n} РєР°РґСЂРѕРІ")

# ---------- 6. РћР‘РЈР§Р•РќРР• ----------
model = YOLO(CKPT)
model.train(
    data=merged_yaml,
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    patience=8,
)

# ---------- 7. РЎРѕС…СЂР°РЅРµРЅРёРµ ----------
best = glob.glob("/content/runs/detect/*/weights/best.pt")[-1]
!mkdir -p /content/weights
!cp {best} /content/weights/weapon_detection_16cls.pt
print("Р“РћРўРћР’Рћ! /content/weights/weapon_detection_16cls.pt")
print("РЎРѕС…СЂР°РЅРё РІ Drive (РІСЃС‚Р°РІСЊ СЏС‡РµР№РєСѓ СЃРЅРёР·Сѓ) РёР»Рё СЃРєР°С‡Р°Р№: Files -> weights")

# ---------- Google Drive ----------
# from google.colab import drive
# drive.mount('/content/drive')
# shutil.copy('/content/weights/weapon_detection_16cls.pt',
#             '/content/drive/MyDrive/weapon_detection_16cls.pt')
# print("РЎРѕС…СЂР°РЅРµРЅРѕ РІ Drive!")
