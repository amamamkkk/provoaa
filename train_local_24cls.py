# ===== Р›РћРљРђР›Р¬РќРћР• РћР‘РЈР§Р•РќРР•: 24 РљР›РђРЎРЎРђ РЅР° RTX 2050 (4GB) =====
# Р’СЃРµ РґР°С‚Р°СЃРµС‚С‹ СЃРєР°С‡РёРІР°СЋС‚СЃСЏ РїРѕ API РЅР° РґРёСЃРє D:. Р—Р°РїСѓСЃРєР°С‚СЊ РёР· РїР°РїРєРё РїСЂРѕРµРєС‚Р°.
# Р РµР·СѓР»СЊС‚Р°С‚: D:\antiterror_ml\weights\weapon_detection_24cls.pt

import os
import glob
import shutil
import random
import yaml

BASE = r"D:\antiterror_ml"
os.makedirs(BASE, exist_ok=True)

# ---------- 1. РРјРїРѕСЂС‚С‹ ----------
from ultralytics import YOLO
import torch

if not torch.cuda.is_available():
    raise SystemExit("РќР•Рў GPU. РџСЂРѕРІРµСЂСЊ, С‡С‚Рѕ CUDA РІРёРґРёС‚ RTX 2050.")
print("GPU:", torch.cuda.get_device_name(0))

# ---------- 2. РС‚РѕРіРѕРІС‹Рµ РєР»Р°СЃСЃС‹ (РїРѕСЂСЏРґРѕРє = РёРЅРґРµРєСЃС‹) ----------
CLASS_NAMES = [
    # --- СЃС‚Р°СЂС‹Рµ 16 (РЅРµ РјРµРЅСЏС‚СЊ РїРѕСЂСЏРґРѕРє!) ---
    'Knife', 'ak', 'ax', 'cleaver', 'cutter', 'eto', 'long sword',
    'm16', 'revolver', 'rifle', 'semi automatic', 'short sword',
    'shotgun', 'spear', 'gun', 'pistol',
    # --- РЅРѕРІС‹Рµ 8 ---
    'assault rifle', 'Bazooka', 'Hand-grenade', 'Landmine',
    'Machine Gun', 'SMG', 'Sniper Rifle', 'grenade-launcher',
]
NAME2IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
NC = len(CLASS_NAMES)
print(f"РС‚РѕРіРѕРІС‹С… РєР»Р°СЃСЃРѕРІ: {NC}")

# ---------- 3. РЎРєР°С‡РёРІР°РЅРёРµ 7 РґР°С‚Р°СЃРµС‚РѕРІ РїРѕ API ----------
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_KEY")

DATASETS = [
    ("old",       "buildx",            "weapon-detection-7kro8", 2),
    ("knife",     "object-detect-xejgt","knife-detect",           1),
    ("gun",       "niksha-zml5f",      "gun-rpy2m",               1),
    ("rifle_new", "em2023",            "rifle-6srh6",             3),
    ("assault_new","ml-intern-2023",   "assault-rifle",           1),
    ("gun10_new", "feici6s-workspace", "gun-lsvyo",               2),
    ("gun2_new",  "gosha",             "gun-ie4f6",              11),
]
for name, ws, proj, ver in DATASETS:
    dst = os.path.join(BASE, name)
    if os.path.isdir(dst) and os.listdir(dst):
        print(f"РЈР¶Рµ РµСЃС‚СЊ: {name}")
        continue
    print(f"РЎРєР°С‡РёРІР°СЋ: {name} (v{ver})...")
    rf.workspace(ws).project(proj).version(ver).download(
        "yolov8", location=dst, overwrite=False)
    print(f"РЎРєР°С‡Р°РЅ {name}")

def find_base(root):
    c = glob.glob(f"{root}/**/data.yaml", recursive=True)
    if not c:
        raise SystemExit(f"data.yaml РЅРµ РЅР°Р№РґРµРЅ РІ {root}")
    return os.path.dirname(c[0])

# ---------- 4. РњР°РїРїРёРЅРіРё ----------
GUNV1_MAP = {0: NAME2IDX['gun'], 1: NAME2IDX['Knife'], 3: NAME2IDX['pistol']}

def name_map(base, rules):
    """rules: {РёРјСЏ_РєР»Р°СЃСЃР°_РІ_РґР°С‚Р°СЃРµС‚Рµ: РёРјСЏ_РІ_РёС‚РѕРіРѕРІРѕРј}. person/РёРЅС‹Рµ -> None (РІС‹РєРёРЅСѓС‚СЊ)."""
    yaml_f = glob.glob(f"{base}/**/data.yaml", recursive=True)[0]
    with open(yaml_f) as fh:
        d = yaml.safe_load(fh)
    names = d.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    res = {}
    for old_idx, old_name in enumerate(names):
        target = rules.get(old_name)
        if target is None:
            continue
        res[old_idx] = NAME2IDX[target]
    print(f"  {os.path.basename(base)}: {len(res)} РєР»Р°СЃСЃРѕРІ -> {res}")
    return res

old_base = find_base(os.path.join(BASE, "old"))
knife_base = find_base(os.path.join(BASE, "knife"))
gun_base = find_base(os.path.join(BASE, "gun"))
RIFLE_MAP = name_map(os.path.join(BASE, "rifle_new"),   {"rifle": "rifle"})
ASSAULT_MAP = name_map(os.path.join(BASE, "assault_new"), {"assaultrifle": "assault rifle"})
GUN10_MAP = name_map(os.path.join(BASE, "gun10_new"), {
    "Bazooka": "Bazooka", "Hand-grenade": "Hand-grenade", "Landmine": "Landmine",
    "Machine Gun": "Machine Gun", "SMG": "SMG", "Sniper Rifle": "Sniper Rifle",
    "grenade-launcher": "grenade-launcher", "pistol": "pistol",
    "rifle": "rifle", "shotgun": "shotgun",
})
GUN2_MAP = name_map(os.path.join(BASE, "gun2_new"), {"2": "gun"})

# ---------- 5. РЎР±РѕСЂРєР° РѕР±СЉРµРґРёРЅС‘РЅРЅРѕРіРѕ РґР°С‚Р°СЃРµС‚Р° ----------
MERGE = os.path.join(BASE, "merged")
if os.path.isdir(MERGE):
    shutil.rmtree(MERGE)
for split in ("train", "valid", "test"):
    os.makedirs(f"{MERGE}/{split}/images", exist_ok=True)
    os.makedirs(f"{MERGE}/{split}/labels", exist_ok=True)

MAX_GUN_TRAIN, MAX_GUN_VAL, MAX_GUN_TEST = 6000, 1000, 300

def resolve_split(base):
    """Р РµРєСѓСЂСЃРёРІРЅРѕ РёС‰РµС‚ РїР°РїРєРё train/valid/test СЃ images РїРѕРґ base."""
    res = {}
    for key in ("train", "valid", "test"):
        found = glob.glob(f"{base}/**/{key}/images", recursive=True)
        found += glob.glob(f"{base}/**/{key}*/images", recursive=True)
        for p in found:
            if "merged" not in p and os.path.isdir(p):
                res[key] = p
                break
        if key not in res:
            print(f"  Р’РќРРњРђРќРР•: РЅРµ РЅР°С€С‘Р» {key}/images РїРѕРґ {base}")
    return res

def copy_split(name, base, class_map=None):
    """РљРѕРїРёСЂСѓРµС‚ images+labels РёР· СЃРїР»РёС‚РѕРІ Р±Р°Р·С‹ РІ MERGE.
    class_map: {СЃС‚Р°СЂС‹Р№_РёРЅРґРµРєСЃ: РЅРѕРІС‹Р№_РёРЅРґРµРєСЃ}; None = РєР°Рє РµСЃС‚СЊ."""
    dirs = resolve_split(base)
    for split, simg in dirs.items():
        if not simg:
            continue
        bases = [os.path.splitext(f)[0] for f in os.listdir(simg)]
        maxn = {"train": MAX_GUN_TRAIN, "valid": MAX_GUN_VAL, "test": MAX_GUN_TEST}.get(split)
        if maxn:
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
                                new_cls = class_map.get(int(parts[0]))
                                if new_cls is None:
                                    continue
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

splitz = [
    ("old",   old_base,   None),
    ("knf",   knife_base, None),
    ("gunv1", gun_base,   GUNV1_MAP),
    ("rif",   os.path.join(BASE, "rifle_new"),   RIFLE_MAP),
    ("asl",   os.path.join(BASE, "assault_new"), ASSAULT_MAP),
    ("g10",   os.path.join(BASE, "gun10_new"),   GUN10_MAP),
    ("g2",    os.path.join(BASE, "gun2_new"),    GUN2_MAP),
]
for name, base, cmap in splitz:
    copy_split(name, base, cmap)

merged_yaml = os.path.join(MERGE, "data.yaml")
with open(merged_yaml, "w") as f:
    yaml.dump({
        "path": MERGE,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": NC,
        "names": CLASS_NAMES,
    }, f, default_flow_style=False, allow_unicode=True)

print("РћР±СЉРµРґРёРЅС‘РЅРЅС‹Р№ РґР°С‚Р°СЃРµС‚:")
for split in ("train", "valid", "test"):
    n = len(os.listdir(os.path.join(MERGE, split, "images")))
    print(f"  {split}: {n} РєР°РґСЂРѕРІ")

# ---------- 6. РћР‘РЈР§Р•РќРР• (РїРѕРґ RTX 2050 4GB) ----------
CKPT = "yolo11s.pt"
model = YOLO(CKPT)
model.train(
    data=merged_yaml,
    epochs=25,
    imgsz=512,           # 640 РЅРµ РІР»РµР·РµС‚ РІ 4GB СЃ batch>8
    batch=8,
    workers=0,           # РЅР° Windows РѕР±СЏР·Р°С‚РµР»СЊРЅo 0 (РёРЅР°С‡Рµ spawn СѓРїР°РґС‘С‚)
    device=0,
    patience=10,
)

# ---------- 7. РЎРѕС…СЂР°РЅРµРЅРёРµ ----------
best = glob.glob(os.path.join(BASE, "runs", "detect", "*", "weights", "best.pt"))[-1]
out = os.path.join(BASE, "weights")
os.makedirs(out, exist_ok=True)
shutil.copy(best, os.path.join(out, "weapon_detection_24cls.pt"))
print("Р“РћРўРћР’Рћ!", os.path.join(out, "weapon_detection_24cls.pt"))
