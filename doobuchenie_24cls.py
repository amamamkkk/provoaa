# ===== Р¤РРќРђР›Р¬РќРћР• РћР‘РЈР§Р•РќРР•: 24 РљР›РђРЎРЎРђ (16 СЃС‚Р°СЂС‹С… + 8 РЅРѕРІС‹С…) =====
# Р’СЃС‘ СЃРєР°С‡РёРІР°РµС‚СЃСЏ РїРѕ API, Р·РёРїС‹ РїРµСЂРµРЅРѕСЃРёС‚СЊ РќР• РЅСѓР¶РЅРѕ. T4 GPU -> Ctrl+F9.
# Р РµР·СѓР»СЊС‚Р°С‚: /content/weights/weapon_detection_24cls.pt

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

# ---------- 3. РС‚РѕРіРѕРІС‹Рµ РєР»Р°СЃСЃС‹ (РїРѕСЂСЏРґРѕРє = РёРЅРґРµРєСЃС‹) ----------
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

# ---------- 4. РЎС‚Р°СЂС‹Рµ 3 РґР°С‚Р°СЃРµС‚Р° РїРѕ API (РєР°Рє СЂР°РЅСЊС€Рµ) ----------
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

# ---------- 5. РќРѕРІС‹Рµ 4 РґР°С‚Р°СЃРµС‚Р° РїСЂСЏРјРѕ РїРѕ API (Р·РёРїС‹ РїРµСЂРµРЅРѕСЃРёС‚СЊ РќР• РЅСѓР¶РЅРѕ) ----------
NEW_DATASETS = [
    # (РїР°РїРєР°, workspace, project, version)
    ("rifle_new",   "em2023",           "rifle-6srh6",  3),   # Rifle.v3
    ("assault_new", "ml-intern-2023",   "assault-rifle", 1),  # assault rifle.v1
    ("gun10_new",   "feici6s-workspace","gun-lsvyo",    2),   # gun.v2 (10 РєР»Р°СЃСЃРѕРІ)
    ("gun2_new",    "gosha",            "gun-ie4f6",    11),  # gun.v11
]
for name, ws, proj, ver in NEW_DATASETS:
    dst = f"/content/{name}"
    if os.path.isdir(dst) and os.listdir(dst):
        print(f"РЈР¶Рµ РµСЃС‚СЊ: {name}")
        continue
    rf.workspace(ws).project(proj).version(ver).download(
        "yolov8", location=dst, overwrite=False)
    print(f"РЎРєР°С‡Р°РЅ {name} (v{ver})")

# ---------- 6. РњР°РїРїРёРЅРіРё СЃС‚Р°СЂС‹С… РґР°С‚Р°СЃРµС‚РѕРІ ----------
old_base = find_base("/content/old_data", "old")
knife_base = find_base("/content/knife_data", "knife")
gun_base = find_base("/content/gun_data", "gun")

# gun v1 (Roboflow): 0=gun, 1=knife, 2=person(РІС‹РєРёРЅСѓС‚СЊ), 3=pistol
GUNV1_MAP = {0: NAME2IDX['gun'], 1: NAME2IDX['Knife'], 3: NAME2IDX['pistol']}

# РќРѕРІС‹Рµ РґР°С‚Р°СЃРµС‚С‹: РјР°РїРїРёРј РїРѕ РРњР•РќР РєР»Р°СЃСЃР° (РёРЅРґРµРєСЃС‹ РІ РЅРёС… РґСЂСѓРіРёРµ!)
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
    print(f"  {os.path.basename(base)}: {len(res)} РєР»Р°СЃСЃРѕРІ -> РјР°РїРїРёРЅРі {res}")
    return res

RIFLE_MAP = name_map("/content/rifle_new",      {"rifle": "rifle"})
ASSAULT_MAP = name_map("/content/assault_new",  {"assaultrifle": "assault rifle"})
GUN10_MAP = name_map("/content/gun10_new", {
    "Bazooka": "Bazooka", "Hand-grenade": "Hand-grenade", "Landmine": "Landmine",
    "Machine Gun": "Machine Gun", "SMG": "SMG", "Sniper Rifle": "Sniper Rifle",
    "grenade-launcher": "grenade-launcher", "pistol": "pistol",
    "rifle": "rifle", "shotgun": "shotgun",
})
# gun.v11i: РёРјСЏ РєР»Р°СЃСЃР° "2" -> СЃС‡РёС‚Р°РµРј С‡С‚Рѕ СЌС‚Рѕ gun. Р•СЃР»Рё РЅРµ С‚Р°Рє - РїРѕРјРµРЅСЏР№.
GUN2_MAP = name_map("/content/gun2_new", {"2": "gun"})

# ---------- 7. РЎР±РѕСЂРєР° РѕР±СЉРµРґРёРЅС‘РЅРЅРѕРіРѕ РґР°С‚Р°СЃРµС‚Р° ----------
MERGE = "/content/merged"
!rm -rf {MERGE}
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
            if "/merged/" not in p and os.path.isdir(p):
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
    ("old",    old_base, None),
    ("knf",    knife_base, None),
    ("gunv1",  gun_base, GUNV1_MAP),
    ("rif",    "/content/rifle_new", RIFLE_MAP),
    ("asl",    "/content/assault_new", ASSAULT_MAP),
    ("g10",    "/content/gun10_new", GUN10_MAP),
    ("g2",     "/content/gun2_new", GUN2_MAP),
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

# ---------- 8. РћР‘РЈР§Р•РќРР• + Р•Р–Р•РњРРќРЈРўРќРћР• РђР’РўРћРЎРћРҐР РђРќР•РќРР• ----------
# СЃС‚Р°СЂС‚СѓРµРј РѕС‚ С‚РµРєСѓС‰РµР№ 16-РєР»Р°СЃСЃРЅРѕР№ РјРѕРґРµР»Рё (Р·РЅР°РєРѕРјС‹Рµ РєР»Р°СЃСЃС‹ РїРµСЂРµРЅРµСЃСѓС‚СЃСЏ)
# ---------- 4. РњРѕРЅС‚РёСЂСѓРµРј Drive Рё РІС‹Р±РёСЂР°РµРј СЃС‚Р°СЂС‚РѕРІСѓСЋ РјРѕРґРµР»СЊ ----------
DRV_DIR = "/content/drive/MyDrive"
SAVE_DIR = f"{DRV_DIR}/antiterror_24cls"

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_OK = os.path.isdir(DRV_DIR)
except Exception:
    DRIVE_OK = False
print("Drive:", "OK" if DRIVE_OK else "РЅРµ СЃРјРѕРЅС‚РёСЂРѕРІР°РЅ (Р°РІС‚РѕСЃРµР№РІ СЂР°Р±РѕС‚Р°С‚СЊ РЅРµ Р±СѓРґРµС‚)")
if DRIVE_OK:
    os.makedirs(SAVE_DIR, exist_ok=True)

# РЈР¶Рµ РµСЃС‚СЊ СЃРµР№РІ СЃ РїСЂРѕС€Р»РѕРіРѕ Р·Р°РїСѓСЃРєР° (Colab РѕС‚РєР»СЋС‡Р°Р»СЃСЏ)? РўРѕРіРґР° РїСЂРѕРґРѕР»Р¶РёРј СЃ РЅРµРіРѕ.
OLD_CKPT = "/content/weapon_detection_16cls.pt"
DRV_BEST = os.path.join(SAVE_DIR, "best.pt")
if DRIVE_OK and os.path.isfile(DRV_BEST):
    CKPT = DRV_BEST
    print("РќР°Р№РґРµРЅ СЃРµР№РІ РЅР° Drive:", DRV_BEST, "-> РґРѕРѕР±СѓС‡Р°СЋ СЃ РЅРµРіРѕ (РїСЂРѕРґРѕР»Р¶РµРЅРёРµ)")
elif os.path.isfile(OLD_CKPT):
    CKPT = OLD_CKPT
    print("РЎС‚Р°СЂС‚РѕРІР°СЏ РјРѕРґРµР»СЊ: weapon_detection_16cls.pt")
else:
    CKPT = "yolo11s.pt"
    print("РЎС‚Р°СЂС‚РѕРІР°СЏ РјРѕРґРµР»СЊ: yolo11s.pt (СЃ РЅСѓР»СЏ)")

# ---------- 5. РћР±СѓС‡РµРЅРёРµ РІ С„РѕРЅРѕРІРѕРј РїРѕС‚РѕРєРµ ----------
import threading
import time as _time
import shutil, hashlib

TRAIN_ERR = []

def run_train():
    try:
        model = YOLO(CKPT)
        model.train(
            data=merged_yaml,
            epochs=25,
            imgsz=640,
            batch=16,
            device=0,
            patience=10,
            save_period=1,      # РїРёСЃР°С‚СЊ last.pt РєР°Р¶РґСѓСЋ СЌРїРѕС…Сѓ
        )
    except Exception as e:
        TRAIN_ERR.append(e)
        print("РћРЁРР‘РљРђ РћР‘РЈР§Р•РќРРЇ:", e)

t = threading.Thread(target=run_train, daemon=True)
t.start()

# --- РєР°Р¶РґС‹Рµ 30 СЃРµРє РєРѕРїРёСЂСѓРµРј СЃРІРµР¶РёРµ РІРµСЃР° РЅР° Drive Рё РџР РћР’Р•Р РЇР•Рњ, С‡С‚Рѕ Р»РµРіР»Рѕ РІРµСЂРЅРѕ ---
# РџСЂРѕРІРµСЂРєРё:
#  1) Drive СЂРµР°Р»СЊРЅРѕ СЃРјРѕРЅС‚РёСЂРѕРІР°РЅ (РµСЃР»Рё РЅРµС‚ вЂ” РїСЂРѕР±СѓРµРј СЃРјРѕРЅС‚РёСЂРѕРІР°С‚СЊ РµС‰С‘ СЂР°Р·);
#  2) РєРѕРїРёСЂСѓРµРј С‚РѕР»СЊРєРѕ РµСЃР»Рё md5 Р»РѕРєР°Р»СЊРЅРѕРіРѕ С„Р°Р№Р»Р° РёР·РјРµРЅРёР»СЃСЏ;
#  3) РџРћРЎР›Р• РєРѕРїРёСЂРѕРІР°РЅРёСЏ СЃРІРµСЂСЏРµРј md5 РєРѕРїРёРё РЅР° Drive СЃ РёСЃС…РѕРґРЅРёРєРѕРј (РёРЅР°С‡Рµ СЃС‡РёС‚Р°РµРј СЃР±РѕР№);
#  4) РїСЂРё СЃР±РѕРµ РїРµС‡Р°С‚Р°РµРј РїСЂРёС‡РёРЅСѓ Рё РїСЂРѕР±СѓРµРј РЅР° СЃР»РµРґСѓСЋС‰РµРј С‚РёРєРµ;
#  5) СЃРїР°РјР° РЅРµС‚: РїРµС‡Р°С‚Р°РµРј С‚РѕР»СЊРєРѕ РїСЂРё СЂРµР°Р»СЊРЅРѕРј СЃРѕР±С‹С‚РёРё, СЂР°Р· РІ 10 РјРёРЅСѓС‚ вЂ” heartbeat.
FILES = (("last.pt", "last.pt"),
         ("best.pt", "best.pt"))

def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _drive_ok():
    if DRIVE_OK:
        return True
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return os.path.isdir(DRV_DIR)
    except Exception:
        return False

STATE = {}        # name -> md5 Р»РѕРєР°Р»СЊРЅРѕРіРѕ С„Р°Р№Р»Р°, РєРѕС‚РѕСЂС‹Р№ СѓР¶Рµ РїС‹С‚Р°Р»РёСЃСЊ СЃРѕС…СЂР°РЅРёС‚СЊ
CONFIRMED = {}    # name -> md5, РєРѕС‚РѕСЂС‹Р№ РџРћР”РўР’Р•Р Р–Р”РЃРќ РЅР° Drive (РєРѕРїРёСЏ СЃРІРµСЂРµРЅР°)
last_beat = _time.time()

while t.is_alive():
    try:
        mounted = _drive_ok()
        for name, drv in FILES:
            ws = glob.glob(f"/content/runs/detect/*/weights/{name}")
            if not ws:
                continue
            src = ws[0]
            h = _md5(src)
            if CONFIRMED.get(name) == h:
                continue          # СЌС‚Р° РІРµСЂСЃРёСЏ СѓР¶Рµ Р»РµР¶РёС‚ РЅР° Drive Рё СЃРІРµСЂРµРЅР°
            if not mounted:
                print(f"[{_time.strftime('%H:%M:%S')}] Drive РќР• СЃРјРѕРЅС‚РёСЂРѕРІР°РЅ вЂ” РїСЂРѕРїСѓСЃРєР°СЋ {name}")
                continue
            dst = f"{SAVE_DIR}/{drv}"
            shutil.copy(src, dst)
            # РїСЂРѕРІРµСЂРєР° РїРѕСЃР»Рµ РєРѕРїРёСЂРѕРІР°РЅРёСЏ
            if _md5(dst) != h:
                print(f"[{_time.strftime('%H:%M:%S')}] РћРЁРР‘РљРђ: {name} СЃРєРѕРїРёСЂРѕРІР°РЅ, РЅРѕ md5 РЅРµ СЃРѕРІРїР°Р» вЂ” РїРѕРїСЂРѕР±СѓСЋ СЃРЅРѕРІР°")
                try:
                    os.remove(dst)
                except OSError:
                    pass
                continue
            CONFIRMED[name] = h
            print(f"[РЎРћРҐР РђРќРЃРќ {_time.strftime('%H:%M:%S')}] {name} -> Drive ({os.path.getsize(src)//1024} KB, md5 {h[:12]}..., СЃРІРµСЂРµРЅРѕ РћРљ)")
        if _time.time() - last_beat > 600:
            last_beat = _time.time()
            print(f"[{_time.strftime('%H:%M:%S')}] heartbeat: РѕР±СѓС‡РµРЅРёРµ РёРґС‘С‚, РјРѕРЅРёС‚РѕСЂ Р¶РёРІ. "
                  f"Drive: {len(CONFIRMED)}/{len(FILES)} С„Р°Р№Р»РѕРІ СЃРІРµСЂРµРЅРѕ")
    except Exception as e:
        print(f"[{_time.strftime('%H:%M:%S')}] РђРІС‚РѕСЃРµР№РІ РѕС€РёР±РєР°:", repr(e))
    _time.sleep(30)

print("Р¤РѕРЅРѕРІРѕРµ РѕР±СѓС‡РµРЅРёРµ Р·Р°РІРµСЂС€РёР»РѕСЃСЊ вЂ” РІС‹С…РѕРґРёРј РёР· РјРѕРЅРёС‚РѕСЂР°")

# ---------- 9. Р¤РёРЅР°Р»СЊРЅРѕРµ СЃРѕС…СЂР°РЅРµРЅРёРµ ----------
# (РґРѕ СЃСЋРґР° РґРѕР№РґС‘Рј С‚РѕР»СЊРєРѕ РєРѕРіРґР° РѕР±СѓС‡РµРЅРёРµ Р·Р°РІРµСЂС€РёС‚СЃСЏ РёР»Рё СЃР»РѕРјР°РµС‚СЃСЏ)
if not TRAIN_ERR:
    best = glob.glob("/content/runs/detect/*/weights/best.pt")
    if best:
        final_src = best[-1]
        os.makedirs("/content/weights", exist_ok=True)
        shutil.copy(final_src, "/content/weights/weapon_detection_24cls.pt")
        print("Р“РћРўРћР’Рћ! /content/weights/weapon_detection_24cls.pt")
        if DRIVE_OK:
            try:
                dst = f"{SAVE_DIR}/weapon_detection_24cls_final.pt"
                shutil.copy(final_src, dst)
                if _md5(dst) == _md5(final_src):
                    print(f"РЎРћРҐР РђРќР•РќРћ Р’ DRIVE: {dst} "
                          f"({os.path.getsize(dst)//1024} KB, md5 {_md5(dst)[:12]}..., СЃРІРµСЂРµРЅРѕ РћРљ)")
                else:
                    print("РќР• РЈР”РђР›РћРЎР¬: С„РёРЅР°Р» РЅР° Drive РїРѕРІСЂРµР¶РґС‘РЅ (md5 РЅРµ СЃРѕРІРїР°Р»)")
            except Exception as e:
                print("РќРµ СЃРѕС…СЂР°РЅРёР»РѕСЃСЊ РЅР° Drive:", e)
    else:
        print("best.pt РЅРµ РЅР°Р№РґРµРЅ вЂ” РѕР±СѓС‡РµРЅРёРµ РЅРµ Р·Р°РІРµСЂС€РёР»РѕСЃСЊ")
else:
    print("РћР±СѓС‡РµРЅРёРµ СѓРїР°Р»Рѕ:", TRAIN_ERR[-1])
print("РЎРєР°С‡Р°Р№ РІСЂСѓС‡РЅСѓСЋ: Files -> weights -> Download")

