# ============ Р”РћРћР‘РЈР§Р•РќРР• РњРћР”Р•Р›Р РћР РЈР–РРЇ РќРђ РќРћР–РђРҐ (transfer learning) ============
# Р§С‚Рѕ РЅСѓР¶РЅРѕ СЃРґРµР»Р°С‚СЊ Р”Рћ Р·Р°РїСѓСЃРєР° (Files СЃР»РµРІР°):
#   РџРµСЂРµС‚Р°С‰Рё СЃРІРѕСЋ РѕР±СѓС‡РµРЅРЅСѓСЋ РјРѕРґРµР»СЊ weapon_detection.pt -> /content/weapon_detection.pt
#   (Р°СЂС…РёРІ СЃ РЅРѕР¶Р°РјРё РќР• РЅСѓР¶РµРЅ вЂ” РґР°С‚Р°СЃРµС‚ СЃРєР°С‡Р°РµС‚СЃСЏ СЃР°Рј РїРѕ API)
# Р—Р°С‚РµРј: Runtime -> Change runtime type -> T4 GPU -> Ctrl+F9
# Р РµР·СѓР»СЊС‚Р°С‚: /content/weights/weapon_detection_knife.pt (РєР°С‡Р°Р№ РёР»Рё Р±РµСЂРё РёР· Drive)

import os
import glob
import shutil
import yaml

# ---------- 1. РЈСЃС‚Р°РЅРѕРІРєР° Р±РёР±Р»РёРѕС‚РµРє ----------
!pip install -q ultralytics
!pip install -q roboflow
from ultralytics import YOLO

# ---------- 2. РџСЂРѕРІРµСЂРєР° GPU ----------
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("РќР•Рў GPU. Runtime -> Change runtime type -> T4 GPU")

# ---------- 3. РЎС‚Р°СЂР°СЏ РјРѕРґРµР»СЊ ----------
CKPT = "/content/weapon_detection.pt"
assert os.path.exists(CKPT), "РќРµ РІРёР¶Сѓ /content/weapon_detection.pt! РџРµСЂРµС‚Р°С‰Рё РјРѕРґРµР»СЊ РІ Files."
print("РњРѕРґРµР»СЊ Р·Р°РіСЂСѓР¶РµРЅР°:", CKPT)

# ---------- 4. РќРѕР¶РµРІРѕР№ РґР°С‚Р°СЃРµС‚ (СЃРєР°С‡РёРІР°РµС‚СЃСЏ СЃР°Рј, Р°СЂС…РёРІ РЅРµ РЅСѓР¶РµРЅ) ----------
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_KEY")
rf.workspace("object-detect-xejgt").project("knife-detect").version(1).download(
    "yolov8", location="/content/knife_data")
knife_yaml = glob.glob("/content/knife_data/**/data.yaml", recursive=True)[0]
knife_base = os.path.dirname(knife_yaml)
print("РќРѕР¶РµРІРѕР№ РґР°С‚Р°СЃРµС‚:", knife_base)

# ---------- 5. РЎС‚Р°СЂС‹Р№ РґР°С‚Р°СЃРµС‚ (С‡С‚РѕР±С‹ РЅРµ РїРѕС‚РµСЂСЏС‚СЊ РІСЃРµ 14 РєР»Р°СЃСЃРѕРІ) ----------
rf.workspace("buildx").project("weapon-detection-7kro8").version(2).download(
    "yolov8", location="/content/old_data")
old_yaml = glob.glob("/content/old_data/**/data.yaml", recursive=True)[0]
old_base = os.path.dirname(old_yaml)
print("РЎС‚Р°СЂС‹Р№ РґР°С‚Р°СЃРµС‚:", old_base)

# ---------- 6. РћР±СЉРµРґРёРЅСЏРµРј РІ РѕРґРёРЅ РґР°С‚Р°СЃРµС‚ СЃ 14 РєР»Р°СЃСЃР°РјРё ----------
MERGE = "/content/merged"
!rm -rf {MERGE}
os.makedirs(f"{MERGE}/train/images", exist_ok=True)
os.makedirs(f"{MERGE}/train/labels", exist_ok=True)
os.makedirs(f"{MERGE}/valid/images", exist_ok=True)
os.makedirs(f"{MERGE}/valid/labels", exist_ok=True)
os.makedirs(f"{MERGE}/test/images", exist_ok=True)
os.makedirs(f"{MERGE}/test/labels", exist_ok=True)

CLASS_NAMES = ['Knife', 'ak', 'ax', 'cleaver', 'cutter', 'eto', 'long sword',
               'm16', 'revolver', 'rifle', 'semi automatic', 'short sword',
               'shotgun', 'spear']

def copy_split(src_base, class_names):
    """РљРѕРїРёСЂСѓРµС‚ images+labels РёР· YOLO-РґР°С‚Р°СЃРµС‚Р° РІ РѕР±С‰РёР№ (train/valid/test)."""
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            s = os.path.join(src_base, split, sub)
            if not os.path.isdir(s):
                continue
            dst = os.path.join(MERGE, split, sub)
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(s):
                shutil.copy2(os.path.join(s, f), dst)

# РєРѕРїРёСЂСѓРµРј СЃС‚Р°СЂС‹Рµ (14 РєР»Р°СЃСЃРѕРІ, РјРµС‚РєРё СѓР¶Рµ РІРµСЂРЅС‹Рµ)
copy_split(old_base, CLASS_NAMES)           # train/valid/test

# РєРѕРїРёСЂСѓРµРј РЅРѕР¶Рё: РєР»Р°СЃСЃ РІСЃСЋРґСѓ Р·Р°РјРµРЅСЏРµРј РЅР° 0 (Knife); РїСЂРµС„РёРєСЃ knf_ вЂ” С‡С‚РѕР±С‹ РЅРµ РїРµСЂРµСЃРµРєР»РёСЃСЊ РёРјРµРЅР°
for split in ["train", "valid", "test"]:
    simg = os.path.join(knife_base, split, "images")
    slab = os.path.join(knife_base, split, "labels")
    if not os.path.isdir(simg):
        continue
    for f in os.listdir(simg):
        shutil.copy2(os.path.join(simg, f),
                     os.path.join(MERGE, split, "images", "knf_" + f))
    if os.path.isdir(slab):
        for f in os.listdir(slab):
            src = os.path.join(slab, f)
            # РїРµСЂРµРїРёСЃС‹РІР°РµРј РїРµСЂРІС‹Рµ СЃРёРјРІРѕР»С‹ СЃС‚СЂРѕРєРё РІ "0"
            lines = []
            with open(src) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = "0"
                        lines.append(" ".join(parts))
            with open(os.path.join(MERGE, split, "labels", "knf_" + f), "w") as fh:
                fh.write("\n".join(lines))

# data.yaml РґР»СЏ РѕР±СЉРµРґРёРЅС‘РЅРЅРѕРіРѕ
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

print("РћР±СЉРµРґРёРЅС‘РЅРЅС‹Р№ РґР°С‚Р°СЃРµС‚:", MERGE)
for split in ["train", "valid", "test"]:
    n = len(os.listdir(os.path.join(MERGE, split, "images")))
    print(f"  {split}: {n} РєР°РґСЂРѕРІ")

# ---------- 7. Р”РћРћР‘РЈР§Р•РќРР• С‚РІРѕРµР№ РјРѕРґРµР»Рё ----------
model = YOLO(CKPT)
model.train(
    data=merged_yaml,
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    patience=10,
)

# ---------- 8. РЎРѕС…СЂР°РЅРµРЅРёРµ ----------
best = glob.glob("/content/runs/detect/*/weights/best.pt")[-1]
!mkdir -p /content/weights
!cp {best} /content/weights/weapon_detection_knife.pt
size_mb = os.path.getsize("/content/weights/weapon_detection_knife.pt") / 1e6
print(f"Р“РћРўРћР’Рћ! {size_mb:.1f} РњР‘: /content/weights/weapon_detection_knife.pt")
print("РЎРєР°С‡Р°Р№: Files -> weights -> weapon_detection_knife.pt -> РџРљРњ -> Download")

# ---------- (РЅРµРѕР±СЏР·Р°С‚РµР»СЊРЅРѕ) Google Drive ----------
# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/weights/weapon_detection_knife.pt /content/drive/MyDrive/weapon_detection_knife.pt
# print("РЎРѕС…СЂР°РЅРµРЅРѕ РІ Drive!")
