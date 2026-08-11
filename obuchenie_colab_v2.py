# ============ РћР‘РЈР§Р•РќРР• Р”Р•РўР•РљР¦РР РћР РЈР–РРЇ  v2.0 (GOOGLE COLAB) ============
# Р—Р°РїСѓСЃРє: Runtime -> Change runtime type -> T4 GPU, Р·Р°С‚РµРј Ctrl+F9 (Run all)

import os
import glob
import yaml
os.environ.setdefault("PIP_CACHE_DIR", "/content/pip_cache")

# ---------- 1. РЈСЃС‚Р°РЅРѕРІРєР° Р±РёР±Р»РёРѕС‚РµРє ----------
!pip install -q ultralytics
!pip install -q roboflow
from ultralytics import YOLO

# ---------- 2. РџСЂРѕРІРµСЂРєР° GPU ----------
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("РќР•Рў GPU. РћС‚РєСЂРѕР№: Runtime -> Change runtime type -> T4 GPU")

# ---------- 3. РЎРєР°С‡РёРІР°РЅРёРµ РґР°С‚Р°СЃРµС‚Р° СЃ Roboflow ----------
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_KEY")
project = rf.workspace("buildx").project("weapon-detection-7kro8")
dataset = project.version(2).download("yolov8", location="/content/dataset")
print("Р”Р°С‚Р°СЃРµС‚ Р·Р°РіСЂСѓР¶РµРЅ")

# ---------- 3.1. РќР°С…РѕРґРёРј data.yaml Рё С‡РёРЅРёРј РїСѓС‚Рё ----------
candidates = glob.glob("/content/dataset/**/data.yaml", recursive=True)
if not candidates:
    raise SystemExit("data.yaml РЅРµ РЅР°Р№РґРµРЅ! РћС‚РєСЂРѕР№ СЃР»РµРІР° РїР°РїРєСѓ dataset Рё РїСЂРѕРІРµСЂСЊ СЃС‚СЂСѓРєС‚СѓСЂСѓ.")
yaml_path = candidates[0]
base = os.path.dirname(yaml_path)

with open(yaml_path) as f:
    cfg = yaml.safe_load(f)

cfg["path"] = base
for key in ("train", "val", "test"):
    if key in cfg:
        found = None
        for sub in os.listdir(base):
            if sub.lower().startswith(key) and os.path.isdir(os.path.join(base, sub, "images")):
                found = sub
                break
        if found:
            cfg[key] = f"{found}/images"
            print(f"  {key} -> {found}/images")

with open(yaml_path, "w") as f:
    yaml.dump(cfg, f)
print("data.yaml РёСЃРїСЂР°РІР»РµРЅ:", yaml_path)

# ---------- 4. РћР±СѓС‡РµРЅРёРµ ----------
model = YOLO("yolo11s.pt")
model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    patience=20,
)

# ---------- 5. РЎРѕС…СЂР°РЅРµРЅРёРµ РіРѕС‚РѕРІРѕР№ РјРѕРґРµР»Рё ----------
best = glob.glob("/content/runs/detect/*/weights/best.pt")[-1]
!mkdir -p /content/weights
!cp {best} /content/weights/weapon_detection.pt
size_mb = os.path.getsize('/content/weights/weapon_detection.pt') / 1e6
print(f"Р“РћРўРћР’Рћ! РњРѕРґРµР»СЊ: /content/weights/weapon_detection.pt ({size_mb:.1f} РњР‘)")
print("РЎРєР°С‡Р°Р№ РµС‘: СЃР»РµРІР° Files -> weights -> weapon_detection.pt -> РџРљРњ -> Download")
print("РР»Рё СЃРѕС…СЂР°РЅРё РІ Google Drive СЏС‡РµР№РєРѕР№ РЅРёР¶Рµ.")

# ---------- (РЅРµРѕР±СЏР·Р°С‚РµР»СЊРЅРѕ) РЎРѕС…СЂР°РЅРµРЅРёРµ РІ Google Drive ----------
# Р Р°СЃРєРѕРјРјРµРЅС‚РёСЂСѓР№ Рё Р·Р°РїСѓСЃС‚Рё РїРѕСЃР»Рµ РјРѕРЅС‚РёСЂРѕРІР°РЅРёСЏ:
# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/weights/weapon_detection.pt /content/drive/MyDrive/weapon_detection.pt
# print("РЎРѕС…СЂР°РЅРµРЅРѕ РІ Google Drive: weapon_detection.pt")
