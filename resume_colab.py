# ===== РџР РћР”РћР›Р–Р•РќРР• РћР‘РЈР§Р•РќРРЇ РЎ 7 Р­РџРћРҐР (Google Colab) =====
# РџРµСЂРµРґ Р·Р°РїСѓСЃРєРѕРј:
#  1. РџРµСЂРµС‚Р°С‰Рё weapon_last.pt РІ Colab (Files -> РїРѕСЏРІРёС‚СЃСЏ /content/weapon_last.pt)
#  2. Runtime -> Change runtime type -> T4 GPU
#  3. Ctrl+F9 (Run all)
# РЎРґРµР»Р°РЅРѕ С‡С‚РѕР±С‹ РїСЂРѕРґРѕР»Р¶РёС‚СЊ СЃ СЃРѕС…СЂР°РЅС‘РЅРЅРѕР№ СЌРїРѕС…Рё, Р° РЅРµ РЅР°С‡РёРЅР°С‚СЊ СЃ РЅСѓР»СЏ.

import os
import glob
import yaml

# ---------- 1. РЈСЃС‚Р°РЅРѕРІРєР° Р±РёР±Р»РёРѕС‚РµРє ----------
!pip install -q ultralytics
!pip install -q roboflow

# ---------- 2. РџСЂРѕРІРµСЂРєР° GPU ----------
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("РќР•Рў GPU. Runtime -> Change runtime type -> T4 GPU")

# ---------- 3. РЎРєР°С‡РёРІР°РЅРёРµ РґР°С‚Р°СЃРµС‚Р° ----------
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_KEY")
rf.workspace("buildx").project("weapon-detection-7kro8").version(2).download("yolov8", location="/content/dataset")
print("Р”Р°С‚Р°СЃРµС‚ РіРѕС‚РѕРІ")

# ---------- 4. Р§РёРЅРёРј data.yaml ----------
c = glob.glob("/content/dataset/**/data.yaml", recursive=True)[0]
with open(c) as f:
    cfg = yaml.safe_load(f)
base = os.path.dirname(c)
cfg["path"] = base
m = {"train": "train", "val": "valid", "test": "test"}
for k in ("train", "val", "test"):
    if k in cfg:
        cfg[k] = f"{m[k]}/images"
with open(c, "w") as f:
    yaml.dump(cfg, f)
print("data.yaml:", c)

# ---------- 5. РџР РћР”РћР›Р–РРўР¬ РћР‘РЈР§Р•РќРР• РЎ 7 Р­РџРћРҐР ----------
from ultralytics import YOLO
CKPT = "/content/weapon_last.pt"   # РїСѓС‚СЊ Рє С‚РІРѕРµРјСѓ last.pt РІ Colab

try:
    model = YOLO(CKPT)
    model.train(
        data="/content/dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
        resume=True,
    )
except Exception as e:
    print("resume РЅРµ СЃСЂР°Р±РѕС‚Р°Р», СЃС‚Р°СЂС‚СѓРµРј РѕС‚ РѕР±СѓС‡РµРЅРЅС‹С… РІРµСЃРѕРІ:", e)
    model = YOLO(CKPT)
    model.train(
        data="/content/dataset/data.yaml",
        epochs=93,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
    )

# ---------- 6. РЎРѕС…СЂР°РЅРµРЅРёРµ Р»СѓС‡С€Рµ РІРµСЃРѕРІ ----------
best = glob.glob("/content/runs/detect/*/weights/best.pt")[-1]
!mkdir -p /content/weights
!cp {best} /content/weights/weapon_detection.pt
print("Р“РћРўРћР’Рћ! РљР°С‡Р°Р№: Files -> weights -> weapon_detection.pt")
# Р”РѕРї. СЃРѕС…СЂР°РЅРµРЅРёРµ РІ Drive (СЂР°СЃРєРѕРјРјРµРЅС‚РёСЂСѓР№ РїРѕСЃР»Рµ РјРѕРЅС‚РёСЂРѕРІР°РЅРёСЏ):
# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/weights/weapon_detection.pt /content/drive/MyDrive/weapon_detection.pt
# print("РЎРѕС…СЂР°РЅРµРЅРѕ РІ Google Drive")
