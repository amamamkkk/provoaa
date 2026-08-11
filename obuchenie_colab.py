# ============ РћР‘РЈР§Р•РќРР• Р”Р•РўР•РљР¦РР РћР РЈР–РРЇ Р’ GOOGLE COLAB ============
# Р—Р°РїСѓСЃРє: СЃРІРµСЂС…Сѓ РјРµРЅСЋ Runtime -> Run all  (РёР»Рё РЅР°Р¶РјРё Ctrl+F9)
# РџРµСЂРµРґ Р·Р°РїСѓСЃРєРѕРј: Runtime -> Change runtime type -> T4 GPU

import os
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

# ---------- 3.1. РСЃРїСЂР°РІР»СЏРµРј РїСѓС‚Рё РІ data.yaml (Roboflow РѕС‚РґР°С‘С‚ СЃР»РѕРјР°РЅРЅС‹Рµ) ----------
import glob, yaml
candidates = glob.glob("/content/dataset/**/data.yaml", recursive=True)
if not candidates:
    raise SystemExit("data.yaml РЅРµ РЅР°Р№РґРµРЅ. РќР°Р№РґРё РµРіРѕ РІ Files (РїР°РїРєР° СЃР»РµРІР°) Рё РІСЃС‚Р°РІСЊ РїРѕР»РЅС‹Р№ РїСѓС‚СЊ РЅРёР¶Рµ.")
yaml_path = candidates[0]
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)
base = os.path.dirname(yaml_path)
cfg["path"] = base
mapping = {"train": "train", "val": "valid", "test": "test"}
for k in ("train", "val", "test"):
    if k in cfg:
        cfg[k] = f"{mapping[k]}/images"
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
!mkdir -p /content/weights
!cp /content/runs/detect/train/weights/best.pt /content/weights/weapon_detection.pt
import os
size = os.path.getsize('/content/weights/weapon_detection.pt') / 1e6
print(f"Р“РћРўРћР’Рћ! Р’РµСЃ РјРѕРґРµР»Рё: {size:.1f} РњР‘ -> /content/weights/weapon_detection.pt")
