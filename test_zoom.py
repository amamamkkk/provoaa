import glob
import sys
sys.path.insert(0, r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror")
import cv2
from detector import WeaponDetector, load_config

cfg = load_config(r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror\config.yaml")
det = WeaponDetector(cfg)

base = "C:/Users/MSI KATANA/Downloads/Weapon Detection.v2i.yolov8/valid/images/*.jpg"
imgs = sorted(glob.glob(base))[:200]

total_danger = 0
total_danger_zoom = 0
frames_with_weapon = 0
frames_with_weapon_zoom = 0
zoom_only = 0

for p in imgs:
    frame = cv2.imread(p)
    if frame is None:
        continue
    # обычный проход
    cfg2 = dict(cfg)
    cfg2["zoom"] = {"enabled": False}
    det2 = WeaponDetector(cfg2)
    d1 = det2.detect(frame)
    # с зумом
    d2 = det.detect(frame)
    w1 = [d for d in d1 if d.level in ("danger", "warning")]
    w2 = [d for d in d2 if d.level in ("danger", "warning")]
    if w1:
        frames_with_weapon += 1
        total_danger += len(w1)
    if w2:
        frames_with_weapon_zoom += 1
        total_danger_zoom += len(w2)
    if w2 and not w1:
        zoom_only += 1

print(f"кадров: {len(imgs)}")
print(f"без зума:  оружие на {frames_with_weapon} кадрах ({total_danger} боксов)")
print(f"с зумом:   оружие на {frames_with_weapon_zoom} кадрах ({total_danger_zoom} боксов)")
print(f"только зум поймал: {zoom_only} кадров")