import glob
import sys
import time
sys.path.insert(0, r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror")
import cv2
from detector import WeaponDetector, load_config
from map_display import MapDisplay

cfg = load_config(r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror\config.yaml")
det = WeaponDetector(cfg)
md = MapDisplay(cfg)

base = "C:/Users/MSI KATANA/Downloads/Weapon Detection.v2i.yolov8/valid/images/*.jpg"
for p in sorted(glob.glob(base))[:60]:
    frame = cv2.imread(p)
    if frame is None:
        continue
    dets = det.detect(frame)
    danger = [d for d in dets if d.level == "danger"]
    if danger:
        print("ТРЕВОГА! класс:", danger[0].label)
        cam = md.coords_for_camera(0)
        md.show(cam["lat"], cam["lon"], cam["name"])
        print("Карта открыта. Жду 25 сек, затем закрываюсь.")
        time.sleep(25)
        print("done")
        break
else:
    print("danger не найден в первых 60 кадрах")