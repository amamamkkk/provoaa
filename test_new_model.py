import glob
import os
import sys
sys.path.insert(0, r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror")
import cv2
import collections
from ultralytics import YOLO

def test_model(ckpt):
    model = YOLO(ckpt)
    lab_dir = "C:/Users/MSI KATANA/Downloads/Weapon Detection.v2i.yolov8/valid/labels"
    img_dir = "C:/Users/MSI KATANA/Downloads/Weapon Detection.v2i.yolov8/valid/images"
    labs = sorted(glob.glob(lab_dir + "/*.txt"))
    names = model.names
    by_class = collections.defaultdict(list)
    for lab in labs:
        with open(lab, "r") as f:
            first = f.readline().strip()
        if not first:
            continue
        cls = int(first.split()[0])
        img = os.path.join(img_dir, os.path.splitext(os.path.basename(lab))[0] + ".jpg")
        if os.path.exists(img) and len(by_class[cls]) < 8:
            by_class[cls].append(img)
    out = []
    for cls in sorted(by_class):
        label = names.get(cls, str(cls))
        imgs = by_class[cls]
        hits = 0
        for p in imgs:
            f = cv2.imread(p)
            if f is None:
                continue
            r = model.predict(source=f, conf=0.25, imgsz=640, verbose=False)
            boxes = r[0].boxes
            if boxes is not None and len(boxes) > 0:
                hits += 1
        out.append(f"{label:14s} {hits}/{len(imgs)}")
    return out

res_new = test_model(r"C:\Users\MSI KATANA\Downloads\weapon_detection_knife.pt")
print("=== НОВАЯ (knife) ===")
for l in res_new:
    print(l)

res_old = test_model(r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror\weights\weapon_detection.pt")
print("=== СТАРАЯ ===")
for l in res_old:
    print(l)