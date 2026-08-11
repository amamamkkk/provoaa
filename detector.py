import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


WEAPON_CLASSES = {
    "knife": "нож",
    "semi automatic": "полуавтомат",
    "revolver": "револьвер",
    "long sword": "меч",
    "short sword": "короткий меч",
    "cleaver": "тесак",
    "cutter": "режущее",
    "ak": "АК",
    "ax": "топор",
    "m16": "M16",
    "spear": "копьё",
    "eto": "оружие",
    "gun": "пистолет",
    "pistol": "пистолет",
    "rifle": "винтовка",
    "shotgun": "дробовик",
    "sword": "меч",
    "axe": "топор",
    "bat": "бита",
    "weapon": "оружие",
    "wrench": "гаечный ключ",
    "hammer": "молоток",
}

ALERT_COLORS = {
    "danger": (0, 0, 255),      # красный
    "warning": (0, 200, 255),   # жёлтый
    "normal": (0, 255, 0),      # зелёный
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Detection:
    cls_name: str
    conf: float
    box: tuple[int, int, int, int]
    level: str
    label: str


class WeaponDetector:
    def __init__(self, config: dict) -> None:
        self.config = config
        device = config.get("device", "auto")

        if device == "auto" or device not in ("cpu", 0, 1, "0", "1"):
            device = 0 if torch.cuda.is_available() else "cpu"
        if device in ("0", "1"):
            device = int(device)
        self.device = device

        self.model = YOLO(config["model_path"])
        self.model.to(self.device)

        names = self.model.names
        self._idx_to_label = {i: str(n).lower() for i, n in names.items()}
        self._weapon_idxs = {
            i for i, n in self._idx_to_label.items()
            if self._label_is_weapon(n)
        }
        self._person_idxs = {
            i for i, n in self._idx_to_label.items()
            if n in ("person", "human")
        }

        person_path = config.get("person_model_path", "yolo11n.pt")
        self.person_model = YOLO(person_path)
        self.person_model.to(self.device)
        self._person_coco_idx = 0  # в COCO класс person имеет индекс 0
        self._frame_count = 0

    @staticmethod
    def _label_is_weapon(label: str) -> bool:
        for key in WEAPON_CLASSES:
            if key in label:
                return True
        return False

    def _classify(self, cls_name: str, conf: float) -> str:
        cfg = self.config["conf"]
        if self._label_is_weapon(cls_name):
            if conf >= cfg["weapon_high"]:
                return "danger"
            if conf >= cfg["weapon_low"]:
                return "warning"
        return "normal"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._frame_count += 1
        detections = self._detect_weapons(frame)

        zoom_cfg = self.config.get("zoom", {})
        if zoom_cfg.get("enabled", False):
            if not zoom_cfg.get("only_if_empty", True) or not detections:
                detections += self._detect_weapons_zoomed(frame, zoom_cfg)

        n = self.config.get("person_every_n", 1)
        if self.config.get("detect_persons", True) and (self._frame_count % n == 0 or not detections):
            detections += self._detect_persons(frame)
        return detections

    def _detect_weapons_zoomed(self, frame: np.ndarray, zoom_cfg: dict) -> list[Detection]:
        """Цифровой зум центра кадра + повторный проход для дальних мелких объектов."""
        h, w = frame.shape[:2]
        frac = zoom_cfg.get("crop_fraction", 0.55)
        cw = max(64, int(w * frac))
        ch = max(64, int(h * frac))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        crop = frame[y0:y0 + ch, x0:x0 + cw]
        zoomed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        cfg = self.config
        base_conf = cfg["conf"]["weapon_low"]
        conf = max(0.05, base_conf - zoom_cfg.get("conf_drop", 0.05))
        results = self.model.predict(
            source=zoomed,
            imgsz=cfg.get("imgsz", 640),
            conf=conf,
            max_det=cfg.get("max_det", 300),
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        sx = w / cw
        sy = h / ch
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self._idx_to_label.get(cls_id, "unknown")
            if not self._label_is_weapon(cls_name):
                continue
            confv = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            # обратное преобразование координат из увеличенного в исходный кадр
            ox1 = int(x0 + x1 / sx)
            oy1 = int(y0 + y1 / sy)
            ox2 = int(x0 + x2 / sx)
            oy2 = int(y0 + y2 / sy)
            level = self._classify(cls_name, confv)
            label = self._build_label(cls_name, confv, level)
            detections.append(Detection(cls_name, confv, (ox1, oy1, ox2, oy2), level, label))
        return detections

    def _detect_weapons(self, frame: np.ndarray) -> list[Detection]:
        cfg = self.config
        results = self.model.predict(
            source=frame,
            imgsz=cfg.get("imgsz", 640),
            conf=cfg["conf"]["weapon_low"],
            max_det=cfg.get("max_det", 300),
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self._idx_to_label.get(cls_id, "unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            level = self._classify(cls_name, conf)
            label = self._build_label(cls_name, conf, level)
            detections.append(Detection(cls_name, conf, (x1, y1, x2, y2), level, label))
        return detections

    def _detect_persons(self, frame: np.ndarray) -> list[Detection]:
        cfg = self.config
        conf = cfg["conf"].get("person", 0.35)
        results = self.person_model.predict(
            source=frame,
            imgsz=cfg.get("imgsz", 640),
            conf=conf,
            classes=[self._person_coco_idx],
            max_det=cfg.get("max_det", 50),
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(Detection("person", conf, (x1, y1, x2, y2), "normal", f"человек {conf:.2f}"))
        return detections

    def _build_label(self, cls_name: str, conf: float, level: str) -> str:
        ru = WEAPON_CLASSES.get(cls_name, cls_name)
        marker = {"danger": "!", "warning": "?", "normal": ""}[level]
        return f"{marker}{ru} {conf:.2f}"


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for d in detections:
        x1, y1, x2, y2 = d.box
        color = ALERT_COLORS[d.level]
        thickness = 3 if d.level == "danger" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(d.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, d.label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def save_alert(frame: np.ndarray, detections: list[Detection], alert_dir: str) -> None:
    Path(alert_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    danger = any(d.level == "danger" for d in detections)
    prefix = "DANGER" if danger else "WARNING"
    path = Path(alert_dir) / f"{prefix}_{stamp}.jpg"
    cv2.imwrite(str(path), frame)


def analyze(frame: np.ndarray, detector: WeaponDetector, config: dict) -> None:
    detections = detector.detect(frame)
    draw_detections(frame, detections)

    danger = [d for d in detections if d.level == "danger"]
    warning = [d for d in detections if d.level == "warning"]

    if danger and config.get("save_alerts", True):
        save_alert(frame, detections, config.get("alert_dir", "alerts"))

    if config.get("alert_sound", False) and danger:
        print("\a", end="")

    status = "ТРЕВОГА: ОРУЖИЕ" if danger else ("ВНИМАНИЕ" if warning else "НОРМА")
    color = ALERT_COLORS["danger"] if danger else (
        ALERT_COLORS["warning"] if warning else ALERT_COLORS["normal"]
    )
    cv2.putText(frame, f"[{status}]", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
