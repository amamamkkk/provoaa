import argparse
import glob
import time

import cv2

from detector import WeaponDetector, ALERT_COLORS, draw_detections, load_config
from map_display import MapWindow


def _open_camera(source: int):
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
    return None


def camera_loop(detector, map_win, config, src) -> None:
    """Выполняется в отдельном потоке: камера + детекция + карта."""
    if src is None:
        cam = _open_camera(0)
        if cam is not None:
            cap = cam
            fps = 30.0
        else:
            videos = sorted(glob.glob("alerts/demo*.mp4"))
            cap = cv2.VideoCapture(videos[-1] if videos else "alerts/demo.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
    elif isinstance(src, str) and src.isdigit():
        cap = _open_camera(int(src))
        fps = 30.0
    else:
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if cap is None or not cap.isOpened():
        print("НЕ УДАЛОСЬ открыть источник")
        return

    print("РЕЖИМ: реальное время. q - выход | пробел - пауза | s - снимок")
    paused = False
    try:
        while True:
            if paused:
                k = cv2.waitKey(30) & 0xFF
                if k == ord(" "):
                    paused = False
                elif k == ord("q"):
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # зацикливаем демку
                continue

            detections = detector.detect(frame)
            draw_detections(frame, detections)
            danger = any(d.level == "danger" for d in detections)
            warning = any(d.level == "warning" for d in detections)
            status = "ТРЕВОГА: ОРУЖИЕ" if danger else ("ВНИМАНИЕ" if warning else "НОРМА")
            color = ALERT_COLORS["danger"] if danger else (
                ALERT_COLORS["warning"] if warning else ALERT_COLORS["normal"])
            cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(frame, f"объектов: {len(detections)}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if danger:
                cam = map_win.coords_for_camera(0)
                if cam:
                    map_win.show(cam["lat"], cam["lon"], cam["name"])

            cv2.imshow("Anti-Terror LIVE", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                paused = True
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None,
                        help="видео/фото/камера. По умолчанию: камера 0, иначе демо-видео")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = WeaponDetector(config)
    map_win = MapWindow(config)

    if map_win.usable:
        # карта живёт в главном потоке, камера — в отдельном
        map_win.create()
        map_win.run(lambda: camera_loop(detector, map_win, config, args.source))
    else:
        camera_loop(detector, map_win, config, args.source)


if __name__ == "__main__":
    main()