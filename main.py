import argparse
import glob
import os
import time

import cv2

from detector import WeaponDetector, analyze, load_config


def _open_camera(source: int) -> cv2.VideoCapture | None:
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
    return None


def make_demo_video(config, detector, path="alerts/demo.mp4") -> str:
    os.makedirs("alerts", exist_ok=True)
    images = sorted(glob.glob("C:/Users/MSI KATANA/Downloads/Weapon Detection.v2i.yolov8/valid/images/*.jpg"))
    images = images[:40]
    if not images:
        raise RuntimeError("Нет кадров для демо-видео")
    first = cv2.imread(images[0])
    h, w = first.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (w, h))
    for img in images:
        frame = cv2.imread(img)
        if frame is None:
            continue
        analyze(frame, detector, config)
        out.write(frame)
    out.release()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Анти-террор: детекция в реальном времени")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = WeaponDetector(config)

    source = config["source"]
    is_path = isinstance(source, str) and os.path.exists(source)

    if is_path:
        cap = cv2.VideoCapture(source)
    elif isinstance(source, int) or source.isdigit():
        cap = _open_camera(int(source) if isinstance(source, int) else int(source))
    else:
        cap = _open_camera(source)

    if cap is not None and cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = None
        print("Камера не найдена. Строю демо-видео из кадров датасета...")
        demo = make_demo_video(config, detector)
        cap = cv2.VideoCapture(demo)
        fps = 3.0

    out = None
    if config.get("save_video", False):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter("alerts/record.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_time = time.perf_counter()
    print("Нажмите 'q' для выхода")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            analyze(frame, detector, config)

            now = time.perf_counter()
            dt = now - frame_time
            frame_time = now
            if dt > 0:
                cv2.putText(frame, f"FPS: {1 / dt:4.1f}", (15, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if out is not None:
                out.write(frame)

            cv2.imshow("Anti-Terror", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()