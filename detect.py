import argparse

import cv2

from detector import WeaponDetector, analyze, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Детекция на файле (фото/видео)")
    parser.add_argument("input", type=str, help="Путь к фото или видео")
    parser.add_argument("--output", type=str, default=None,
                        help="Куда сохранить результат (по умолчанию input_annotated.*)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = WeaponDetector(config)
    cap = cv2.VideoCapture(args.input)

    is_video = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1
    out = None
    if is_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dest = args.output or f"{args.input.rsplit('.', 1)[0]}_annotated.mp4"
        out = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        analyze(frame, detector, config)
        if is_video:
            out.write(frame)
        else:
            dest = args.output or f"{args.input.rsplit('.', 1)[0]}_annotated.jpg"
            cv2.imwrite(dest, frame)
            print(f"Сохранено: {dest}")
            break

    cap.release()
    if out is not None:
        out.release()
        print(f"Сохранено: {dest}")
    print("Готово")


if __name__ == "__main__":
    main()
