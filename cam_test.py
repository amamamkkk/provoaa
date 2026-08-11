import cv2
for backend, name in [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_ANY, "ANY")]:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"КАМЕРА ОК через {name}: {w}x{h}")
            cap.release()
            break
    cap.release()
else:
    print("Камера НЕ открылась из этой консоли")