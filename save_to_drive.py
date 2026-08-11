# Сохранить модель на Google Drive (ищет по всем возможным путям)
from google.colab import drive
drive.mount('/content/drive')

import glob
import shutil

# все возможные пути до .pt
candidates = []
candidates += glob.glob("/content/weights/*.pt")
candidates += glob.glob("/content/runs/detect/*/weights/*.pt")
candidates += glob.glob("/content/*.pt")
candidates = sorted(set(candidates))
# в приоритете best.pt, потом веса с большим числом классов
candidates.sort(key=lambda p: (0 if "best" in p else 1, -len(p)))

print("Найдено .pt:")
for p in candidates:
    print(" ", p)

if not candidates:
    print("НЕТ .pt файлов! Обучение не завершилось или рантайм перезапущен.")
else:
    src = candidates[0]
    dst = "/content/drive/MyDrive/weapon_detection_24cls.pt"
    shutil.copy(src, dst)
    print(f"Сохранено: {src}")
    print("  ->", dst)
