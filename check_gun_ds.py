import collections
from zipfile import ZipFile

z = ZipFile(r"C:\Users\MSI KATANA\Downloads\Gun.v1-1.yolov8.zip")
names = ["gun", "knife", "person", "pistol"]
counts = collections.Counter()
splits = collections.Counter()
labels_checked = 0
for name in z.namelist():
    if "/labels/" not in name or not name.endswith(".txt"):
        continue
    parts = name.split("/")
    split = parts[0]
    text = z.read(name).decode("utf-8", errors="replace")
    lines = [l for l in text.strip().splitlines() if l]
    if lines:
        cls = int(lines[0].split()[0])
        counts[names[cls]] += 1
        splits[split] += 1
        labels_checked += 1

print(f"меток просмотрено: {labels_checked}")
print("по классам:", dict(counts))
print("по сплитам:", dict(splits))
z.close()