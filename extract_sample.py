from zipfile import ZipFile

z = ZipFile(r"C:\Users\MSI KATANA\Downloads\Gun.v1-1.yolov8.zip")
target = "train/images/1-1-t_jpg.rf.01837c1206255e04bc3f2d8325f62361.jpg"
data = z.read(target)
with open(r"C:\Users\MSI KATANA\Documents\Default Project\anti_terror\test_out\gun_sample.jpg", "wb") as f:
    f.write(data)
print("saved", len(data), "bytes")
z.close()