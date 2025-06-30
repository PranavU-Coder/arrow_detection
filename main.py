from ultralytics import YOLO

model = YOLO("yolov8m.yaml")

results = model.train(data="config.yaml", epochs=150)
