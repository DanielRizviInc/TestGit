from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8m.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='sanitize.yaml', epochs=25, imgsz=640)