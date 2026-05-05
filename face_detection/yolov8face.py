import cv2
from ultralytics import YOLO

class YOLOv8Face:
    def __init__(self, model_path, conf=0.4, device="cpu"):
        """
        model_path: path to face model weights
        conf: confidence threshold
        device: 'cpu' or 'cuda'
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf

    def detect(self, img):
        """
        Runs detection on an image
        Returns: list of detections (x1, y1, x2, y2, conf)
        """
        results = self.model(img, conf=self.conf)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

        return detections

    def draw(self, frame, detections):
        """
        Draw bounding boxes on frame
        """
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"Face {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)
        return frame

    def detect_and_draw(self, frame):
        """
        Convenience method
        """
        detections = self.detect(frame)
        return self.draw(frame, detections)
