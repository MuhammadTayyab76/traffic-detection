import torch
import torchvision
from ultralytics import YOLO
from src.models.base_detector import BaseDetector
from src.analytics.stats_collector import Detection

class YOLODetector(BaseDetector):
    def load_model(self):
        self.model=YOLO(self.weights_path)

    def detect(self,frame):
        results=self.model(frame,verbose=False)
        detections=[]
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0].tolist()
                conf=float(box.conf[0])
                cls_id=int(box.cls[0])
                class_name=self.model.names[cls_id]
                
                w = x2 - x1
                h = y2 - y1
                x_c = x1 + (w / 2)
                y_c = y1 + (h / 2)
                
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    x_center=x_c,
                    y_center=y_c,
                    width=w,
                    height=h
                ))
        return detections

class SSDDetector(BaseDetector):
    def load_model(self):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=torchvision.models.detection.ssd300_vgg16(weights=None,num_classes=11)
        self.model.load_state_dict(torch.load(self.weights_path,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.classes=[
            'background','bike','bus','car','motor','person',
            'rider','traffic light','traffic sign','train','truck'
        ]

    def detect(self,frame):
        img_tensor=torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions=self.model(img_tensor)[0]
        detections=[]
        for i in range(len(predictions['boxes'])):
            conf=float(predictions['scores'][i])
            if(conf>0.25):
                x1,y1,x2,y2=predictions['boxes'][i].tolist()
                cls_id=int(predictions['labels'][i])
                class_name=self.classes[cls_id] if cls_id<len(self.classes) else "unknown"
                detections.append({
                    'bbox':[int(x1),int(y1),int(x2),int(y2)],
                    'confidence':conf,
                    'class_id':cls_id,
                    'class_name':class_name
                })
        return detections