import cv2

class Postprocessor:
    def draw_boxes(self,frame,detections):
        out_frame=frame.copy()
        for det in detections:
            x1,y1,x2,y2=det['bbox']
            label=f"ID:{det.get('track_id','?')} {det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(out_frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(out_frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        return out_frame