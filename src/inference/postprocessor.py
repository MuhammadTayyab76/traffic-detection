import cv2

class Postprocessor:
    def draw_boxes(self,frame,detections):
        out_frame=frame.copy()
        for det in detections:
            # Reconstruct integers for cv2 drawing
            x1 = int(det.x_center - (det.width / 2))
            y1 = int(det.y_center - (det.height / 2))
            x2 = int(det.x_center + (det.width / 2))
            y2 = int(det.y_center + (det.height / 2))
            
            track_str = det.track_id if det.track_id is not None else "?"
            label=f"ID:{track_str} {det.class_name} {det.confidence:.2f}"
            
            cv2.rectangle(out_frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(out_frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            
        return out_frame