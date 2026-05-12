import cv2
from src.ingestion.video_loader import load_video, get_video_info

class StreamHandler:
    def __init__(self, detector, tracker, postprocessor, stats_collector):
        self.detector = detector
        self.tracker = tracker
        self.postprocessor = postprocessor
        self.stats_collector = stats_collector

    def process_stream(self, source_path, output_video_path, output_json_path):
        print(f"Opening video source: {source_path}")
        cap = load_video(source_path)
        video_info = get_video_info(cap)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, video_info['fps'], (video_info['width'], video_info['height']))

        frame_id = 0
        print("Processing frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_id += 1
            
            # 1. Detect
            detections = self.detector.detect(frame)
            
            # 2. Track
            tracked_objects = self.tracker.update(detections)
            
            # 3. Log Stats
            self.stats_collector.update(tracked_objects)
            
            # 4. Draw Boxes
            out_frame = self.postprocessor.draw_boxes(frame, tracked_objects)
            
            # 5. Save Frame
            out.write(out_frame)
            
            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{video_info['total_frames']} frames...")

        # Cleanup
        cap.release()
        out.release()
        
        self.stats_collector.save_log(output_json_path)
        print(f"Processing complete! Video saved to {output_video_path} and stats to {output_json_path}")