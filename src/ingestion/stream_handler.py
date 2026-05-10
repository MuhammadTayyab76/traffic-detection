import cv2
from src.ingestion.video_loader import load_video, get_video_info

class StreamHandler:
    def __init__(self, detector, tracker, postprocessor, stats_collector):
        #The handler receives the initialized components from the main script
        self.detector = detector
        self.tracker = tracker
        self.postprocessor = postprocessor
        self.stats_collector = stats_collector

    def process_stream(self, source_path, output_video_path, output_json_path):
        #1. Wire the video_loader.py
        print(f"Opening video source: {source_path}")
        cap = load_video(source_path)
        video_info = get_video_info(cap)
        
        #Setup the video writer (Output File)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, video_info['fps'], (video_info['width'], video_info['height']))

        frame_id = 0
        print("Processing frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_id += 1
            
            #2. Wire the detector.py
            detections = self.detector.detect(frame)
            
            #3. Wire the tracker.py
            tracked_objects = self.tracker.update(detections)
            
            #4. Wire the stats_collector.py
            self.stats_collector.update(frame_id, tracked_objects)
            
            #5. Wire the postprocessor.py
            out_frame = self.postprocessor.draw_boxes(frame, tracked_objects)
            
            #6. Write to output file
            out.write(out_frame)
            
            if frame_id % 100 == 0:
                print(f"Processed {frame_id}/{video_info['total_frames']} frames...")

        cap.release()
        out.release()
        
        self.stats_collector.export(output_json_path)
        print(f"Processing complete! Video saved to {output_video_path} and stats to {output_json_path}")