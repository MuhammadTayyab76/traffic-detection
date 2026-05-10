import argparse
from src.inference.detector import YOLODetector, SSDDetector
from src.inference.tracker import SimpleTracker
from src.inference.postprocessor import Postprocessor
from src.analytics.stats_collector import StatsCollector
from src.ingestion.stream_handler import StreamHandler

def main():
    parser = argparse.ArgumentParser(description="Run Traffic Detection Demo")
    parser.add_argument("--model", type=str, choices=["yolo", "ssd"], default="yolo", help="Choose 'yolo' or 'ssd'")
    parser.add_argument("--video", type=str, default="test_video.mp4", help="Path to your unseen input video")
    args = parser.parse_args()

    INPUT_VIDEO = args.video
    OUTPUT_VIDEO = f"demo_output_{args.model}.mp4"
    OUTPUT_JSON = f"demo_stats_{args.model}.json"

    classes = [
        'background', 'bike', 'bus', 'car', 'motor', 'person', 
        'rider', 'traffic light', 'traffic sign', 'train', 'truck'
    ]

    print(f"Initializing components for {args.model.upper()}...")
    
    if args.model == "yolo":
        detector = YOLODetector(weights_path="weights/yolo_best.pt")
    else:
        detector = SSDDetector(weights_path="weights/ssd_best.pt")

    tracker = SimpleTracker()
    postprocessor = Postprocessor()
    stats_collector = StatsCollector(class_names=classes)

    handler = StreamHandler(
        detector=detector,
        tracker=tracker,
        postprocessor=postprocessor,
        stats_collector=stats_collector
    )

    print(f"Starting {args.model.upper()} inference on {INPUT_VIDEO}...")
    handler.process_stream(
        source_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        output_json_path=OUTPUT_JSON
    )

if __name__ == "__main__":
    main()