"""
rules/red_light.py

Red light violation detection rule.

STATUS: Stub / scaffold — demonstrates the extensibility pattern.

To make this fully functional you would need:
  1. A traffic light state detector (red/yellow/green classification)
     — add as a secondary model or use the existing traffic light bbox
       plus colour analysis of the bbox region
  2. A spatial relationship check — is a vehicle bbox overlapping
     or past the stop line while light is red?

Even as a stub this registers and runs cleanly —
it just never triggers until the logic is implemented.
"""

from __future__ import annotations

from src.analytics.rule_engine import Rule, Event
from src.analytics.stats_collector import Detection, FrameStats


class RedLightRule(Rule):
    """
    Detects vehicles crossing an intersection during a red light.

    Future implementation steps:
      1. Crop the traffic light bbox from the frame
      2. Classify its colour (red/yellow/green) using HSV thresholds
         or a small secondary classifier
      3. If red: check whether any vehicle bbox centroid is past
         a configurable stop-line y-coordinate
      4. Emit a 'critical' Event with the offending vehicle's track_id
    """

    def __init__(self, stop_line_y: float = 0.6):
        """
        Args:
            stop_line_y: Normalised y-coordinate of the stop line [0,1].
                         Vehicles with y_center > this value AND red light
                         active would be flagged. Default 0.6 (lower 40%).
        """
        self.stop_line_y      = stop_line_y
        self._consecutive_red = 0    # frames with red light detected
        self._min_red_frames  = 3    # must see red for N frames before flagging

    @property
    def name(self) -> str:
        return "red_light_violation"

    @property
    def description(self) -> str:
        return (
            "Detects vehicles crossing an intersection while the "
            "traffic light is red. Requires traffic light state "
            "classification (stub — not yet implemented)."
        )

    def evaluate(
        self,
        detections: list[Detection],
        frame_stats: FrameStats,
    ) -> list[Event]:
        """
        Stub implementation — returns no events until logic is added.

        When implemented, this will:
          1. Find traffic light detections
          2. Classify light colour
          3. Find vehicle detections past the stop line
          4. Emit critical events for violations
        """
        # Step 1: Find traffic lights
        traffic_lights = [
            d for d in detections
            if d.class_name == "traffic light"
        ]

        if not traffic_lights:
            self._consecutive_red = 0
            return []

        # Step 2: Classify light colour 
        # TODO: implement HSV colour classification on the
        # cropped bounding box region from the actual frame.
        # For now we cannot determine red/green without the
        # actual pixel data — stub returns nothing.
        light_is_red = False   # placeholder

        if not light_is_red:
            self._consecutive_red = 0
            return []

        self._consecutive_red += 1
        if self._consecutive_red < self._min_red_frames:
            return []

        # Step 3: Find violating vehicles 
        vehicles = [
            d for d in detections
            if d.class_name in ("car", "truck", "bus", "motorcycle")
            and d.y_center > self.stop_line_y
        ]

        if not vehicles:
            return []

        # Step 4: Emit events 
        events = []
        for vehicle in vehicles:
            events.append(Event(
                rule_name    = self.name,
                severity     = "critical",
                message      = (
                    f"Red light violation: {vehicle.class_name} "
                    f"(track_id={vehicle.track_id}) past stop line"
                ),
                frame_idx    = frame_stats.frame_idx,
                timestamp_ms = frame_stats.timestamp_ms,
                detections   = [vehicle],
                metadata     = {
                    "stop_line_y":  self.stop_line_y,
                    "vehicle_y":    vehicle.y_center,
                    "track_id":     vehicle.track_id,
                },
            ))
        return events

    def reset(self) -> None:
        self._consecutive_red = 0


# Additional rule stubs for future implementation 

class CrowdingRule(Rule):
    """
    Triggers when more than N pedestrians are detected in a frame.
    Useful for crowd density monitoring at intersections.
    """

    def __init__(self, threshold: int = 10):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "crowding_alert"

    @property
    def description(self) -> str:
        return f"Alerts when more than {self.threshold} pedestrians detected."

    def evaluate(
        self,
        detections: list[Detection],
        frame_stats: FrameStats,
    ) -> list[Event]:
        pedestrian_count = frame_stats.counts.get("pedestrian", 0)

        if pedestrian_count <= self.threshold:
            return []

        return [Event(
            rule_name    = self.name,
            severity     = "warning",
            message      = (
                f"High pedestrian density: {pedestrian_count} "
                f"pedestrians detected (threshold: {self.threshold})"
            ),
            frame_idx    = frame_stats.frame_idx,
            timestamp_ms = frame_stats.timestamp_ms,
            detections   = [d for d in detections
                            if d.class_name == "pedestrian"],
            metadata     = {"count": pedestrian_count,
                            "threshold": self.threshold},
        )]


class HighTrafficRule(Rule):
    """
    Triggers when total vehicle count exceeds a threshold.
    Useful for congestion detection.
    """

    def __init__(self, threshold: int = 15):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "high_traffic_alert"

    @property
    def description(self) -> str:
        return f"Alerts when more than {self.threshold} vehicles detected."

    def evaluate(
        self,
        detections: list[Detection],
        frame_stats: FrameStats,
    ) -> list[Event]:
        vehicle_classes = ("car", "truck", "bus", "motorcycle")
        vehicle_count   = sum(
            frame_stats.counts.get(c, 0) for c in vehicle_classes
        )

        if vehicle_count <= self.threshold:
            return []

        return [Event(
            rule_name    = self.name,
            severity     = "info",
            message      = (
                f"High traffic: {vehicle_count} vehicles "
                f"(threshold: {self.threshold})"
            ),
            frame_idx    = frame_stats.frame_idx,
            timestamp_ms = frame_stats.timestamp_ms,
            detections   = [d for d in detections
                            if d.class_name in vehicle_classes],
            metadata     = {"vehicle_count": vehicle_count,
                            "threshold": self.threshold},
        )]