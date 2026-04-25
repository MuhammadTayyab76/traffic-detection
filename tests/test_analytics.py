"""
tests/test_analytics.py
Run with: pytest tests/test_analytics.py -v
"""

import time
import pytest
from src.analytics.stats_collector import Detection, FrameStats, StatsCollector
from src.analytics.rule_engine import Event, Rule, RuleEngine
from src.analytics.rules.red_light import CrowdingRule, HighTrafficRule


# Fixtures 

CLASS_NAMES = ["car", "pedestrian", "traffic light", "traffic sign"]


def make_detection(class_name: str, confidence: float = 0.9,
                   track_id: int = None) -> Detection:
    class_id = CLASS_NAMES.index(class_name)
    return Detection(
        class_id   = class_id,
        class_name = class_name,
        confidence = confidence,
        x_center   = 0.5,
        y_center   = 0.5,
        width      = 0.1,
        height     = 0.1,
        track_id   = track_id,
    )


# StatsCollector tests 

def test_stats_collector_basic_counts():
    """update() should correctly count detections per class."""
    collector = StatsCollector(CLASS_NAMES)
    detections = [
        make_detection("car"),
        make_detection("car"),
        make_detection("pedestrian"),
    ]
    stats = collector.update(detections)
    assert stats.counts["car"]       == 2
    assert stats.counts["pedestrian"]== 1
    assert stats.total_detections    == 3


def test_stats_collector_frame_index_increments():
    """Frame index should increment on each update call."""
    collector = StatsCollector(CLASS_NAMES)
    stats1 = collector.update([])
    stats2 = collector.update([])
    assert stats1.frame_idx == 0
    assert stats2.frame_idx == 1


def test_stats_collector_session_summary():
    """Session summary should reflect accumulated counts."""
    collector = StatsCollector(CLASS_NAMES)
    collector.update([make_detection("car"), make_detection("car")])
    collector.update([make_detection("car")])
    summary = collector.get_session_summary()
    assert summary["session_counts"]["car"] == 3
    assert summary["total_frames"] == 2


def test_stats_collector_reset():
    """reset() should clear all state."""
    collector = StatsCollector(CLASS_NAMES)
    collector.update([make_detection("car")])
    collector.reset()
    summary = collector.get_session_summary()
    assert summary["total_frames"] == 0
    assert summary["session_counts"] == {}


def test_stats_collector_avg_confidence():
    """Average confidence should be computed correctly per class."""
    collector = StatsCollector(CLASS_NAMES)
    detections = [
        make_detection("car", confidence=0.8),
        make_detection("car", confidence=0.6),
    ]
    stats = collector.update(detections)
    assert stats.avg_confidence["car"] == pytest.approx(0.7, abs=0.01)


def test_stats_collector_save_log(tmp_path):
    """save_log() should write a valid JSON file."""
    import json
    collector = StatsCollector(CLASS_NAMES)
    collector.update([make_detection("car")])
    log_path = str(tmp_path / "test_log.json")
    collector.save_log(log_path)

    with open(log_path) as f:
        data = json.load(f)
    assert "session_summary" in data
    assert "frames" in data
    assert len(data["frames"]) == 1


# RuleEngine tests 

def make_frame_stats(counts: dict = None, frame_idx: int = 0) -> FrameStats:
    return FrameStats(
        frame_idx        = frame_idx,
        timestamp_ms     = 1000.0,
        total_detections = sum((counts or {}).values()),
        counts           = counts or {},
        avg_confidence   = {},
        fps              = 30.0,
    )


def test_rule_engine_register():
    """Registered rules should appear in list_rules()."""
    engine = RuleEngine()
    rule   = CrowdingRule(threshold=5)
    engine.register(rule)
    names  = [r["name"] for r in engine.list_rules()]
    assert "crowding_alert" in names


def test_rule_engine_unregister():
    """Unregistered rules should be removed."""
    engine = RuleEngine()
    engine.register(CrowdingRule(threshold=5))
    engine.unregister("crowding_alert")
    assert len(engine.list_rules()) == 0


def test_crowding_rule_triggers():
    """CrowdingRule should emit an event when threshold exceeded."""
    engine = RuleEngine()
    engine.register(CrowdingRule(threshold=3))

    detections = [make_detection("pedestrian") for _ in range(5)]
    stats      = make_frame_stats(counts={"pedestrian": 5})
    events     = engine.evaluate(detections, stats)

    assert len(events) == 1
    assert events[0].rule_name == "crowding_alert"
    assert events[0].severity  == "warning"


def test_crowding_rule_no_trigger_below_threshold():
    """CrowdingRule should not trigger below the threshold."""
    engine = RuleEngine()
    engine.register(CrowdingRule(threshold=10))

    detections = [make_detection("pedestrian") for _ in range(3)]
    stats      = make_frame_stats(counts={"pedestrian": 3})
    events     = engine.evaluate(detections, stats)
    assert events == []


def test_high_traffic_rule_triggers():
    """HighTrafficRule should fire when vehicle count exceeds threshold."""
    engine = RuleEngine()
    engine.register(HighTrafficRule(threshold=2))

    detections = [make_detection("car") for _ in range(4)]
    stats      = make_frame_stats(counts={"car": 4})
    events     = engine.evaluate(detections, stats)

    assert len(events) == 1
    assert events[0].rule_name == "high_traffic_alert"


def test_rule_engine_reset_clears_history():
    """reset() should clear the event history."""
    engine = RuleEngine()
    engine.register(CrowdingRule(threshold=1))

    detections = [make_detection("pedestrian") for _ in range(3)]
    stats      = make_frame_stats(counts={"pedestrian": 3})
    engine.evaluate(detections, stats)

    assert len(engine.get_event_history()) > 0
    engine.reset()
    assert len(engine.get_event_history()) == 0


def test_multiple_rules_same_frame():
    """Multiple rules can trigger on the same frame independently."""
    engine = RuleEngine()
    engine.register(CrowdingRule(threshold=2))
    engine.register(HighTrafficRule(threshold=1))

    detections = (
        [make_detection("pedestrian") for _ in range(3)] +
        [make_detection("car")        for _ in range(2)]
    )
    stats  = make_frame_stats(counts={"pedestrian": 3, "car": 2})
    events = engine.evaluate(detections, stats)

    rule_names = [e.rule_name for e in events]
    assert "crowding_alert"     in rule_names
    assert "high_traffic_alert" in rule_names


def test_event_summary():
    """get_event_summary() should count events per rule."""
    engine = RuleEngine()
    engine.register(CrowdingRule(threshold=1))

    for i in range(3):
        stats  = make_frame_stats(counts={"pedestrian": 5}, frame_idx=i)
        engine.evaluate([make_detection("pedestrian") for _ in range(5)], stats)

    summary = engine.get_event_summary()
    assert summary["crowding_alert"] == 3