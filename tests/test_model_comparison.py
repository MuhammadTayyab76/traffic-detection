"""
Tests for feature/model-comparison-report.
All tests use synthetic evaluation JSONs — no trained weights needed.

Run from project root:
    pytest tests/test_model_comparison.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


YOLO_EVAL = {
    "model":      "yolov8s",
    "weights":    "weights/yolov8s_best.pt",
    "split":      "valid",
    "num_images": 1000,
    "map50":      0.630,
    "map50_95":   0.318,
    "precision":  0.710,
    "recall":     0.580,
    "fps":        82.4,
    "latency_ms": 12.1,
    "per_class": {
        "car":           {"ap50": 0.78},
        "bus":           {"ap50": 0.65},
        "person":        {"ap50": 0.61},
        "bike":          {"ap50": 0.42},
        "traffic light": {"ap50": 0.55},
    },
}

SSD_EVAL = {
    "model":      "ssdlite320_mobilenet_v3_large",
    "weights":    "weights/ssd_best.pt",
    "split":      "valid",
    "num_images": 1000,
    "map50":      0.510,
    "map50_95":   0.241,
    "precision":  0.620,
    "recall":     0.510,
    "fps":        118.7,
    "latency_ms": 8.4,
    "per_class": {
        "car":           {"ap50": 0.69},
        "bus":           {"ap50": 0.58},
        "person":        {"ap50": 0.50},
        "bike":          {"ap50": 0.35},
        "traffic light": {"ap50": 0.44},
    },
}


@pytest.fixture()
def yolo_json(tmp_path: Path) -> Path:
    p = tmp_path / "yolo_eval.json"
    p.write_text(json.dumps(YOLO_EVAL))
    return p


@pytest.fixture()
def ssd_json(tmp_path: Path) -> Path:
    p = tmp_path / "ssd_eval.json"
    p.write_text(json.dumps(SSD_EVAL))
    return p


@pytest.fixture()
def comparison(yolo_json, ssd_json):
    from scripts.compare_models import build_comparison, _load
    yolo = _load(str(yolo_json))
    ssd  = _load(str(ssd_json))
    return build_comparison(yolo, ssd)


class TestLoadFunction:

    def test_load_valid_json(self, yolo_json):
        from scripts.compare_models import _load
        data = _load(str(yolo_json))
        assert data["model"] == "yolov8s"

    def test_load_missing_file_raises(self, tmp_path):
        from scripts.compare_models import _load
        with pytest.raises(FileNotFoundError):
            _load(str(tmp_path / "nonexistent.json"))


class TestBuildComparison:

    def test_returns_dict(self, comparison):
        assert isinstance(comparison, dict)

    def test_top_level_keys(self, comparison):
        for key in ("models", "metrics", "per_class",
                    "overall_winner_count", "recommendation"):
            assert key in comparison, f"Missing key: {key}"

    def test_metrics_keys_present(self, comparison):
        for key in ("map50", "map50_95", "precision", "recall", "fps", "latency_ms"):
            assert key in comparison["metrics"], f"Missing metric: {key}"

    def test_each_metric_has_required_fields(self, comparison):
        for key, vals in comparison["metrics"].items():
            for field in ("label", "yolov8s", "ssdlite", "winner", "difference"):
                assert field in vals, f"Metric '{key}' missing field '{field}'"

    def test_winner_is_valid_model_name(self, comparison):
        valid = {"YOLOv8s", "SSDLite"}
        for key, vals in comparison["metrics"].items():
            assert vals["winner"] in valid, \
                f"Metric '{key}' winner '{vals['winner']}' is not a valid model name"

    def test_difference_is_non_negative(self, comparison):
        for key, vals in comparison["metrics"].items():
            assert vals["difference"] >= 0, \
                f"Metric '{key}' has negative difference"

    def test_map50_winner_is_yolo(self, comparison):
        assert comparison["metrics"]["map50"]["winner"] == "YOLOv8s", \
            "YOLOv8s has higher mAP@0.5 in fixture, should win"

    def test_fps_winner_is_ssd(self, comparison):
        assert comparison["metrics"]["fps"]["winner"] == "SSDLite", \
            "SSDLite is faster in fixture, should win FPS"

    def test_latency_winner_is_ssd(self, comparison):
        assert comparison["metrics"]["latency_ms"]["winner"] == "SSDLite", \
            "SSDLite has lower latency in fixture, should win"

    def test_per_class_covers_all_classes(self, comparison):
        expected = set(YOLO_EVAL["per_class"].keys()) | set(SSD_EVAL["per_class"].keys())
        actual   = set(comparison["per_class"].keys())
        assert actual == expected

    def test_per_class_winner_is_valid(self, comparison):
        valid = {"YOLOv8s", "SSDLite"}
        for cls, vals in comparison["per_class"].items():
            assert vals["winner"] in valid, \
                f"Class '{cls}' winner '{vals['winner']}' is invalid"

    def test_overall_winner_count_sums_to_metric_count(self, comparison):
        counts    = comparison["overall_winner_count"]
        total     = counts["yolov8s"] + counts["ssdlite"]
        n_metrics = len(comparison["metrics"])
        assert total == n_metrics, \
            f"Winner counts ({total}) do not sum to metric count ({n_metrics})"

    def test_recommendation_is_valid_model(self, comparison):
        assert comparison["recommendation"] in ("YOLOv8s", "SSDLite")

    def test_recommendation_matches_most_wins(self, comparison):
        counts = comparison["overall_winner_count"]
        expected = "YOLOv8s" if counts["yolov8s"] >= counts["ssdlite"] else "SSDLite"
        assert comparison["recommendation"] == expected


class TestOutputFile:

    def test_comparison_saves_to_json(self, yolo_json, ssd_json, tmp_path):
        from scripts.compare_models import build_comparison, _load
        out_path = tmp_path / "comparison_report.json"
        yolo = _load(str(yolo_json))
        ssd  = _load(str(ssd_json))
        result = build_comparison(yolo, ssd)
        out_path.write_text(json.dumps(result, indent=2))
        assert out_path.exists()

    def test_saved_json_is_valid(self, yolo_json, ssd_json, tmp_path):
        from scripts.compare_models import build_comparison, _load
        out_path = tmp_path / "comparison_report.json"
        yolo = _load(str(yolo_json))
        ssd  = _load(str(ssd_json))
        result = build_comparison(yolo, ssd)
        out_path.write_text(json.dumps(result, indent=2))
        loaded = json.loads(out_path.read_text())
        assert "recommendation" in loaded

    def test_evaluation_outputs_dir_exists(self):
        assert Path("evaluation_outputs").exists(), \
            "evaluation_outputs/ directory missing"


class TestSSDEvalSchema:

    def test_ssd_eval_schema_has_required_keys(self, ssd_json):
        data = json.loads(ssd_json.read_text())
        for key in ("model", "map50", "map50_95", "precision",
                    "recall", "fps", "latency_ms", "per_class"):
            assert key in data, f"SSD eval JSON missing key: {key}"

    def test_yolo_eval_schema_has_required_keys(self, yolo_json):
        data = json.loads(yolo_json.read_text())
        for key in ("model", "map50", "map50_95", "precision",
                    "recall", "fps", "latency_ms", "per_class"):
            assert key in data, f"YOLO eval JSON missing key: {key}"