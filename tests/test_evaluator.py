"""
tests/test_evaluator.py
Run with: pytest tests/test_evaluator.py -v
"""

import json
import pytest
from pathlib import Path
from src.training.evaluator import (
    extract_metric,
    save_evaluation_summary,
    print_comparison_table,
)


@pytest.fixture
def dummy_results():
    return {
        "car":           {"ap50": 0.776, "precision": 0.773, "recall": 0.710},
        "pedestrian":    {"ap50": 0.548, "precision": 0.679, "recall": 0.475},
        "traffic light": {"ap50": 0.561, "precision": 0.664, "recall": 0.544},
        "traffic sign":  {"ap50": 0.637, "precision": 0.716, "recall": 0.581},
        "_summary": {
            "mAP50":     0.631,
            "mAP50_95":  0.318,
            "precision": 0.708,
            "recall":    0.577,
        },
    }


def test_extract_metric_basic():
    rows = [
        {"metrics/mAP50(B)": "0.416"},
        {"metrics/mAP50(B)": "0.515"},
        {"metrics/mAP50(B)": "0.631"},
    ]
    vals = extract_metric(rows, "metrics/mAP50(B)")
    assert vals == pytest.approx([0.416, 0.515, 0.631])


def test_extract_metric_missing_column():
    rows = [{"other_col": "0.5"}]
    vals = extract_metric(rows, "metrics/mAP50(B)")
    assert vals == []


def test_save_evaluation_summary(tmp_path, dummy_results):
    path = save_evaluation_summary(
        results=dummy_results,
        model_name="test_model",
        output_dir=str(tmp_path),
    )
    assert Path(path).exists()
    with open(path) as f:
        data = json.load(f)
    assert data["model"] == "test_model"
    assert "car" in data["per_class"]
    assert data["summary"]["mAP50"] == pytest.approx(0.631)


def test_summary_excludes_underscore_keys(tmp_path, dummy_results):
    path = save_evaluation_summary(
        results=dummy_results,
        model_name="test_model",
        output_dir=str(tmp_path),
    )
    with open(path) as f:
        data = json.load(f)
    assert "_summary" not in data["per_class"]


def test_print_comparison_table_no_crash():
    summaries = [
        {"model": "yolov8s_30ep", "mAP50": 0.631,
         "mAP50_95": 0.318, "precision": 0.708,
         "recall": 0.577, "fps": 117.6},
    ]
    # Should not raise
    print_comparison_table(summaries)