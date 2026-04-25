"""
tests/test_exporter.py
Run with: pytest tests/test_exporter.py -v
"""

import pytest
from pathlib import Path
from src.inference.exporter import (
    export_model,
    verify_onnx,
    SUPPORTED_FORMATS,
)


def test_supported_formats_not_empty():
    """SUPPORTED_FORMATS must contain at least onnx and torchscript."""
    assert "onnx" in SUPPORTED_FORMATS
    assert "torchscript" in SUPPORTED_FORMATS


def test_export_model_missing_weights():
    """Should raise FileNotFoundError for non-existent weights."""
    with pytest.raises(FileNotFoundError):
        export_model(
            weights_path="weights/nonexistent.pt",
            export_format="onnx",
        )


def test_export_model_unsupported_format():
    """Should raise ValueError for unsupported export format."""
    with pytest.raises(ValueError, match="Unsupported format"):
        export_model(
            weights_path="weights/bdd100k_yolov8s_30ep_best.pt",
            export_format="coreml",
        )


def test_verify_onnx_missing_file():
    """verify_onnx should return False for a missing file."""
    result = verify_onnx("weights/nonexistent.onnx")
    assert result is False