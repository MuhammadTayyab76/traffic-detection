"""
Tests for feature/extended-training.
Validates that hyperparams.yaml is correctly configured for the
80-epoch run and that the trainer reads every field without error.

No GPU needed — no actual training is performed.

Run from project root:
    pytest tests/test_extended_training.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


HYPERPARAMS_PATH = Path("src/training/hyperparams.yaml")
DATASET_YAML     = Path("configs/dataset.yaml")


@pytest.fixture(scope="module")
def hp() -> dict:
    with open(HYPERPARAMS_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def dataset_cfg() -> dict:
    with open(DATASET_YAML) as f:
        return yaml.safe_load(f)


class TestHyperparamsStructure:

    def test_hyperparams_file_exists(self):
        assert HYPERPARAMS_PATH.exists(), \
            f"{HYPERPARAMS_PATH} not found"

    def test_top_level_keys_present(self, hp):
        for key in ("model", "dataset", "training", "output"):
            assert key in hp, f"Missing top-level key: {key}"

    def test_model_keys_present(self, hp):
        for key in ("architecture", "pretrained"):
            assert key in hp["model"], f"Missing model key: {key}"

    def test_training_keys_present(self, hp):
        required = (
            "epochs", "batch_size", "patience", "workers",
            "device", "optimizer", "lr0", "lrf", "momentum",
            "weight_decay", "warmup_epochs", "cos_lr",
            "mosaic", "fliplr",
        )
        for key in required:
            assert key in hp["training"], f"Missing training key: {key}"

    def test_output_keys_present(self, hp):
        for key in ("project", "name", "save_period", "weights_dir"):
            assert key in hp["output"], f"Missing output key: {key}"


class TestHyperparamsValues:

    def test_architecture_is_yolov8s(self, hp):
        assert hp["model"]["architecture"] == "yolov8s.pt", \
            "Model must stay as yolov8s.pt for this project"

    def test_pretrained_is_true(self, hp):
        assert hp["model"]["pretrained"] is True

    def test_epochs_is_80(self, hp):
        assert hp["training"]["epochs"] == 80, \
            f"Expected 80 epochs, got {hp['training']['epochs']}"

    def test_patience_is_15(self, hp):
        assert hp["training"]["patience"] == 15, \
            f"Expected patience 15, got {hp['training']['patience']}"

    def test_batch_size_is_16(self, hp):
        assert hp["training"]["batch_size"] == 16

    def test_image_size_is_640(self, hp):
        assert hp["dataset"]["image_size"] == 640

    def test_cos_lr_is_enabled(self, hp):
        assert hp["training"]["cos_lr"] is True, \
            "Cosine LR schedule must be enabled for 80-epoch run"

    def test_lr0_in_valid_range(self, hp):
        lr0 = hp["training"]["lr0"]
        assert 1e-4 <= lr0 <= 1e-1, \
            f"lr0={lr0} is outside the expected range [1e-4, 1e-1]"

    def test_lrf_is_valid_multiplier(self, hp):
        lrf = hp["training"]["lrf"]
        assert 0.0 < lrf <= 1.0, \
            f"lrf={lrf} must be a fraction in (0, 1] — it is a multiplier of lr0, not an absolute value"

    def test_weight_decay_in_valid_range(self, hp):
        wd = hp["training"]["weight_decay"]
        assert 1e-6 <= wd <= 1e-2, \
            f"weight_decay={wd} is outside expected range"

    def test_warmup_epochs_less_than_total(self, hp):
        assert hp["training"]["warmup_epochs"] < hp["training"]["epochs"], \
            "warmup_epochs must be less than total epochs"

    def test_mosaic_augmentation_enabled(self, hp):
        assert hp["training"]["mosaic"] > 0.0, \
            "Mosaic augmentation should be enabled"

    def test_fliplr_is_valid_probability(self, hp):
        fliplr = hp["training"]["fliplr"]
        assert 0.0 <= fliplr <= 1.0, \
            f"fliplr={fliplr} is not a valid probability"

    def test_save_period_is_multiple_of_epochs(self, hp):
        sp = hp["output"]["save_period"]
        ep = hp["training"]["epochs"]
        assert ep % sp == 0, \
            f"save_period={sp} does not divide evenly into epochs={ep}"

    def test_run_name_reflects_epoch_count(self, hp):
        name = hp["output"]["name"]
        assert "80" in name, \
            f"Run name '{name}' should contain '80' to match epoch count"

    def test_run_name_contains_model(self, hp):
        name = hp["output"]["name"]
        assert "yolov8s" in name.lower(), \
            f"Run name '{name}' should contain model identifier"

    def test_weights_dir_is_set(self, hp):
        assert hp["output"]["weights_dir"], \
            "weights_dir must not be empty"

    def test_optimizer_is_valid(self, hp):
        assert hp["training"]["optimizer"] in ("AdamW", "Adam", "SGD"), \
            f"Unrecognised optimizer: {hp['training']['optimizer']}"


class TestDatasetAlignment:

    def test_dataset_yaml_exists(self):
        assert DATASET_YAML.exists()

    def test_dataset_config_path_in_hyperparams_matches_file(self, hp):
        cfg_path = Path(hp["dataset"]["config"])
        assert cfg_path.exists(), \
            f"dataset.config in hyperparams points to missing file: {cfg_path}"

    def test_nc_is_10(self, dataset_cfg):
        assert dataset_cfg["nc"] == 10, \
            f"Expected nc=10, got {dataset_cfg['nc']}"

    def test_names_count_matches_nc(self, dataset_cfg):
        assert len(dataset_cfg["names"]) == dataset_cfg["nc"], \
            "Number of class names does not match nc"


class TestTrainerImport:

    def test_trainer_module_importable(self):
        from src.training.trainer import Trainer
        assert Trainer is not None

    def test_trainer_loads_hyperparams_without_error(self):
        from src.training.trainer import Trainer
        trainer = Trainer(str(HYPERPARAMS_PATH))
        assert trainer is not None

    def test_trainer_exposes_hp_dict(self):
        from src.training.trainer import Trainer
        trainer = Trainer(str(HYPERPARAMS_PATH))
        assert hasattr(trainer, "hp")
        assert isinstance(trainer.hp, dict)

    def test_trainer_epoch_count_matches_yaml(self):
        from src.training.trainer import Trainer
        trainer = Trainer(str(HYPERPARAMS_PATH))
        assert trainer.hp["training"]["epochs"] == 80