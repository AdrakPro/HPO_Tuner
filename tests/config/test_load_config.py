import json
import os
from unittest.mock import MagicMock, patch, mock_open

import pytest

MODULE_PATH = "src.config.load_config"  # Update if file name differs

import src.config.load_config as config_mgr


@pytest.fixture
def mock_logger(mocker):
    """Mocks the logger to prevent console spam and verify error calls."""
    return mocker.patch(f"{MODULE_PATH}.logger")


@pytest.fixture
def mock_enum_helper(mocker):
    """Mocks the get_enum_names function used in validation."""
    mock_get = mocker.patch(f"{MODULE_PATH}.get_enum_names")
    mock_get.return_value = ["A", "B", "SGD", "ADAM", "NONE", "LIGHT", "MEDIUM"]
    return mock_get


@pytest.fixture
def valid_config():
    """A fully valid configuration dictionary based on your validators."""
    return {
        "project": {"name": "TestProject", "seed": 42},
        "checkpoint_config": {"interval_per_gen": 10},
        "parallel_config": {
            "execution": {
                "evaluation_mode": "GPU",
                "enable_parallel": True,
                "gpu_workers": 2,
                "cpu_workers": 4,
                "dataloader_workers": {"per_gpu": 2, "per_cpu": 0},
            }
        },
        "neural_network_config": {
            "input_shape": [3, 32, 32],
            "conv_blocks": 3,
            "output_classes": 10,
            "base_filters": 16,
            "hyperparameter_space": {
                "lr": {"type": "float", "range": [0.001, 0.1]},
                "layers": {"type": "int", "range": [1, 5]},
                "optimizer_schedule": {
                    "type": "enum",
                    "values": ["SGD", "ADAM"],
                },
                "batch_size": {"type": "int", "values": [32, 64]},
            },
        },
        "nested_validation_config": {"enabled": True, "outer_k_folds": 5},
        "genetic_algorithm_config": {
            "genetic_operators": {
                "active": ["selection", "crossover", "mutation"],
                "selection": {"type": "tournament"},
                "crossover": {"type": "uniform", "crossover_prob": 0.5},
                "mutation": {"prob_a": 0.1},
                "elitism_percent": 0.1,
            },
            "calibration": {
                "enabled": True,
                "data_subset_percentage": 0.5,
                "population_size": 10,
                "generations": 2,
                "training_epochs": 1,
                "mutation_decay_rate": 0.9,
                "stop_conditions": {
                    "early_stop_generations": 1,
                    "early_stop_epochs": 1,
                    "time_limit_minutes": 10.0,
                    "fitness_goal": 0.95,
                },
            },
            "main_algorithm": {
                "enabled": True,
                "population_size": 20,
                "generations": 10,
                "training_epochs": 5,
                "mutation_decay_rate": 0.99,
                "stop_conditions": {
                    "early_stop_generations": 3,
                    "early_stop_epochs": 2,
                    "time_limit_minutes": 60.0,
                    "fitness_goal": 0.99,
                },
            },
        },
    }


class TestValidationHelpers:
    def test_check_non_negative_int(self):
        config_mgr._check_non_negative_int(5, "test")  # Pass
        with pytest.raises(ValueError):
            config_mgr._check_non_negative_int(-1, "test")
        with pytest.raises(ValueError):
            config_mgr._check_non_negative_int(5.5, "test")

    def test_check_float_in_range(self):
        config_mgr._check_float_in_range(0.5, "test")  # Pass
        with pytest.raises(ValueError):
            config_mgr._check_float_in_range(0.0, "test")  # > 0.0
        with pytest.raises(ValueError):
            config_mgr._check_float_in_range(1.1, "test")  # <= 1.0


class TestSanitizeConfig:

    def test_valid_config_passes(
        self, valid_config, mock_logger, mock_enum_helper
    ):
        """Ensure a correct config returns the object without exiting."""
        result = config_mgr.sanitize_config(valid_config)
        assert result == valid_config
        mock_logger.error.assert_not_called()

    def test_invalid_project_seed(self, valid_config, mock_logger):
        valid_config["project"]["seed"] = "invalid_string"
        with pytest.raises(SystemExit) as e:
            config_mgr.sanitize_config(valid_config)
        assert e.type == SystemExit
        assert e.value.code == 1
        mock_logger.error.assert_called()

    def test_invalid_execution_mode(self, valid_config, mock_logger):
        valid_config["parallel_config"]["execution"][
            "evaluation_mode"
        ] = "QUANTUM"
        with pytest.raises(SystemExit):
            config_mgr.sanitize_config(valid_config)

    def test_neural_network_range_logic(self, valid_config, mock_logger):
        """Test min >= max logic."""
        valid_config["neural_network_config"]["hyperparameter_space"]["lr"][
            "range"
        ] = [10.0, 1.0]
        with pytest.raises(SystemExit):
            config_mgr.sanitize_config(valid_config)

    def test_neural_network_enum_validation(
        self, valid_config, mock_enum_helper, mock_logger
    ):
        """Test checking against allowed enum values."""
        valid_config["neural_network_config"]["hyperparameter_space"][
            "optimizer_schedule"
        ]["values"] = ["INVALID_OPT"]

        with pytest.raises(SystemExit):
            config_mgr.sanitize_config(valid_config)

    def test_missing_key(self, valid_config, mock_logger):
        """Test that missing a top-level key causes exit."""
        del valid_config["project"]
        with pytest.raises(SystemExit):
            config_mgr.sanitize_config(valid_config)
        mock_logger.error.assert_called()


class TestFileOperations:

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_prompt_and_load_success(
        self,
        mock_exists,
        mock_file,
        valid_config,
        mock_logger,
        mock_enum_helper,
    ):
        """Simulate user typing a filename and successfully loading it."""
        mock_console = MagicMock()
        mock_console.input.return_value = "my_config.json"

        mock_file.return_value.read.return_value = json.dumps(valid_config)

        result = config_mgr.prompt_and_load_json_config(
            default_config={}, console=mock_console, config_dir="/tmp"
        )

        assert result == valid_config
        mock_console.input.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch(
        "os.path.exists", side_effect=[False, True]
    )  # First fail, then succeed
    def test_prompt_and_load_retry(
        self,
        mock_exists,
        mock_file,
        valid_config,
        mock_logger,
        mock_enum_helper,
    ):
        """Simulate entering a wrong filename first, then a correct one."""
        mock_console = MagicMock()
        mock_console.input.side_effect = ["bad_file.json", "good_file.json"]

        mock_file.return_value.read.return_value = json.dumps(valid_config)

        result = config_mgr.prompt_and_load_json_config(
            {}, mock_console, "/tmp"
        )

        assert result == valid_config
        mock_logger.error.assert_called()
        assert mock_console.input.call_count == 2

    @patch("os.listdir")
    @patch("os.path.getmtime")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_newest_config(
        self,
        mock_file,
        mock_mtime,
        mock_listdir,
        valid_config,
        mock_enum_helper,
    ):
        """Test automatically picking the newest file."""
        mock_listdir.return_value = ["old.json", "new.json"]

        def get_mtime_side_effect(path):
            if "new.json" in path:
                return 200
            return 100

        mock_mtime.side_effect = get_mtime_side_effect

        mock_file.return_value.read.return_value = json.dumps(valid_config)

        result = config_mgr.load_newest_config({}, "/tmp")

        assert result == valid_config
        args, _ = mock_file.call_args
        assert "new.json" in args[0]

    def test_load_newest_no_files(self):
        """Test fallback to default if dir is empty."""
        with patch("os.listdir", return_value=[]):
            default = {"a": 1}
            result = config_mgr.load_newest_config(default, "/tmp")
            assert result == default

    @patch("builtins.open", new_callable=mock_open)
    @patch("src.config.load_config.datetime")
    def test_save_config(self, mock_dt, mock_file, valid_config, mock_logger):
        """Test saving configuration to file."""
        mock_console = MagicMock()
        mock_console.input.side_effect = ["y", ""]

        mock_dt.now.return_value.strftime.return_value = "2023_TEST"

        config_mgr.prompt_and_save_json_config(
            valid_config, mock_console, "/tmp"
        )

        expected_path = os.path.join("/tmp", "config_2023_TEST.json")

        mock_file.assert_called_with(expected_path, "w")
        mock_logger.info.assert_called_with(
            f"Configuration has been saved to '{expected_path}'"
        )
