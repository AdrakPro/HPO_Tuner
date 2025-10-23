from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from src.config.default_config import get_default_config
from src.tui.tui_configurator import (
    _prompt_for_numeric,
    _get_parallel_config,
    _get_hyperparameter_config,
    _deep_merge_dicts,
    _get_nested_config,
)


@pytest.fixture
def default_config_fixture():
    """Provides a deep copy of the default config for each test."""
    return deepcopy(get_default_config())


@pytest.fixture
def mock_console(monkeypatch):
    """Mocks the console object to control input and check output."""
    mock = MagicMock()
    monkeypatch.setattr("src.tui.tui_configurator.console", mock)
    return mock


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """Mocks all external and file-based dependencies."""
    monkeypatch.setattr(
        "src.tui.tui_configurator._detect_hardware_resources",
        MagicMock(
            return_value={
                "max_cpu_workers": 8,
                "gpus_available": True,
                "max_gpu_workers": 2,
            }
        ),
    )

    monkeypatch.setattr("src.tui.tui_configurator.logger", MagicMock())

    monkeypatch.setattr(
        "src.tui.tui_configurator.ensure_dir_exists", MagicMock(return_value="")
    )
    monkeypatch.setattr(
        "src.tui.tui_configurator.prompt_and_save_json_config", MagicMock()
    )
    monkeypatch.setattr(
        "src.tui.tui_configurator.load_newest_config", MagicMock()
    )
    monkeypatch.setattr(
        "src.tui.tui_configurator.prompt_and_load_json_config", MagicMock()
    )


def test_prompt_for_numeric_valid_input(mock_console):
    mock_console.input.return_value = "10"
    result = _prompt_for_numeric("Prompt", default_value=5, value_type=int)
    assert result == 10


def test_prompt_for_numeric_default_on_enter(mock_console):
    mock_console.input.return_value = ""
    result = _prompt_for_numeric("Prompt", default_value=5, value_type=int)
    assert result is None


def test_prompt_for_numeric_invalid_input(mock_console):
    mock_console.input.return_value = "abc"
    result = _prompt_for_numeric("Prompt", default_value=5, value_type=int)
    assert result is None
    mock_console.print.assert_called_with(
        "[yellow]\nInvalid value. Using default: 5.[/yellow]"
    )


def test_prompt_for_numeric_negative_when_positive_only(mock_console):
    mock_console.input.return_value = "-5"
    result = _prompt_for_numeric(
        "Prompt", default_value=5, value_type=int, positive_only=True
    )
    assert result is None
    mock_console.print.assert_called_with(
        "[yellow]Value must be a positive non-zero number. Using default: 5.[/yellow]"
    )


def test_get_nested_config(default_config_fixture):
    """Test safe retrieval of a nested dictionary value."""
    path = [
        "genetic_algorithm_config",
        "genetic_operators",
        "selection",
        "tournament_size",
    ]
    value = _get_nested_config(default_config_fixture, path)
    assert value == 7

    invalid_path = ["path", "does", "not", "exist"]
    value = _get_nested_config(
        default_config_fixture, invalid_path, default="default"
    )
    assert value == "default"


def test_deep_merge_dicts():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    updates = {"b": {"c": 4, "e": 5}, "f": 6}
    expected = {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}
    assert _deep_merge_dicts(base, updates) == expected


def test_get_parallel_config_hybrid_mode(mock_console, default_config_fixture):
    mock_console.input.side_effect = ["3", "4", "1", "2", "1"]

    updates = _get_parallel_config(default_config_fixture)

    expected = {
        "parallel_config": {
            "execution": {
                "evaluation_mode": "HYBRID",
                "enable_parallel": True,
                "cpu_workers": 4,
                "gpu_workers": 1,
                "dataloader_workers": {
                    "per_gpu": 2,
                    "per_cpu": 1,
                },
            }
        }
    }
    assert updates == expected


def test_get_hyperparameter_config_change_range(
    mock_console, default_config_fixture
):
    inputs = ["y", "0.1-0.5"] + ["n"] * (
        len(
            default_config_fixture["neural_network_config"][
                "hyperparameter_space"
            ]
        )
        - 1
    )
    mock_console.input.side_effect = inputs

    updates = _get_hyperparameter_config(default_config_fixture)

    expected_update = {
        "neural_network_config": {
            "hyperparameter_space": {"width_scale": {"range": [0.1, 0.5]}}
        }
    }
    assert updates == expected_update
    mock_console.print.assert_any_call(
        "  [green]Updated range for width_scale to [0.1, 0.5].[/green]"
    )
