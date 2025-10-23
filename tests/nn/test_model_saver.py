from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.nn.model_saver import ModelSaver


@pytest.fixture
def mock_model():
    """Creates a simple mock nn.Module."""
    return nn.Linear(10, 2)


@pytest.fixture
@patch("src.nn.model_saver.datetime")
def saver_instance(mock_datetime):
    """Creates a ModelSaver instance with a fixed timestamp."""
    mock_datetime.now.return_value = datetime(2025, 10, 22, 14, 19, 20)
    return ModelSaver(filename="test_model")


def test_model_saver_init(saver_instance):
    """Test that the filepath is constructed correctly."""
    expected_path = "saved_models/test_model_2025-10-22_14-19-20.pth"
    import os

    assert os.path.normpath(saver_instance.filepath) == os.path.normpath(
        expected_path
    )


@patch("src.nn.model_saver.torch.save")
@patch(
    "src.nn.model_saver.ensure_dir_exists", return_value="Directory created."
)
def test_model_saver_save(
    mock_ensure_dir, mock_torch_save, saver_instance, mock_model
):
    """Test the save method."""
    result = saver_instance.save(mock_model)

    mock_torch_save.assert_called_once()

    call_args, _ = mock_torch_save.call_args
    saved_state_dict, saved_filepath = call_args

    assert saved_filepath == saver_instance.filepath

    original_state_dict = mock_model.state_dict()
    assert list(saved_state_dict.keys()) == list(original_state_dict.keys())
    for key in original_state_dict:
        assert torch.equal(saved_state_dict[key], original_state_dict[key])

    mock_ensure_dir.assert_called_once_with(saver_instance.dir)
    assert result == "Directory created."


@patch("src.nn.model_saver.torch.load")
def test_model_saver_load(mock_torch_load, saver_instance, mock_model):
    """Test the load method."""
    mock_state_dict = {"weight": torch.randn(2, 10), "bias": torch.randn(2)}
    mock_torch_load.return_value = mock_state_dict

    mock_model.load_state_dict = MagicMock()

    saver_instance.load(mock_model, map_location="cuda:0")

    mock_torch_load.assert_called_once_with(
        saver_instance.filepath, map_location="cuda:0"
    )
    mock_model.load_state_dict.assert_called_once_with(mock_state_dict)
