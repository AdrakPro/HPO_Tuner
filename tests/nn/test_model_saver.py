import os
from unittest.mock import MagicMock

import pytest
import torch

from src.nn.model_saver import ModelSaver

MODULE_PATH = "src.nn.model_saver"


@pytest.fixture
def mock_env(monkeypatch):
    """Sets specific environment variables for consistent path testing."""
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "99")
    monkeypatch.setenv("DATA_DIR", "/tmp/test_project")


@pytest.fixture
def mock_datetime(mocker):
    """
    Mocks datetime to return a fixed timestamp.
    We patch the 'datetime' imported INSIDE the module under test.
    """
    mock_dt = mocker.patch(f"{MODULE_PATH}.datetime")
    mock_dt.now.return_value.strftime.return_value = "2024-01-01_12-00-00"
    return mock_dt


@pytest.fixture
def mock_file_helper(mocker):
    """Mocks the ensure_dir_exists utility."""
    return mocker.patch(
        f"{MODULE_PATH}.ensure_dir_exists", return_value="Dir Created"
    )


@pytest.fixture
def mock_torch_save(mocker):
    """Mocks torch.save to prevent actual file writing."""
    return mocker.patch(f"{MODULE_PATH}.torch.save")


@pytest.fixture
def mock_torch_load(mocker):
    """Mocks torch.load."""
    return mocker.patch(f"{MODULE_PATH}.torch.load")


@pytest.fixture
def mock_model():
    """Creates a mock PyTorch model."""
    model = MagicMock(spec=torch.nn.Module)
    model.state_dict.return_value = {"weights": [1, 2, 3]}
    return model


class TestModelSaverInit:

    def test_init_path_construction(self, mock_env, mock_datetime):
        """Verify the file path is constructed correctly using Env vars and Timestamp."""
        saver = ModelSaver(filename="my_model")

        expected_dir = "/tmp/test_project/99/saved_models"

        assert saver.saved_models_dir == expected_dir

        expected_filename = "my_model_2024-01-01_12-00-00.pth"
        expected_path = os.path.join(expected_dir, expected_filename)

        assert saver.filepath == expected_path

    def test_init_defaults(self, monkeypatch, mock_datetime):
        """Verify defaults used when Env vars are missing."""
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("DATA_DIR", raising=False)

        saver = ModelSaver(filename="test")

        assert "0/saved_models" in saver.saved_models_dir


class TestModelSaverSave:

    def test_save_workflow(
        self,
        mock_env,
        mock_datetime,
        mock_file_helper,
        mock_torch_save,
        mock_model,
    ):
        """Verify directory creation and torch.save calling."""
        saver = ModelSaver("test_model")

        result_msg = saver.save(mock_model)

        mock_file_helper.assert_called_once_with(saver.saved_models_dir)
        assert result_msg == "Dir Created"

        mock_model.state_dict.assert_called_once()

        mock_torch_save.assert_called_once_with(
            {"weights": [1, 2, 3]}, saver.filepath
        )


class TestModelSaverLoad:

    def test_load_workflow(
        self, mock_env, mock_datetime, mock_torch_load, mock_model
    ):
        """Verify loading state dict into model."""
        saver = ModelSaver("test_model")

        fake_state_dict = {"weights": [10, 20, 30]}
        mock_torch_load.return_value = fake_state_dict

        saver.load(mock_model)

        mock_torch_load.assert_called_once_with(
            saver.filepath, map_location="cpu"
        )

        mock_model.load_state_dict.assert_called_once_with(fake_state_dict)

    def test_load_with_custom_device(
        self, mock_env, mock_datetime, mock_torch_load, mock_model
    ):
        """Verify map_location is passed correctly."""
        saver = ModelSaver("test_model")

        saver.load(mock_model, map_location="cuda:0")

        mock_torch_load.assert_called_once_with(
            saver.filepath, map_location="cuda:0"
        )
