from unittest.mock import MagicMock, call

import pytest

MODULE_PATH = "src.logger.logger"
from src.logger.logger import _Logger, StreamToLogger


class TestStreamToLogger:
    """Tests the class that redirects stderr to loguru."""

    def test_write_splits_lines(self):
        """Verify write splits buffer into separate log entries."""
        mock_logger_instance = MagicMock()
        stream = StreamToLogger(mock_logger_instance, level="ERROR")

        buffer = "Line 1 error\nLine 2 error\n"
        stream.write(buffer)

        opt_mock = mock_logger_instance.opt.return_value
        assert opt_mock.log.call_count == 2

        expected_calls = [
            call("ERROR", "Line 1 error"),
            call("ERROR", "Line 2 error"),
        ]
        opt_mock.log.assert_has_calls(expected_calls)

    def test_write_ignores_empty_lines(self):
        """Verify empty lines/whitespace are skipped."""
        mock_logger_instance = MagicMock()
        stream = StreamToLogger(mock_logger_instance)

        stream.write("\n   \n")
        mock_logger_instance.opt.return_value.log.assert_not_called()

    def test_flush_exists(self):
        """Verify flush method exists."""
        stream = StreamToLogger(MagicMock())
        stream.flush()


class TestLoggerWrapper:

    @pytest.fixture
    def mock_loguru(self, mocker):
        """Mocks the internal loguru logger."""
        return mocker.patch(f"{MODULE_PATH}.loguru_logger")

    @pytest.fixture
    def mock_sys(self, mocker):
        """Mocks sys to prevent actual stderr redirection during tests."""
        m_sys = mocker.patch(f"{MODULE_PATH}.sys")

        m_sys.__stderr__ = MagicMock()
        m_sys.stderr = MagicMock()
        m_sys.stdout = MagicMock()

        return m_sys

    def test_init_defaults(self, mock_loguru, mock_sys):
        """Test default initialization (Console=True, Stderr=False)."""
        mock_sys.stdout.isatty.return_value = True

        logger = _Logger(console=True, capture_stderr=False)

        mock_loguru.remove.assert_called_once()
        mock_loguru.add.assert_called_once()
        assert mock_sys.stderr != logger

    def test_init_capture_stderr(self, mock_loguru, mock_sys):
        """Test stderr redirection logic."""
        mock_sys.stdout.isatty.return_value = False

        _Logger(console=False, capture_stderr=True)

        assert isinstance(mock_sys.stderr, StreamToLogger)

    def test_add_file_sink(self, mock_loguru, mock_sys):
        """Verify file sink is added only once."""
        mock_sys.stdout.isatty.return_value = False
        logger = _Logger(console=False)

        logger.add_file_sink("test.log")
        assert mock_loguru.add.call_count == 1
        args, kwargs = mock_loguru.add.call_args
        assert args[0] == "test.log"
        assert kwargs["level"] == "INFO"

        logger.add_file_sink("test2.log")
        assert mock_loguru.add.call_count == 1

    def test_add_tui_sink(self, mock_loguru, mock_sys):
        """Verify TUI sink replaces the console sink."""
        mock_sys.stdout.isatty.return_value = True
        mock_loguru.add.return_value = 1

        logger = _Logger(console=True)
        assert logger._console_sink_id == 1

        dummy_sink = lambda x: None
        mock_loguru.add.return_value = 2

        logger.add_tui_sink(dummy_sink)

        mock_loguru.remove.assert_called_with(1)
        assert logger._console_sink_id == 2
        args, _ = mock_loguru.add.call_args
        assert args[0] == dummy_sink

    def test_log_methods_standard(self, mock_loguru, mock_sys):
        """Test info, warning, error without file_only flag."""
        logger = _Logger(console=False)

        logger.info("info msg")
        mock_loguru.info.assert_called_with("info msg")

        logger.error("err msg")
        mock_loguru.error.assert_called_with("err msg")

    def test_log_methods_file_only(self, mock_loguru, mock_sys):
        """Test log methods with file_only=True."""
        logger = _Logger(console=False)

        bound_logger = MagicMock()
        mock_loguru.bind.return_value = bound_logger

        logger.info("secret msg", file_only=True)

        mock_loguru.bind.assert_called_with(file_only=True)
        bound_logger.info.assert_called_with("secret msg")

    def test_console_filter_logic(self, mock_loguru, mock_sys):
        """Test the filter lambda used in console sink."""
        mock_sys.stdout.isatty.return_value = True

        _Logger(console=True)

        filter_fn = mock_loguru.add.call_args[1]["filter"]

        assert filter_fn({"extra": {}}) is True

        assert filter_fn({"extra": {"file_only": True}}) is False
