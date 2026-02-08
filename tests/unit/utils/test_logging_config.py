import logging
import os

from hive_zero_core.utils.logging_config import setup_logger


def test_setup_logger_creates_handlers(reset_logging, tmp_path):
    """
    Test that setup_logger creates both stream and file handlers.
    We patch the Path in the module or change CWD to tmp_path to test file creation safely.
    """
    # Change CWD to tmp_path so 'logs' directory is created there
    os.chdir(tmp_path)

    logger = setup_logger("test_logger", log_level=logging.DEBUG)

    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1

    # Check for StreamHandler
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) > 0

    # Check for FileHandler
    # The implementation creates a file handler in "logs/hive_zero.log"
    # Wait, the implementation hardcodes Path("logs").
    # Since we changed CWD, it should be in tmp_path/logs
    log_dir = tmp_path / "logs"
    assert log_dir.exists()
    assert (log_dir / "hive_zero.log").exists()

def test_logger_formatting(reset_logging, caplog):
    """
    Test that the logger formats messages correctly.
    """
    logger = setup_logger("format_test", log_level=logging.INFO)

    with caplog.at_level(logging.INFO, logger="format_test"):
        logger.info("Test message")

    assert "Test message" in caplog.text
    # The formatter includes asctime, name, levelname.
    # caplog.text might not capture the full formatting unless we configure caplog handler,
    # but we can check the record attributes.
    record = caplog.records[0]
    assert record.levelname == "INFO"
    assert record.name == "format_test"
    assert record.msg == "Test message"

def test_setup_logger_idempotency(reset_logging, tmp_path):
    """
    Test that calling setup_logger multiple times does not duplicate handlers.
    """
    os.chdir(tmp_path)
    logger1 = setup_logger("idempotent_test")
    initial_handler_count = len(logger1.handlers)

    logger2 = setup_logger("idempotent_test")
    assert len(logger2.handlers) == initial_handler_count
    assert logger1 is logger2
