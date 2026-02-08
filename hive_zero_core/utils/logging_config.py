import logging
import sys
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log levels.
    """
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Configures a hardened logger with specific formatting and handlers.

    Args:
        name (str): Name of the logger.
        log_level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        # Create console handler with colored output
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)

        # Create file handler with plain output
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / "hive_zero.log")
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    return logger
