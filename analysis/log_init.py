import os
import sys
from pathlib import Path

from loguru import logger


def init_logging():
    """Configure loguru with compact stdout output and fuller file logs."""
    script_name = Path(sys.argv[0]).stem or "interactive"
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    console_level = os.getenv("LOGURU_LEVEL", "DEBUG").upper()

    log_file = log_dir / f"{script_name}.log"
    log_file_deep = log_dir / f"{script_name}_deep.log"
    fmt_console = "{time:HH:mm:ss} | <level>{level:<8}</level> | {message}"
    fmt_file = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}"

    logger.remove()
    logger.add(sys.stdout, level=console_level, colorize=True, format=fmt_console)
    logger.add(log_file, level="DEBUG", rotation="10 MB", colorize=False, format=fmt_file)
    logger.add(log_file_deep, level="TRACE", colorize=False, format=fmt_file)
