import os
import sys
from pathlib import Path

from loguru import logger


def init_logging():
    """Configure loguru with compact stdout output and fuller file logs."""
    script = Path(sys.argv[0]).stem or "interactive"
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    console_level = os.getenv("LOGURU_LEVEL", "INFO").upper()
    fmt_console = "{time:HH:mm:ss} | <level>{level:<8}</level> | {message}"
    fmt_file    = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}"

    logger.remove()

    logger.add(
        sys.stdout,
        level=console_level,
        colorize=True,
        format=fmt_console,
    )

    logger.add(
        log_dir / f"{script}.log",
        level="DEBUG",
        rotation="10 MB",
        format=fmt_file,
    )

    logger.add(
        log_dir / f"{script}_deep.log",
        level="TRACE",
        format=fmt_file,
    )
