import os
import sys
from pathlib import Path

from loguru import logger


def init_logging():
    # Figure out which script is being run directly
    script_name = Path(sys.argv[0]).stem or "interactive"
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Build filenames dynamically
    log_file = log_dir / f"{script_name}.log"
    log_file_deep = log_dir / f"{script_name}_deep.log"

    # Remove default handler and add ours
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    logger.add(log_file, level="DEBUG", rotation="10 MB", colorize=False)
    logger.add(log_file_deep, level="TRACE", colorize=False)

    logger.debug(f"Logger initialized for script: {script_name}")