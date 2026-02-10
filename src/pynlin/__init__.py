from __future__ import annotations

import sys
import types
import logging


def _ensure_loguru():
    try:
        import loguru  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    base = logging.getLogger("pynlin")
    if not base.handlers:
        logging.basicConfig(level=logging.INFO)

    class _FallbackLogger:
        def __init__(self, logger: logging.Logger):
            self._logger = logger

        def remove(self, *args, **kwargs):
            return None

        def add(self, *args, **kwargs):
            return None

        def log(self, level, message, *args, **kwargs):
            self._logger.log(_coerce_level(level), message, *args, **kwargs)

        def trace(self, message, *args, **kwargs):
            self._logger.debug(message, *args, **kwargs)

        def debug(self, message, *args, **kwargs):
            self._logger.debug(message, *args, **kwargs)

        def info(self, message, *args, **kwargs):
            self._logger.info(message, *args, **kwargs)

        def success(self, message, *args, **kwargs):
            self._logger.info(message, *args, **kwargs)

        def warning(self, message, *args, **kwargs):
            self._logger.warning(message, *args, **kwargs)

        def error(self, message, *args, **kwargs):
            self._logger.error(message, *args, **kwargs)

        def critical(self, message, *args, **kwargs):
            self._logger.critical(message, *args, **kwargs)

    def _coerce_level(level):
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.INFO)
        return int(level)

    stub = types.ModuleType("loguru")
    stub.logger = _FallbackLogger(base)
    sys.modules.setdefault("loguru", stub)


_ensure_loguru()
