from __future__ import annotations

import sys

from loguru import logger

from src.core.config import LoggingConfig

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


def configure_logging(config: LoggingConfig) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=config.level,
        format=_CONSOLE_FORMAT,
        backtrace=False,
        diagnose=False,
        # Synchronous: a queued sink lets stderr logs land inside the stdout
        # report, which makes a mess of the terminal and of screenshots.
        enqueue=False,
    )

    if config.file:
        config.file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            config.file,
            level=config.level,
            rotation=config.rotation,
            retention=config.retention,
            serialize=config.serialize,
            encoding="utf-8",
            enqueue=True,
        )

    logger.debug("Logging configured at level {}", config.level)
