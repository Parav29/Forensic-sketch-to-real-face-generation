"""Project-wide logging setup — replaces scattered ``print`` calls."""
import logging
import sys
from pathlib import Path

_CONFIGURED = set()


def get_logger(name: str = "sketch2photo", logfile: str = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger. Idempotent per (name, logfile) so repeated calls
    do not stack duplicate handlers.
    """
    logger = logging.getLogger(name)
    key = (name, logfile)
    if key in _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED.add(key)
    return logger
