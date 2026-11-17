"""
Logging setup utility for training, API, and evaluation.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ws_vad",
    log_dir: Optional[str | Path] = None,
    log_level: int = logging.INFO,
    log_to_console: bool = True
) -> logging.Logger:
    """Configure a unified logger with both console and file handlers.
    
    Args:
        name: Name of the logger.
        log_dir: Directory where log files should be saved.
        log_level: Logging level (e.g. logging.INFO).
        log_to_console: Whether to stream logs to stdout.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_path / f"{name}_{timestamp}.log", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
