"""
Configuration loader and merger for experiments and production.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML file.
        
    Returns:
        Dictionary containing configuration values.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
        
    return config


def merge_configs(base_config: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, overriding base_config values with override_dict.
    
    Args:
        base_config: Base dictionary.
        override_dict: Overriding dictionary.
        
    Returns:
        Merged dictionary.
    """
    merged = base_config.copy()
    for key, value in override_dict.items():
        if value is None:
            continue
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
