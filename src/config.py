"""Configuration helpers for the training script."""

from pathlib import Path
from typing import Any, Dict

import torch
import yaml

REQUIRED_SECTIONS = ("data", "training", "runtime", "output")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load the YAML config file and validate the required sections."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("The YAML config must contain a top-level mapping.")

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Missing config section(s): {missing}")

    return config


def resolve_device(device_name: str) -> str:
    """Resolve the configured device name to cpu or cuda."""
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be one of: auto, cpu, cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("runtime.device is set to cuda, but CUDA is not available.")
    return device_name
