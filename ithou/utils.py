"""
Utility functions for the I-Thou toolkit.
"""

import logging
import random
from pathlib import Path
from typing import Union
from dataclasses import dataclass

import yaml
import torch
import numpy as np


# =============================================================================
# Logging
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("ithou")


logger = setup_logging()


# =============================================================================
# Configuration
# =============================================================================

def _get_config_dir() -> Path:
    """Get the path to the configs directory."""
    module_dir = Path(__file__).parent.parent
    config_dir = module_dir / "configs"
    if config_dir.exists():
        return config_dir
    
    config_dir = Path.cwd() / "configs"
    if config_dir.exists():
        return config_dir
    
    raise FileNotFoundError("Could not find configs directory.")


def load_config(config_path: Union[str, Path]) -> dict:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_trait_config(trait_name: str) -> dict:
    """Load a trait configuration by name."""
    config_path = _get_config_dir() / "traits" / f"{trait_name}.yaml"
    return load_config(config_path)


@dataclass
class TraitConfig:
    """Structured trait configuration."""
    name: str
    description: str
    positive_instructions: list[str]
    negative_instructions: list[str]
    questions: list[str]
    trait_threshold: int = 50
    coherence_threshold: int = 50
    
    @classmethod
    def from_dict(cls, config: dict) -> "TraitConfig":
        return cls(
            name=config["name"],
            description=config["description"],
            positive_instructions=config["instructions"]["positive"],
            negative_instructions=config["instructions"]["negative"],
            questions=config["questions"],
            trait_threshold=config.get("scoring", {}).get("trait_threshold", 50),
            coherence_threshold=config.get("scoring", {}).get("coherence_threshold", 50),
        )
    
    @classmethod
    def load(cls, trait_name: str) -> "TraitConfig":
        return cls.from_dict(load_trait_config(trait_name))


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Memory Management
# =============================================================================

def clear_cuda_cache() -> None:
    """Clear CUDA cache to free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    prefix = f"{prefix} " if prefix else ""
    logger.debug(f"{prefix}GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


# =============================================================================
# File I/O
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Tensor Utilities
# =============================================================================

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = v1.float().flatten()
    v2 = v2.float().flatten()
    return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()