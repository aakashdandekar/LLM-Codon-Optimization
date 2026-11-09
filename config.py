import os
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file, falling back to CODON_CONFIG or default config.yaml."""
    path = config_path or os.environ.get("CODON_CONFIG", DEFAULT_CONFIG_PATH)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}
