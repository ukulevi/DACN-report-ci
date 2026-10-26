"""Pure configuration helpers shared by local and remote training code."""

import json


def parse_config_json(config_json: str) -> dict:
    """Parse trainer configuration and require a JSON object at the top level."""
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("config_json must be a JSON object") from exc
    if not isinstance(config, dict):
        raise TypeError("config_json must be a JSON object")
    return config
