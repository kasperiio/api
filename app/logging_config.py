"""
Unified logging configuration for the electricity price API.

This module provides a logger that uses a YAML configuration file.
"""

import logging
import logging.config
import yaml


def setup_logging() -> None:
    """
    Configure logging using the hardcoded YAML configuration file.
    """
    with open("log_config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)


# Application logger
logger = logging.getLogger("app")
