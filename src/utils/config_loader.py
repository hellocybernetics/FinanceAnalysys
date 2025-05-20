"""
Configuration Loader Utility
"""
import os
import yaml
import json
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file format is unsupported or if there's an error parsing the file.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()
    config = {}

    try:
        if file_ext == '.yaml' or file_ext == '.yml':
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                if config is None: # Handle empty YAML file case
                    config = {}
                logger.info(f"Loaded YAML configuration from: {config_path}")
        elif file_ext == '.json':
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
                logger.info(f"Loaded JSON configuration from: {config_path}")
        else:
            logger.error(f"Unsupported configuration file format: {file_ext}")
            raise ValueError(f"Unsupported configuration file format: {file_ext}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise ValueError(f"Error parsing YAML file {config_path}: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {config_path}: {e}")
        raise ValueError(f"Error parsing JSON file {config_path}: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise ValueError(f"An unexpected error occurred while loading config {config_path}") from e
        
    if not isinstance(config, dict):
        logger.error(f"Configuration file {config_path} did not load as a dictionary.")
        raise ValueError(f"Configuration file {config_path} must contain a dictionary.")

    return config 