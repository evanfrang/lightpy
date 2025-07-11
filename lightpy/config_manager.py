import json
import os

def load_config(config_name):
    """
    Loads a JSON configuration file from the 'config' directory.

    Args:
        config_name (str): The name of the configuration file (e.g., "single_slit_basic.json").

    Returns:
        dict: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the specified config file is not found.
    """
    # Get the directory of the current script (config_manager.py or __init__.py if moved)
    # This ensures it works regardless of where main.py is run from.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(current_dir, '..', 'config') # Go up one level, then into config
    
    config_path = os.path.join(config_dir, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config