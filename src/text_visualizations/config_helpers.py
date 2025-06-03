import yaml
import argparse
from copy import deepcopy
import importlib
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config file"
    )
    return parser.parse_args()


def deep_update(base_dict, update_dict):
    """
    Recursively update a nested dictionary.

    Args:
        base_dict (dict): The base dictionary to update
        update_dict (dict): The dictionary with updates

    Returns:
        dict: Updated dictionary
    """
    result = deepcopy(base_dict)

    for key, value in update_dict.items():
        # If the value is a nested dictionary, recurse
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        elif value is not None:
            # Otherwise just update the value
            result[key] = value

    return result


def load_config(exp_config_path, configs_dir_path, base_config_path="base_config.yaml"):
    """
    Load configuration from a base config file and an experiment-specific config file.
    The experiment-specific config will override values from the base config.

    Args:
        exp_config_path (str): Name of the experiment-specific config file
        base_config_path (str): Name of the base config file with default parameters
        configs_dir_path (str): Path to the configs/ directory (should be defined inside the py file)

    Returns:
        merged_config (dict): merged configuration
    """
    exp_config_path = configs_dir_path / exp_config_path
    base_config_path = configs_dir_path / base_config_path

    # Load base configuration
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Load experiment-specific configuration
    with open(exp_config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    # Recursively merge configurations
    merged_config = deep_update(base_config, exp_config)

    return merged_config


def get_function(function_str):
    """
    Dynamically import a function from a module.

    Args:
        function_str (str): A string like 'train_stuff.mean_pool' (first python file, second funtion name, separated by a dot)
    """
    assert isinstance(
        function_str, str
    ), "Input must be a string of the format 'train_stuff.mean_pool'"

    # Handle simple string case like 'poolers.mean_pooler'
    module_path, function_name = function_str.rsplit(".", 1)
    module_path = "text_visualizations." + module_path
    module = importlib.import_module(module_path)

    return getattr(module, function_name)


def get_git_commit_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        return "Git hash not available"


def get_nested_value(data, key_path, delimiter="."):
    """
    Retrieve the value from a nested dictionary using a string of keys separated by a delimiter.

    :param data: The nested dictionary to search in.
    :param key_path: A string representing the path of keys (e.g., "parent1.child1.key").
    :param delimiter: The delimiter used to separate keys in the string (default is ".").
    :return: The value corresponding to the last key in the path, or None if any key is not found.
    """
    keys = key_path.split(delimiter)  # Split the key path into individual keys
    current_value = data

    try:
        for key in keys:
            current_value = current_value[key]  # Navigate deeper into the dictionary
        return current_value
    except (KeyError, TypeError):
        return (
            None  # Return None if any key is not found or if the structure is invalid
        )
