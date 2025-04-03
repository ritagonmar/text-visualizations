import yaml
import argparse
import os
from copy import deepcopy
import importlib
from datetime import datetime
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
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
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = deep_update(result[key], value)
        elif value is not None:
            # Otherwise just update the value
            result[key] = value
            
    return result


def load_config(exp_config_path, configs_dir_path, base_config_path='base_config.yaml'):
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
    exp_config_path  = configs_dir_path / exp_config_path
    base_config_path  = configs_dir_path / base_config_path

    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load experiment-specific configuration
    with open(exp_config_path, 'r') as f:
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
    assert isinstance(function_str, str), "Input must be a string of the format 'train_stuff.mean_pool'"
    
    # Handle simple string case like 'poolers.mean_pooler'
    module_path, function_name = function_str.rsplit('.', 1)
    module_path = "text_visualizations." + module_path
    module = importlib.import_module(module_path)

    return getattr(module, function_name)


def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Git hash not available"


# --------------------------------------------


# def create_path(exp_config_path, model_name, variables_path, important_params):

#     # Extract experiment name from config file path
#     exp_name = os.path.basename(config_path).split('.')[0]

    
#     # Create results directory if it doesn't exist
#     results_dir = os.path.join('results', exp_name)
#     os.makedirs(results_dir, exist_ok=True)
    
#     # Add metadata
#     merged_config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     merged_config['base_config'] = base_config_path
#     merged_config['experiment_config'] = config_path
    
#     # Save merged configuration to results directory for reproducibility
#     with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
#         yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
    
#     return merged_config, results_dir



# # Example usage
# if __name__ == '__main__':
#     config, results_dir = load_config('configs/exp001_lr0.001.yaml')
#     print(f"Loaded config for experiment: {results_dir}")
#     print(f"Learning rate: {config['training']['learning_rate']}")