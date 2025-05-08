import yaml
import itertools
import subprocess
from pathlib import Path
import time
from datetime import datetime
import copy
from text_visualizations.config_helpers import get_nested_value, load_config


def run_hyperparameter_sweep(
    exp_config_name, configs_dir_path, variables_path, hyper_sweep_file="train_model.py"
):
    """
    Run hyperparameter sweep based on a single config file that includes sweep parameters.

    Args:
        exp_config_name : e.g. "exp001.yaml"

        configs_dir_path : configs_path "../configs"

    """
    # Load sweep config
    # with open(exp_config_name, "r") as f:
    #     full_config = yaml.safe_load(f)
    full_config = load_config(exp_config_name, configs_dir_path=configs_dir_path)
    assert (
        "hyperparameter_sweep" in full_config.keys()
    ), "Config file is missing hyperparameter_sweep info."

    # Extract base config and hyperparameter ranges
    param_grid = dict()
    for key in full_config["hyperparameter_sweep"]:
        param_grid[key] = get_nested_value(full_config, key)

    # # Remove sweep info from base config
    # if "hyperparameter_sweep" in base_config:
    #     del base_config["hyperparameter_sweep"]

    # Get the config file name without extension
    exp_name = Path(exp_config_name).stem

    # Generate temp configs directory
    configs_dir = Path("../configs/temp_sweep")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Get all combinations of hyperparameters
    param_combinations = []
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]

    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        param_combinations.append(param_dict)

    print(f"Running {len(param_combinations)} experiments...")

    # Run training for each combination
    results = []
    for i, params in enumerate(param_combinations):
        # Create a copy of base config
        experiment_config = copy.deepcopy(full_config)

        # Update config with this parameter combination
        for param_path, value in params.items():
            # Handle nested parameters with dot notation (e.g., "training.lr")
            parts = param_path.split(".")
            config_section = experiment_config
            for part in parts[:-1]:
                config_section = config_section[part]
            config_section[parts[-1]] = value

        # Create descriptive name for this run
        param_str = "_".join(f"{k.split('.')[-1]}-{v}" for k, v in params.items())

        temp_config_name = f"{exp_name}__{param_str}.yaml"
        temp_config_path = configs_dir / temp_config_name

        # Save temporary config
        with open(temp_config_path, "w") as f:
            yaml.dump(experiment_config, f)

        print(f"Running experiment {i+1}/{len(param_combinations)}: {param_str}")

        # Run the training script as a subprocess
        cmd = ["python", hyper_sweep_file, "--config", str(temp_config_path)]
        process = subprocess.run(cmd, capture_output=False, text=True)

        # Store results
        results.append(
            {
                "params": params,
                "config_path": temp_config_path,
                "returncode": process.returncode,
                "success": process.returncode == 0,
            }
        )

        # Add to stdout/stderr if there was an error
        if process.returncode != 0:
            print(f"Error running experiment {i+1}:")
            print(process.stderr)

        print("--------------------------------------------------------")
        # Let's wait a bit between runs
        time.sleep(2)

    # Create summary report
    saving_path = (
        variables_path
        / Path(full_config["model"]["model_name"].lower())
        / Path(full_config["data_loader"]["dataset"].lower())
        / Path(exp_name + "_" + datetime.now().strftime("%Y%m%d"))
    )
    saving_path.mkdir(parents=True, exist_ok=True)
    summary_path = (
        saving_path / f"sweep_summary_{datetime.now().strftime('%Y%m%d')}.yaml"
    )
    with open(summary_path, "w") as f:
        yaml.dump(
            {
                "sweep_config": exp_config_name,
                "total_experiments": len(param_combinations),
                "successful_experiments": sum(r["success"] for r in results),
                "experiment_results": [
                    {"params": r["params"], "success": r["success"]} for r in results
                ],
                "hyper_sweep_file": hyper_sweep_file,
            },
            f,
        )

    print(f"Sweep completed. Summary saved to {summary_path}")
    return results


if __name__ == "__main__":
    # Example sweep config path and variables path
    # train_hyper_sweep
    from text_visualizations.config_helpers import parse_args

    ## set paths
    variables_path = Path("../results/variables")
    data_path = Path("../data")
    configs_path = Path("../configs")

    ## arguments
    args = parse_args()
    exp_config_name = args.config  # "../configs/sweep_config.yaml"

    # run
    results = run_hyperparameter_sweep(
        exp_config_name, configs_dir_path=configs_path, variables_path=variables_path
    )
    # print(results)
