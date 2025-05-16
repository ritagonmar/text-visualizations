from pathlib import Path
from text_visualizations.config_helpers import parse_args

from train_hyper_sweep import run_hyperparameter_sweep

## set paths
variables_path = Path("../results/variables")
data_path = Path("../data")
configs_path = Path("../configs")

additional_saving_path = ""

## arguments
args = parse_args()
exp_config_name = args.config  # "../configs/sweep_config.yaml"

# run
results = run_hyperparameter_sweep(
    exp_config_name,
    configs_dir_path=configs_path,
    variables_path=variables_path,
    hyper_sweep_file="04-rgm-train-tsne-and-augm.py",
    additional_saving_path="training_tsne_and_augm",
)
