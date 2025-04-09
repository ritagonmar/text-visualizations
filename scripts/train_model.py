import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer

from pathlib import Path
import time
from datetime import datetime, timedelta
import os

from text_visualizations.config_helpers import *
from text_visualizations.train_stuff import fix_all_seeds
from text_visualizations.data_stuff import MultOverlappingSentencesPairDataset
from text_visualizations.models import ModelProjector, ModelProjectorWrapper
from text_visualizations.train_stuff import train_loop


### SETUP
variables_path = Path("../results/variables")
data_path = Path("../data")
configs_path = Path("../configs")

## arguments
args = parse_args()
exp_config_name = args.config

## read yaml file
config = load_config(exp_config_name, configs_dir_path=configs_path)

# extract pooler and eval functions
pooler = get_function(config["model"]["pooler"])
eval_function = get_function(config["training"]["eval_function"])

# get eval_rep
eval_rep = (
    pooler.sent_rep
    if config["training"]["eval_rep"] is None
    else config["training"]["eval_rep"]
)

# Add git commit hash to config
config["log"] = dict()
config["log"]["git_commit"] = get_git_commit_hash()
config["log"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# create saving path
exp_name = os.path.basename(exp_config_name).split(".")[0]
saving_path = (
    variables_path
    / Path(config["model"]["model_name"].lower())
    / Path(config["data_loader"]["dataset"].lower())
    / Path(exp_name + "_" + datetime.now().strftime("%Y%m%d"))
)
saving_path.mkdir(parents=True, exist_ok=True)

# save enhanced config file to results directory
with open(os.path.join(saving_path, "config.yaml"), "w") as f:
    yaml.dump(config, f)


## read dataset
# ENH: implement a class? iclr and huggingface datasets are very different
# SOLUTION FOR NOW:
iclr = pd.read_parquet(
    data_path / "iclr25v2.parquet",
    engine="fastparquet",
)
eval_train_data = iclr.abstract[iclr.labels != "unlabeled"].to_list()
eval_train_labels = iclr.labels[iclr.labels != "unlabeled"].to_list()


### EXPERIMENT
start = time.time()

## set up model
# fix random seeds
fix_all_seeds()

# set up model
print("Model: ", config["model"]["model_name"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device: {}".format(device))

tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])

model = ModelProjector(
    checkpoint=config["model"]["model_path"],
    pooler=pooler,
    in_dim=config["model"]["in_dim"],
    hidden_dims=config["model"]["hidden_dims"],
    output_dim=config["model"]["output_dim"],
    freeze_backbone=config["model"]["freeze_backbone"],
)
print("Hidden dimensions: ", config["model"]["hidden_dims"])

# wrap model
wrapped_model = ModelProjectorWrapper(model, tokenizer)


## set up dataloader
# data
training_dataset = MultOverlappingSentencesPairDataset(
    iclr.abstract,  # ENH: change this with the dataset importing class
    tokenizer,
    device,
    n_cons_sntcs=config["data_loader"]["n_cons_sntcs"],
)

gen = torch.Generator()
gen.manual_seed(42)
training_loader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=config["data_loader"]["batch_size"],
    shuffle=True,
    generator=gen,
)
print(f"Training for {config["training"]["n_epochs"]} epochs")
## train model
train_loop(
    wrapped_model,
    training_loader,
    device,
    eval_train_data=eval_train_data,
    eval_train_labels=eval_train_labels,
    eval_every_epochs=config["training"]["eval_every_epochs"],
    eval_every_batches=config["training"]["eval_every_batches"],
    eval_function=eval_function,
    eval_rep=eval_rep,
    dist_metric=config["training"]["dist_metric"],
    saving_path=saving_path,
    mteb_tasks=config["training"]["mteb_tasks"],
    n_epochs=config["training"]["n_epochs"],
    lr=config["training"]["lr"],
    scale=config["training"]["scale"],
)
# save model checkpoint
wrapped_model.model.save_model(saving_path / "trained_model.pt", include_pooler=True)

# save 2D embeddings
# This is needed because the intermediate 2D embeddings are not of the full data,
# but of the eval_training_data (which in the ICLR case is only labeled points).
# This is not ideal but the only way possible now because train_loop does not take the
# data in its raw format but as the loader, and only takes it in its "raw format"
# for evaluation, and in the ICLR case this is only done with the labeled subset.
_, _, embeddings_2d = wrapped_model.encode_dataset(
    iclr.abstract.to_list(), device=device
)
np.save(saving_path / "embeddings_2d", embeddings_2d)

# runtime
end = time.time()
runtime_total = end - start
config["log"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
config["log"]["runtime_total"] = str(timedelta(seconds=runtime_total))
print("Total runtime: ", str(timedelta(seconds=runtime_total)))
with open(os.path.join(saving_path, "config.yaml"), "w") as f:
    yaml.dump(config, f)
