
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer

from pathlib import Path
import time
from datetime import datetime
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

# Add git commit hash to config
config["log"]['git_commit'] = get_git_commit_hash() 
config["log"]['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# create saving path
exp_name = os.path.basename(exp_config_name).split('.')[0]
saving_path = (
    variables_path
    / Path(config["model"]['model_name'].lower())
    / Path(config["data_loader"]['dataset'].lower())
    / Path(exp_name + "_"+ config["timestamp"].lower())
)
saving_path.mkdir(parents=True, exist_ok=True)

# save enhanced config file to results directory
with open(os.path.join(saving_path, 'config.yaml'), 'w') as f:
    yaml.dump(config, f)


## read dataset
# TODO: implement a class? iclr and huggingface datasets are very different
# TODO: create:     
# eval_train_data=None, #iclr2024.abstract[labels_iclr != "unlabeled"].to_list(),
# eval_train_labels=None,
# SOLUTION FOR NOW:
iclr = pd.read_parquet(
    data_path / "iclr25v2.parquet",
    engine="fastparquet",
)



### EXPERIMENT
start = time.time()

## set up model
# fix random seeds
fix_all_seeds()

# set up model
print("Model: ", config["model"]['model_name'])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device: {}".format(device))

tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])

model = ModelProjector(
    checkpoint= config["model"]["model_path"],
    pooler= pooler,
    in_dim = config["model"]["in_dim"],
    hidden_dims=config["model"]["hidden_dims"],
    output_dim=config["model"]["output_dim"],
)

# wrap model
wrapped_model = ModelProjectorWrapper(model, tokenizer)


## set up dataloader
# data
training_dataset = MultOverlappingSentencesPairDataset(
    iclr.abstract,  # TODO: change this with the dataset importing class
    tokenizer,
    device,
    n_cons_sntcs = config["data_loader"]["n_cons_sntcs"],
)

gen = torch.Generator()
gen.manual_seed(42)
training_loader = torch.utils.data.DataLoader(
    training_dataset, 
    batch_size=config["data_loader"]["batch_size"],
    shuffle=True, 
    generator=gen,
)


## train model
# training # TODO: Not sure if this code works to train my model
losses, df_training_eval_results = train_loop(
    wrapped_model,
    training_loader,
    device,
    eval_train_data=None, #iclr2024.abstract[labels_iclr != "unlabeled"].to_list(),
    eval_train_labels=None, #labels_iclr[labels_iclr != "unlabeled"],
    eval_every_epochs=False,
    eval_every_batches=50,
    eval_function=eval_function, 
    pooler=pooler,
    mteb_saving_path=None, 
    mteb_tasks=None,
    n_epochs=1,
)

# Save results as parquet
# TODO: implement result logger??
np.save(saving_path / "losses", losses)
df_training_eval_results.to_parquet(
    saving_path / "df_log_training", 
    index=False,
    engine="pyarrow",
    compression="gzip",
)
# runtime
end = time.time()
runtime_total = end - start
config["log"]['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
config["log"]['runtime_total'] = str(datetime.timedelta(seconds=runtime_total))
with open(os.path.join(saving_path, 'config.yaml'), 'w') as f:
    yaml.dump(config, f)


