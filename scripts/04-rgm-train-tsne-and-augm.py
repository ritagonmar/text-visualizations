import pandas as pd
import numpy as np
from text_visualizations.embeddings import generate_embeddings
import torch
from transformers import AutoTokenizer
from openTSNE import affinity
from scipy.sparse import diags

from pathlib import Path
import time
from datetime import datetime, timedelta
import os


from text_visualizations.config_helpers import *
from text_visualizations.train_stuff import fix_all_seeds
from text_visualizations.data_stuff import (
    MultOverlappingSentencesPairDataset,
    NeighborAbstracts,
)
from text_visualizations.models import ModelProjector, ModelProjectorWrapper
from text_visualizations.train_stuff import train_loop
from text_visualizations.logger import MyTrainingLogger


### SETUP
variables_path = Path("../results/variables")
data_path = Path("../data")
configs_path = Path("../configs")

## arguments
args = parse_args()
exp_config_name = args.config

## read yaml file
config = load_config(exp_config_name, configs_dir_path=configs_path)

assert "tsne_obj" in config.keys(), "Config file is missing tsne_obj info."

# extract pooler and eval functions
pooler = get_function(config["model"]["pooler"])
eval_function = get_function(config["training"]["eval_function"])
loss_class = get_function(config["training"]["loss_class"])
loss_class_tsne = get_function(config["tsne_obj"]["loss_class"])
data_augm = get_function(config["data_loader"]["augmentation"])
print(data_augm)
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
exp_name = Path(exp_config_name).stem
saving_path = (
    variables_path
    / Path(config["model"]["model_name"].lower())
    / Path(config["data_loader"]["dataset"].lower())
    / Path("training_tsne_and_augm")
    / Path(exp_name + "_" + datetime.now().strftime("%Y%m%d"))
)
if "hyperparameter_sweep" in config.keys():
    exp_number = exp_name.split("__")[0]  #'exp007'
    param_values = exp_name.split("__")[1]  #'lr-0.001_scale-0.2'
    saving_path = (
        variables_path
        / Path(config["model"]["model_name"].lower())
        / Path(config["data_loader"]["dataset"].lower())
        / Path("training_tsne_and_augm")
        / Path(exp_number + "_" + datetime.now().strftime("%Y%m%d"))
        / Path(param_values)
    )
saving_path.mkdir(parents=True, exist_ok=True)

# save enhanced config file to results directory
with open(os.path.join(saving_path, "config.yaml"), "w") as f:
    yaml.dump(config, f)

### TRAINING
## read dataset
# ENH: implement a class? iclr and huggingface datasets are very different
# SOLUTION FOR NOW:
iclr = pd.read_parquet(
    data_path / "iclr25v2.parquet",
    engine="fastparquet",
)
# iclr = iclr[:500]
eval_train_data = iclr.abstract[iclr.labels != "unlabeled"].to_list()
eval_train_labels = iclr.labels[iclr.labels != "unlabeled"].to_list()


### EXPERIMENT
start = time.time()

# ## set up model
# # fix random seeds
# fix_all_seeds()

# # set up model
# print("Model: ", config["model"]["model_name"])

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Running on device: {}".format(device))

# tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])

# model = ModelProjector(
#     checkpoint=config["model"]["model_path"],
#     pooler=pooler,
#     in_dim=config["model"]["in_dim"],
#     hidden_dims=config["model"]["hidden_dims"],
#     output_dim=config["model"]["output_dim"],
#     freeze_backbone=config["model"]["freeze_backbone"],
# )
# print("Hidden dimensions: ", config["model"]["hidden_dims"])

# # wrap model
# wrapped_model = ModelProjectorWrapper(model, tokenizer)


# ### t-SNE -----------------------------------------------------------
# ## get high-dim representation
# # get embeddings
# model = model.to(device)
# _, _, embedding_av = generate_embeddings(
#     iclr.abstract.to_list(), tokenizer, model.backbone, device, batch_size=256
# )

# ## calculate affinities
# A = affinity.Uniform(
#     embedding_av,
#     verbose=True,
#     random_state=42,
#     k_neighbors=config["tsne_obj"]["k"],
#     symmetrize=config["tsne_obj"]["symmetrize"],
# )

# # add ones to diag FIXME: eliminate adding ones
# ones_diag = diags([1], [0], shape=(A.P.shape[0], A.P.shape[1]), format="csr")
# affinities_mat = A.P + ones_diag

# ## set up dataloader
# # data
# training_dataset = NeighborAbstracts(
#     iclr.abstract.to_list(),
#     tokenizer,
#     device,
#     affinities_mat,
# )

# gen = torch.Generator()
# gen.manual_seed(42)
# training_loader = torch.utils.data.DataLoader(
#     training_dataset,
#     batch_size=config["tsne_obj"]["batch_size"],
#     shuffle=True,
#     generator=gen,
# )
# print(f"Training with t-SNE objective for {config["tsne_obj"]["n_epochs"]} epochs")

# # logger
# logger = MyTrainingLogger(
#     saving_path=saving_path,
#     saving_name_df="df_log_training_tsne.h5",
#     saving_name_embd="interm_embeddings_2d_tsne",
# )

# ## train model
# train_loop(
#     wrapped_model,
#     training_loader,
#     device,
#     eval_train_data=eval_train_data,
#     eval_train_labels=eval_train_labels,
#     eval_every_epochs=config["training"]["eval_every_epochs"],
#     eval_every_batches=config["training"]["eval_every_batches"],
#     eval_function=eval_function,
#     eval_rep=eval_rep,
#     dist_metric=config["training"]["dist_metric"],
#     mteb_saving_path=saving_path,
#     mteb_tasks=config["training"]["mteb_tasks"],
#     n_epochs=config["tsne_obj"]["n_epochs"],
#     lr=config["tsne_obj"]["lr"],
#     scale=config["tsne_obj"]["scale"],
#     save_interm_embeds=config["training"]["save_interm_embeds"],
#     logger=logger,
#     loss_class=loss_class_tsne,
# )

# # save model checkpoint
# wrapped_model.model.save_model(
#     saving_path / "trained_model_after_tsne.pt", include_pooler=True
# )

# # save 2D embeddings
# # This is needed because the intermediate 2D embeddings are not of the full data,
# # but of the eval_training_data (which in the ICLR case is only labeled points).
# # This is not ideal but the only way possible now because train_loop does not take the
# # data in its raw format but as the loader, and only takes it in its "raw format"
# # for evaluation, and in the ICLR case this is only done with the labeled subset.
# _, _, embeddings_2d = wrapped_model.encode_dataset(
#     iclr.abstract.to_list(), device=device
# )
# np.save(saving_path / "embeddings_2d_after_tsne", embeddings_2d)

# # runtime
# end = time.time()
# runtime_total = end - start
# config["log"]["end_time_tsne"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# config["log"]["runtime_after_tsne"] = str(timedelta(seconds=runtime_total))
# print("Runtime after t-SNE: ", str(timedelta(seconds=runtime_total)))
# with open(os.path.join(saving_path, "config.yaml"), "w") as f:
#     yaml.dump(config, f)


### CRASHED EXPERIMENT -- DELETE AFTER
# fix random seeds
fix_all_seeds()

## set up model
print("Model: ", config["model"]["model_name"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device: {}".format(device))

tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_path"])

## load trained model
tsne_model_path = Path(config["tsne_obj"]["model_path"])

try:
    # Try to load with saved pooler
    loaded_model = ModelProjector.load_model(
        filepath=variables_path / tsne_model_path / "trained_model_after_tsne.pt",
        device=device,
    )
except ValueError:
    # If pooler couldn't be loaded, provide it explicitly
    # from text_visualizations.train_stuff import mean_pool

    loaded_model = ModelProjector.load_model(
        filepath=variables_path / tsne_model_path / "trained_model_after_tsne.pt",
        pooler=pooler,
        device=device,
    )

# wrap model
wrapped_model = ModelProjectorWrapper(loaded_model, tokenizer)

### AUGMENTATIONS -------------------------------------------------------
## set up dataloader
# data
training_dataset = data_augm(
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

# print(f"Training with augmentations for {config["training"]["n_epochs"]} epochs")

# logger
logger = MyTrainingLogger(saving_path=saving_path)

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
    mteb_saving_path=saving_path,
    mteb_tasks=config["training"]["mteb_tasks"],
    n_epochs=config["training"]["n_epochs"],
    lr=config["training"]["lr"],
    scale=config["training"]["scale"],
    save_interm_embeds=config["training"]["save_interm_embeds"],
    logger=logger,
    loss_class=loss_class,
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
