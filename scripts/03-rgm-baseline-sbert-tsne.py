from datetime import timedelta
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModel
from openTSNE import TSNE, affinity, initialization, TSNEEmbedding

from text_visualizations.train_stuff import fix_all_seeds
from text_visualizations.embeddings import generate_embeddings
from text_visualizations.metrics import knn_accuracy

### SETUP
variables_path = Path("../results/variables")
data_path = Path("../data")
configs_path = Path("../configs")

saving_path = variables_path / Path("sbert/iclr/baseline_tsne")
saving_path.mkdir(parents=True, exist_ok=True)

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


## obtain embeddings
start = time.time()

# fix random seeds
fix_all_seeds()

# set up model
model_name = "SBERT"
model_path = "sentence-transformers/all-mpnet-base-v2"

print("Model: ", model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device: {}".format(device))

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
print(model_path)

model = model.to(device)

# get embeddings
_, _, embedding_av = generate_embeddings(
    iclr.abstract.to_list(), tokenizer, model, device, batch_size=256
)

# save embeddings
np.save(saving_path / "embedding_abstracts_only_av", embedding_av)

end = time.time()
runtime_total = end - start
print("Total SBERT runtime: ", str(timedelta(seconds=runtime_total)))

## knn acc in high-dim
knn_acc_high_dim = knn_accuracy(
    embedding_av[iclr.labels != "unlabeled"],
    eval_train_labels,
    test_embeddings=None,
    test_labels=None,
    test_size=0.1,
    k=10,
    rs=42,
    metric="euclidean",
)
# save
np.save(
    saving_path / "knn_acc_high_dim",
    knn_acc_high_dim,
)

### obtain t-SNE
start = time.time()

## callbacks
Zs = []
kls = []
n_iter = []
callbacks_every_iters = 250


def mycallback(iteration, error, embedding):
    Zs.append(np.array(embedding.copy()))
    kls.append(error)
    n_iter.append(iteration)


A = affinity.Uniform(
    embedding_av,
    verbose=True,
    random_state=42,
    k_neighbors=10,
)
## without callbacks
# tsne = TSNE(
#     verbose=True,
#     initialization="pca",
#     random_state=42,
#     callbacks=None,
#     callbacks_every_iters=250,
# ).fit(
#     affinities=A
# )

I = initialization.pca(embedding_av, random_state=42)

E = TSNEEmbedding(I, A, n_jobs=-1, random_state=42, verbose=True)

# early exaggeration
E = E.optimize(
    n_iter=250,
    exaggeration=12,
    momentum=0.5,
    n_jobs=-1,
    verbose=True,
    callbacks=mycallback,
    callbacks_every_iters=callbacks_every_iters,
)

# final optimization without exaggeration
E = E.optimize(
    n_iter=500,
    exaggeration=1,
    momentum=0.8,
    n_jobs=-1,
    verbose=True,
    callbacks=mycallback,
    callbacks_every_iters=callbacks_every_iters,
)
tsne = np.array(E)

end = time.time()
runtime_total = end - start
print("Total t-SNE runtime: ", str(timedelta(seconds=runtime_total)))

## save
# callbacks
f = open(saving_path / "Zs.pkl", "wb")
pickle.dump(Zs, f)
f.close()
np.save(
    saving_path / "kls",
    kls,
)
np.save(
    saving_path / "n_iter",
    n_iter,
)

# embedding
np.save(
    saving_path / "tsne",
    tsne,
)

## knn acc in low-dim
knn_acc_low_dim = knn_accuracy(
    tsne[iclr.labels != "unlabeled"],
    eval_train_labels,
    test_embeddings=None,
    test_labels=None,
    test_size=0.1,
    k=10,
    rs=42,
    metric="euclidean",
)
# save
np.save(
    saving_path / "knn_acc_low_dim",
    knn_acc_low_dim,
)
