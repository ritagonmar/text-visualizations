import zipfile
import numpy as np
import pandas as pd
import random
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

from text_visualizations.eval_functions import KNNEval, MTEBEval


def fix_all_seeds(seed=42):
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  ## this one is new
    ## Set the seed for generating random numbers on all GPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) ## this one I don't use but don't remember why

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed
    random.seed(seed)


def poolerdecorator(name):
    """This function is a decorator.
    When you use the decorator above another function, you can assign to that function an attribute called `.sent_rep` with value `name`.

    """

    def decorator(fun):
        fun.sent_rep = name
        return fun

    return decorator


# pooling functions
@poolerdecorator("av")
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size())
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


@poolerdecorator("sep")
def sep_pool(token_embeds, attention_mask):
    ix = attention_mask.sum(1) - 1
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, ix, :]


@poolerdecorator("cls")
def cls_pool(token_embeds, attention_mask):
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, 0, :]


def train_loop(
    wrapped_model,
    loader,  # training data loader
    device,
    eval_train_data,
    eval_train_labels,
    eval_test_data=None,  # None when eval is on MTEB or no train/test split
    eval_test_labels=None,
    eval_every_epochs=True,  # bool, {True, False}
    eval_every_batches=0,  # int, 0 would be like none
    eval_function=KNNEval,
    eval_rep="av",  # representation to evaluate, if None it is the same used by pooler
    dist_metric="euclidean",
    saving_path=None,
    mteb_tasks=None,
    n_epochs=1,
    lr=2e-5,
    scale=20.0,  # we multiply similarity score by this scale value, it is the inverse of the temperature
):
    """Train loop to train a pytorch sentence embedding model.


    Parameters
    ----------
    ...
    eval_train_data : list, default=None
        Data where the evaluation is run on (if knn or lin acc). In the ICLR case, it would be only the labeled papers.
        If eval_test_data=None, then eval_train_data is split into train and test set to train the classifier.
        If eval_test_data is passed, the full eval_train_data will be the train set of the classifier.

    eval_train_labels : list, default=None
        Labels corresponding to eval_train_data

    eval_test_data : list, default=None
        If you want the training and the evaluation to happen in different splits of the data, you need to pass a test set. None when eval is on MTEB or no train/test split.

    saving_path : str, default=None #TODO: update
        Path where MTEB evaluation will create a directory to save its results.
    ...

    Returns
    -------

    """

    assert not eval_function == MTEBEval or (
        saving_path is not None and mteb_tasks is not None
    ), "You forgot either the MTEB saving path or list of tasks for the MTEB evaluation."

    if (
        eval_every_batches != 0
    ):  # if this happens, eval happens after every X batch and after the last batch
        eval_every_epochs = False  # therefore, we set the evaluation after the full epoch to 0 to not evaluate twice after the last batch of the epoch

    ## training set up
    wrapped_model.model.to(device)

    # define layers to be used in multiple-negatives-ranking
    cos_sim = torch.nn.CosineSimilarity()
    loss_func = torch.nn.CrossEntropyLoss()

    # move layers to device
    cos_sim.to(device)
    loss_func.to(device)

    # initialize Adam optimizer
    optim = torch.optim.Adam(wrapped_model.model.parameters(), lr=lr)

    # setup warmup for first ~10% of steps
    total_steps = len(loader) * n_epochs
    warmup_steps = int(0.1 * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    losses = np.empty((n_epochs, len(loader)))

    # initialize eval list
    if (eval_every_epochs != 0) | (eval_every_batches != 0):
        training_eval_results = defaultdict(list)

    ## training
    for epoch in range(n_epochs):
        wrapped_model.model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for i_batch, batch in enumerate(loop):
            ## train -- finished
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and move all to the active device
            anchor_ids = batch[0][0].to(
                device
            )  # this are all anchor abstracts from the batch,len(anchor_ids)= len(batch)
            anchor_mask = batch[0][1].to(device)
            pos_ids = batch[1][0].to(
                device
            )  # this each positive pair from each anchor, all in one array, also len(batch)
            pos_mask = batch[1][1].to(device)

            # get hidden state
            a = wrapped_model.get_outputs(
                input_ids=anchor_ids, attention_mask=anchor_mask
            )
            p = wrapped_model.get_outputs(input_ids=pos_ids, attention_mask=pos_mask)

            # calculate the cosine similarities
            scores = torch.stack(
                [cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a]
            )
            # get label(s) - we could define this before if confident
            # of consistent batch sizes
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=scores.device
            )  # the labels are just the "label" of which pair it is (0 for the first pair, 1 for the second)
            # they are used in the loss to know which of the cosine similarities should be high and which low

            # and now calculate the loss
            loss = loss_func(scores * scale, labels)
            losses[epoch, i_batch] = loss.item()

            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

            ## evaluation
            if eval_every_batches != 0:
                if (i_batch % eval_every_batches == 0) | (i_batch == len(loader) - 1):
                    # add batch number to the results
                    training_eval_results["batch"].append(i_batch + len(loader) * epoch)

                    # add loss to the results
                    training_eval_results["loss"].append(losses[epoch, i_batch])

                    # path with batch number for saving MTEB results
                    mteb_saving_name = Path(f"results_epoch_{epoch}_batch_{i_batch}")

                    eval_results = eval_function(  # some of these are needed for knn eval and some others for mteb
                        wrapped_model=wrapped_model,
                        device=device,
                        dataset=eval_train_data,
                        labels=eval_train_labels,
                        test_dataset=eval_test_data,
                        test_labels=eval_test_labels,
                        eval_rep=eval_rep,
                        dist_metric=dist_metric,
                        tasks=mteb_tasks,
                        path_to_save=saving_path / mteb_saving_name,
                    )
                    [
                        training_eval_results[k].append(v)
                        for k, v in eval_results.items()
                    ]
                    wrapped_model.model.train()

                    # HOTFIX: Missing 2D embeddings and interm save

        if eval_every_epochs != 0:
            if (epoch % eval_every_epochs == 0) | (epoch == n_epochs - 1):
                print("eval_epoch", epoch)
                # ENH: implement result logger
                # add epoch number to the results
                training_eval_results["epoch"].append(epoch)

                # add epoch number to the results
                training_eval_results["loss"].append(np.mean(losses[epoch]))

                # same code as above for the batches
                mteb_saving_name = Path(f"results_epoch_{epoch}")

                eval_results = eval_function(  # some of these are needed for knn eval and some others for mteb
                    wrapped_model=wrapped_model,
                    device=device,
                    dataset=eval_train_data,
                    labels=eval_train_labels,
                    test_dataset=eval_test_data,
                    test_labels=eval_test_labels,
                    eval_rep=eval_rep,
                    dist_metric=dist_metric,
                    tasks=mteb_tasks,
                    path_to_save=saving_path / mteb_saving_name,
                )
                [
                    (
                        training_eval_results[k].append(v[0])
                        if (k == "knn") | (k == "lin")
                        else training_eval_results[k].append(v)
                    )
                    for k, v in eval_results.items()
                ]

                # add 2D embedding
                embedding_cls, embedding_sep, embedding_av = (
                    wrapped_model.encode_dataset(eval_train_data, device=device)
                )
                embedding_rep_dict = {
                    "cls": (embedding_cls,),
                    "sep": (embedding_sep,),
                    "av": (embedding_av,),
                }  # ENH: when eliminating 3 reps option, modify this

                with zipfile.ZipFile(saving_path / "interm_embeddings_2d", "a") as zf:
                    with zf.open(f"embeddings_2d_epoch_{epoch}.npy", "w") as f:
                        np.save(f, embedding_rep_dict[eval_rep][0])

                # intermediate save
                # ENH: incremental saving hdf5
                pd.DataFrame(training_eval_results).to_hdf(
                    saving_path / "df_log_training.h5",
                    key="df",
                )
