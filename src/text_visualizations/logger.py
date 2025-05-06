from collections import defaultdict
import zipfile
import numpy as np
import pandas as pd


class MyTrainingLogger:
    def __init__(self, saving_path, saving_name_df=None, saving_name_embd=None):
        self.saving_path = saving_path
        self.training_eval_results = defaultdict(list)
        self.saving_name_df = (
            saving_name_df if saving_name_df is not None else "df_log_training.h5"
        )
        self.saving_name_embd = (
            saving_name_embd if saving_name_embd is not None else "interm_embeddings_2d"
        )

    def log_metrics(self, epoch, losses, eval_results, embeddings_2d, **kwargs):
        # epoch
        self.training_eval_results["epoch"].append(epoch)

        # extra stuff, e.g., batches
        for k, v in kwargs.items():
            self.training_eval_results[k].append(v)

        # loss
        self.training_eval_results["loss"].append(losses)

        # eval results
        [
            (
                self.training_eval_results[k].append(v[0])
                if (k == "knn") | (k == "lin")
                else self.training_eval_results[k].append(v)
            )
            for k, v in eval_results.items()
        ]

        # save results
        self._save_metrics()

        # save 2D embeddings
        if embeddings_2d is not None:
            with zipfile.ZipFile(self.saving_path / self.saving_name_embd, "a") as zf:
                with zf.open(f"embeddings_2d_epoch_{epoch}.npy", "w") as f:
                    np.save(f, embeddings_2d)

    def _save_metrics(self):
        # save as HDF5
        pd.DataFrame(self.training_eval_results).to_hdf(
            self.saving_path / self.saving_name_df,
            key="df",
        )
