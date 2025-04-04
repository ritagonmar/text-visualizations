from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import os
import datasets
import numpy as np
from transformers import AutoModel, AutoTokenizer

# from adapters import AutoAdapterModel
from sentence_transformers import SentenceTransformer, models
from tqdm.notebook import tqdm

from text_visualizations.embeddings import (
    generate_embeddings,
    generate_embeddings_embed_layer,
)

# TODO: docstrings and comments
# ENH: make encode_dataset only output the rep of the chosen pooler and not all 3
#       then adapt the code in eval_functions to match
# ENH: delete all models used only in the text-embedding project


class ModelWrapper(ABC):
    # Inits should look like this
    def __init__(self, model, tokenizer):
        self.checkpoint = model.config.name_or_path
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def encode_dataset(self, data, device):
        """For kNN and linear evaluation"""
        # return embedding_cls, embedding_sep, embedding_av
        pass

    @abstractmethod
    def ST_wrapper(self):
        """For MTEB evaluation"""
        # to transform it to a SentenceTransformer for MTEB eval
        # return ST_model
        pass

    @abstractmethod
    def get_outputs(self, input_ids, attention_mask):
        """For the train_loop.
        To get the latent representation as a single output. For using in train_loop.
        output : all token embeddings
        """
        # output = self.model(input_ids, attention_mask=attention_mask)[0]
        ...
        # return output
        pass


class HFModelWrapper(ModelWrapper):
    """ONLY FOR PRE-TRAINED MODELS"""

    def __init__(self, model, tokenizer):
        """"""
        self.checkpoint = model.config.name_or_path
        self.model = model
        self.tokenizer = tokenizer

    def encode_dataset(self, data, device):
        """For knn and linear evaluation"""
        # TODO: potentially make it return only the representation that is being optimized (?)
        (
            embedding_cls,
            embedding_sep,
            embedding_av,
        ) = generate_embeddings(
            data,
            self.tokenizer,
            self.model,
            device,
            batch_size=256,
            return_seventh=False,
        )
        return embedding_cls, embedding_sep, embedding_av

    def get_outputs(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask)[0]
        return output

    def ST_wrapper(self):
        ST_model = SentenceTransformer(self.checkpoint)
        return ST_model


class FineTunedHFModelWrapper(HFModelWrapper):
    """FOR HF MODELS THAT ARE ALREADY/GOING TO BE FINE-TUNED"""

    def __init__(self, model, tokenizer, checkpoint=None):
        if checkpoint is None:
            checkpoint = "bert-base-uncased"
        self.base_checkpoint = checkpoint
        self.model = model
        self.tokenizer = tokenizer

    def ST_wrapper(self):
        # Create a new SentenceTransformer model
        new_modules = []

        # Wrap your custom base model in a Transformer module
        transformer_model = models.Transformer(
            model_name_or_path=self.base_checkpoint,  # None gives an error,
            # so I initialize with the pre-trained model that
            # will be substituted by my fine-tuned model below
            max_seq_length=384,  # You can adjust this as needed
            do_lower_case=False,  # Adjust based on your tokenizer
        )
        # Replace the auto_model in the Transformer wrapper with your custom model
        transformer_model.auto_model = self.model
        # Set the tokenizer
        transformer_model.tokenizer = self.tokenizer

        # Add the wrapped model as the first module
        new_modules.append(transformer_model)

        # Add Pooling layer
        pooling_model = models.Pooling(
            word_embedding_dimension=self.model.config.hidden_size,
            pooling_mode_cls_token=False,
            pooling_mode_mean_tokens=True,  # TODO: CHANGE POOLING JENACHDEM
            pooling_mode_max_tokens=False,
            pooling_mode_mean_sqrt_len_tokens=False,
        )
        new_modules.append(pooling_model)

        # Add Normalize layer
        new_modules.append(models.Normalize())

        # Create the new SentenceTransformer model
        ST_model = SentenceTransformer(modules=new_modules)
        return ST_model


class ModelWithProjectionHead(nn.Module):
    def __init__(self, checkpoint, pooler, in_dim=768, feat_dim=128, hidden_dim=512):
        super().__init__()  # to inherit from the nn.Module
        self.pooler = pooler  # pooler function
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # load model
        # if checkpoint == "allenai/specter2_base":
        #     self.backbone = AutoAdapterModel.from_pretrained(checkpoint)
        #     # add adapter proximity
        #     self.backbone.load_adapter(
        #         "allenai/specter2",
        #         source="hf",
        #         load_as="specter2",
        #         set_active=True,
        #     )
        # else:
        self.backbone = AutoModel.from_pretrained(checkpoint)

        # add projection head
        self.projection_head = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feat_dim),
        )

    def forward(self, input_ids, attention_mask):
        """
        pooler : {mean_pool, cls_pool, sep_pool, seventh_pool?}
        """
        # Extract outputs from the body
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]

        # pooling
        h = self.pooler(outputs, attention_mask)

        # Add custom layers
        z = self.projection_head(h)  # .view(-1,768)

        return z


class ModelWithProjectionHeadWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        """
        model : full model with the projection head
        self.model : full model with the projection head
        self.backbone : model without the projection head
        """
        self.checkpoint = model.backbone.config.name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.backbone = model.backbone

    def get_outputs(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def encode_dataset(self, data, device):
        """For knn and linear evaluation.
        It gives back the representation after the backbone, i.e., before the projection head.
        """
        (
            embedding_cls,
            embedding_sep,
            embedding_av,
        ) = generate_embeddings(
            data,
            self.tokenizer,
            self.backbone,
            device,
            batch_size=256,
            return_seventh=False,
        )
        return embedding_cls, embedding_sep, embedding_av

    def ST_wrapper(self):
        # this function was not needed so far therefore not implemented
        pass


class EmbeddingOnlyModel(torch.nn.Module):
    """Create a new model with only the embedding layer.
    Valid function for both the layer and the module (04/09/2024)

    Parameters
    ----------
    """

    def __init__(self, model_name_or_embeddings):
        super().__init__()
        if isinstance(model_name_or_embeddings, str):
            # If a string is provided, load the pretrained model and extract embeddings
            pretrained_model = AutoModel.from_pretrained(model_name_or_embeddings)
            self.embeddings = pretrained_model.embeddings.word_embeddings
        else:
            # If embeddings are provided directly, use them
            self.embeddings = model_name_or_embeddings

    def forward(self, input_ids):
        return self.embeddings(input_ids)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))

    @classmethod
    def from_pretrained(cls, load_directory, base_model="bert-base-uncased"):
        # Q?: is it a problem that is bert-base-uncased when using a pre-trained embedding layer from MPNet?
        # Load the state dict
        state_dict = torch.load(os.path.join(load_directory, "model.pt"))

        # Create a new instance of the model with a dummy model name
        # We'll replace the embeddings with the loaded state dict
        model = cls(base_model)

        # Load the state dict
        model.load_state_dict(state_dict)

        return model


# Q?: is it possible to make this the ST_wrapper function of class above? I think not.
class MyEmbeddingSentenceModel:
    """Sentence embedding model using only the embedding layer of a transformer.
    Uses class EmbeddingOnlyModel (see above) and puts it in the format of Sentence Transformers, to be able to evaluate it in the MTEB tasks.
    """

    def __init__(self, model, tokenizer, pooler):
        self.tokenizer = tokenizer
        self.model = model
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.pooler = pooler

        self.model.to(self.device)

    @torch.no_grad()  # Q?: what is the difference between no_grad() and inference_mode()?
    def encode(self, input_texts, batch_size=None, **kwargs):
        inputs = self.tokenizer(
            input_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        dataset = datasets.Dataset.from_dict(inputs)
        dataset.set_format(type="torch", output_all_columns=True)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=10
        )
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) for k, v in batch.items()
                }  # it used to be device only (not self.device) but I think it only wroked bc of jupyter having device defined somewhere else
                outputs = self.model(batch["input_ids"])
                embdd = self.pooler(outputs, batch["attention_mask"])
                embeddings.append(embdd.detach().cpu().numpy())

        embeddings = np.vstack(embeddings)
        return embeddings


class EmbeddingOnlyModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer, pooler):
        "model : instance of EmbeddingOnlyModel"
        self.model = model
        self.tokenizer = tokenizer
        self.pooler = pooler

    def get_outputs(self, input_ids, **kwargs):
        """
        **kwargs : attention_mask is also always passed to get_outputs but in this case it is not used (because it is only the embedding layer)
        """
        outputs = self.model(input_ids)
        return outputs

    def encode_dataset(self, data, device):
        """For knn and linear evaluation"""
        (
            embedding_cls,
            embedding_sep,
            embedding_av,
        ) = generate_embeddings_embed_layer(
            data,
            self.tokenizer,
            self.model,
            device,
            batch_size=256,
            return_seventh=False,
        )
        return embedding_cls, embedding_sep, embedding_av

    def ST_wrapper(self):
        ST_model = MyEmbeddingSentenceModel(
            model=self.model, tokenizer=self.tokenizer, pooler=self.pooler
        )
        return ST_model


# extra function
def check_models_equal(original_model, loaded_model):
    """Checks if models have identical parameters.
    The == operator does not work because two separately instantiated model objects will always be considered different, even if they have identical parameters.

    """
    # Check if state dictionaries are equal
    original_state_dict = original_model.state_dict()
    loaded_state_dict = loaded_model.state_dict()

    # This checks if all keys and tensor values are the same
    are_equal = all(
        torch.equal(original_state_dict[key], loaded_state_dict[key])
        for key in original_state_dict
    )

    print(f"Models have identical parameters: {are_equal}")

    # # You can also check individual layers if needed
    # print(torch.equal(original_model.embeddings.word_embeddings.weight,
    #                   loaded_model.embeddings.word_embeddings.weight))


class ModelProjector(nn.Module):
    """Model that projects to 2D
    Very similar to ModelWithProjectionHead, only changing the projection head.
    """

    def __init__(
        self,
        checkpoint,
        pooler,
        hidden_dims,
        in_dim=768,
        output_dim=2,
        freeze_backbone=True,
    ):
        super().__init__()  # to inherit from the nn.Module
        self.pooler = pooler  # pooler function
        self.checkpoint = checkpoint
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.hidden_dims = (
            [hidden_dims] if not isinstance(hidden_dims, list) else hidden_dims
        )
        self.freeze_backbone = freeze_backbone

        # load model
        self.backbone = AutoModel.from_pretrained(checkpoint)

        # freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # create projection head
        layers = []
        ## input layer
        layers.append(nn.Linear(self.in_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())

        ## hidden layers
        for i in range(1, len(self.hidden_dims)):
            layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            layers.append(nn.ReLU(inplace=True))

        ## output layer
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        # add projection head
        self.projection_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        """
        pooler : {mean_pool, cls_pool, sep_pool, seventh_pool?}
        """
        # Extract outputs from the body
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]

        # pooling
        h = self.pooler(outputs, attention_mask)

        # Add custom layers
        z = self.projection_head(h)  # .view(-1,768)

        return z

    def save_model(self, filepath, include_pooler=False):
        """
        Save the model's state dictionary and configuration to a file

        Args:
            filepath: Path where to save the model
            include_pooler: Whether to try saving the pooler function (only works for simple functions)
        """
        # Save model configuration along with state dict
        save_dict = {
            "state_dict": self.state_dict(),
            "config": {
                "checkpoint": self.checkpoint,
                "in_dim": self.in_dim,
                "hidden_dims": self.hidden_dims,
                "output_dim": self.output_dim,
                "freeze_backbone": any(
                    not p.requires_grad for p in self.backbone.parameters()
                ),
            },
        }

        # Try to save pooler if requested (may not work for all pooler functions)
        if include_pooler:
            try:
                import dill

                save_dict["pooler"] = dill.dumps(self.pooler)
                print("Pooler function serialized successfully")
            except Exception as e:
                print(f"Warning: Could not serialize pooler function: {e}")
                print("You'll need to provide the pooler function when loading")

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, pooler=None, device=None):
        """
        Load a model from a saved file with configuration

        Args:
            filepath: Path to the saved model file
            pooler: Pooling function to use (required if not saved or can't be deserialized)
            device: Device to load the model to (default: None)

        Returns:
            model: Loaded model instance
        """
        # Load the saved dictionary
        if device is None:
            save_dict = torch.load(filepath)
        else:
            save_dict = torch.load(filepath, map_location=device)

        # Extract configuration
        config = save_dict["config"]

        # Try to load saved pooler if present
        if "pooler" in save_dict and pooler is None:
            try:
                import dill

                pooler = dill.loads(save_dict["pooler"])
                print("Pooler function loaded successfully")
            except Exception as e:
                print(f"Warning: Could not deserialize saved pooler function: {e}")
                if pooler is None:
                    raise ValueError(
                        "Pooler function is required but couldn't be loaded from file. Please provide one."
                    )

        if pooler is None:
            raise ValueError(
                "Pooler function is required but not provided and not found in saved file"
            )

        # Create a new model with the saved configuration
        model = cls(
            checkpoint=config["checkpoint"],
            pooler=pooler,
            hidden_dims=config["hidden_dims"],
            in_dim=config["in_dim"],
            output_dim=config["output_dim"],
            freeze_backbone=config.get("freeze_backbone", False),
        )

        # Load the state dictionary
        model.load_state_dict(save_dict["state_dict"])

        # Move to specified device if provided
        if device is not None:
            model = model.to(device)

        print(f"Model loaded from {filepath}")
        return model


class ModelProjectorWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        """
        model : full model with the projection head
        self.model : full model with the projection head
        self.backbone : model without the projection head
        """
        self.checkpoint = model.backbone.config.name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.backbone = model.backbone
        self.projection_head = model.projection_head

    def get_outputs(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def encode_dataset(self, data, device):
        """
        data : list of str!!!!!
        """
        inputs = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        dataset = datasets.Dataset.from_dict(inputs)
        dataset.set_format(type="torch", output_all_columns=True)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=256,  # ENH: batchsize cannot be passed as input right now
        )

        embedding_av = []
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)
                embedding_av.append(out.detach().cpu().numpy())

        embedding_av = np.vstack(embedding_av)

        return embedding_av, embedding_av, embedding_av  # HOTFIX

    def ST_wrapper(self):
        # this function was not needed so far therefore not implemented
        pass


def compare_model_weights(model1, model2, atol=1e-7, verbose=True):
    """
    Compare the weights of two ModelProjector instances to check if they're identical

    Args:
        model1: First ModelProjector instance
        model2: Second ModelProjector instance
        atol: Absolute tolerance for floating point comparisons
        verbose: Whether to print detailed comparison results

    Returns:
        bool: True if weights are identical, False otherwise
    """
    weights_identical = True
    differences = []

    # Get state dictionaries (model weights)
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if they have the same keys
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        missing_in_2 = set(state_dict1.keys()) - set(state_dict2.keys())
        missing_in_1 = set(state_dict2.keys()) - set(state_dict1.keys())
        if verbose:
            if missing_in_2:
                differences.append(f"Keys missing in model2: {missing_in_2}")
            if missing_in_1:
                differences.append(f"Keys missing in model1: {missing_in_1}")
        weights_identical = False
    else:
        # Check if the parameter values are the same
        for key in state_dict1:
            if not torch.allclose(state_dict1[key], state_dict2[key], atol=atol):
                if verbose:
                    # Calculate max difference for informational purposes
                    max_diff = (state_dict1[key] - state_dict2[key]).abs().max().item()
                    differences.append(
                        f"Parameters differ for key '{key}' (max difference: {max_diff:.6e})"
                    )
                weights_identical = False

    # Print differences if verbose
    if verbose and not weights_identical:
        print("Model weights are not identical. Differences found:")
        for diff in differences:
            print(f"- {diff}")
    elif verbose and weights_identical:
        print("Model weights are identical!")

    return weights_identical
