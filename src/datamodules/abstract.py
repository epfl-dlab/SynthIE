from abc import ABC
from typing import Optional, Callable

import hydra
import torch

import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import src.utils as utils

log = utils.get_pylogger(__name__)


class AbstractDataset(Dataset):
    """
    A dataset implements 2 functions
        - __len__  (returns the number of samples in our dataset)
        - __getitem__ (returns a sample from the dataset at the given index idx)
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.random_state = np.random.RandomState(self.params["seed"])

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def _load_data(self):
        raise NotImplementedError()

    def get_random_sample(self):
        idx = self.random_state.randint(0, len(self.data))
        return self.data[idx]

    def get_random_subset(self, k):
        idxs = self.random_state.choice(len(self.data), k, replace=False)
        return [self.data[idx] for idx in idxs]

    def get_bootstrapped_data(self, seed):
        data = self.data
        num_datapoints = len(data)

        random_state = np.random.RandomState(seed)
        bootstrap_ids = random_state.choice(len(self.data), num_datapoints, replace=True)

        bootstrap_data = [data[i] for i in bootstrap_ids]
        return bootstrap_data


class AbstractOutputDataset(AbstractDataset, ABC):
    @staticmethod
    def get_predictions(item, key="prediction", top_pred_only=True):
        preds = item[key]

        if top_pred_only and not isinstance(preds, str):
            return preds[0]

        return preds

    @staticmethod
    def get_targets(item, key="target", wrap_in_list=False):
        tgts = item[key]

        if wrap_in_list and not isinstance(tgts, list):
            return [tgts]

        return tgts


class AbstractPLDataModule(LightningDataModule, ABC):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(self, seed: int, collate_fn: Callable = None, num_workers: int = 0, **kwargs):
        """

        Parameters
        ----------
        seed : Random seed
        collate_fn : The collate function which is model specific
        num_workers : Setting num_workers as a positive integer will turn on multiprocess data loading with the specified number of loader worker processes

        kwargs: dataset specific parameters

        Returns
        -------
        An instance of the Grid dataset that extends pytorch_lightning.DataModule
        """
        super().__init__()
        self.collate_fn = collate_fn
        self.tokenizer = None

        # Concerning the loaders
        self.num_workers = num_workers
        self.seed = seed

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset_parameters = kwargs["dataset_parameters"]

    def set_collate_fn(self, collate_fn):
        self.collate_fn = collate_fn

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test.
        Note that the result of hydra instantiation inherits abstract_grid_dataset for which we have:
            def __getitem__(self, idx):
                return {"id": self.data[idx][0], "text": self.data[idx][1]}

        So when a sample from such dataset is picked (self.data_train/val/test), the sample has the
        above form, and to extract its text, we need to pass the keyword "text".
        """
        assert stage in set(["fit", "validate", "test", None])

        if (stage == "fit" or stage is None) and self.data_train is None:
            self.data_train = hydra.utils.instantiate(
                self.dataset_parameters["train"]["dataset"], tokenizer=self.tokenizer
            )
            log.info("The train dataset has been loaded and has %d samples" % len(self.data_train))

        if (stage == "validate" or stage == "fit" or stage is None) and self.data_val is None:
            if "datasets" in self.dataset_parameters["val"]:
                self.data_val = [
                    hydra.utils.instantiate(dataset_parameters, tokenizer=self.tokenizer)
                    for dataset_parameters in self.dataset_parameters["val"]["datasets"]
                ]
                log.info(
                    f"The validation datasets have been loaded and have `{[len(data) for data in self.data_val]}` samples"
                )
            else:
                self.data_val = hydra.utils.instantiate(
                    self.dataset_parameters["val"]["dataset"], tokenizer=self.tokenizer
                )
                log.info("The validation dataset has been loaded and has %d samples" % len(self.data_val))

        if (stage == "test" or stage is None) and self.data_test is None:
            self.data_test = hydra.utils.instantiate(
                self.dataset_parameters["test"]["dataset"], tokenizer=self.tokenizer
            )
            log.info("The test dataset has been loaded and has %d samples" % len(self.data_test))

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.dataset_parameters["train"]["dataloader"]["batch_size"],
            num_workers=self.dataset_parameters["train"]["dataloader"]["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=False,
            shuffle=True,
            generator=g,
        )

    def _get_dataloader(self, data, dataloader_parameters, drop_last, shuffle):
        return DataLoader(
            dataset=data,
            batch_size=dataloader_parameters["batch_size"],
            num_workers=dataloader_parameters["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=drop_last,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        if isinstance(self.data_val, list):
            return [
                self._get_dataloader(data, self.dataset_parameters["val"]["dataloader"], drop_last=False, shuffle=False)
                for data in self.data_val
            ]

        return self._get_dataloader(
            self.data_val, self.dataset_parameters["val"]["dataloader"], drop_last=False, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.dataset_parameters["test"]["dataloader"]["batch_size"],
            num_workers=self.dataset_parameters["test"]["dataloader"]["num_workers"],
            collate_fn=self.collate_fn,
            drop_last=False,
            shuffle=False,
        )
