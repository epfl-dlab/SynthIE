import collections
import json
import warnings
from typing import List, Callable

import jsonlines
import hydra
import gzip
import time
import os
import shutil

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase, LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src import utils
from importlib.util import find_spec

log = utils.get_pylogger(__name__)


def run_task(cfg: DictConfig, run_func: Callable) -> None:
    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    utils.general_helpers.extras(cfg)

    # execute the task
    try:
        start_time = time.time()
        run_func(cfg)
    except Exception as ex:
        log.exception("")  # save exception to `.log` file
        raise ex
    finally:
        #     ToDo log also:
        #     - Number of CPU cores
        #     - Type of CPUs
        #     - Number of GPU cores
        #     - Type of GPUs
        #     - Number of GPU hours
        current_time_stamp = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        path = Path(cfg.output_dir, f"exec_time_{current_time_stamp}.log")
        content = (
            f"Execution time: {time.time() - start_time:.2f} seconds "
            f"-- {(time.time() - start_time) / 60:.2f} minutes "
            f"-- {(time.time() - start_time) / 3600:.2f} hours"
        )
        log.info(content)
        utils.general_helpers.save_string_to_file(path, content)  # save task execution time (even if exception occurs)
        utils.general_helpers.close_loggers()  # close loggers (even if exception occurs so multirun won't fail)
        log.info(f"Output dir: {cfg.output_dir}")

        # Temporary solution to Hydra + PL + DDP issue
        # https://github.com/Lightning-AI/lightning/pull/11617#issuecomment-1245842064
        # https://github.com/ashleve/lightning-hydra-template/issues/393
        # problem should be resolved in PL version 1.8.3
        _clean_working_dir_from_subprocesses_output(cfg.work_dir, cfg.output_dir)


@rank_zero_only
def _clean_working_dir_from_subprocesses_output(work_dir, output_dir):
    # Move the hydra folder
    hydra_folder_src = os.path.join(work_dir, ".hydra")
    hydra_folder_target = os.path.join(output_dir, ".hydra_subprocesses")
    if os.path.exists(hydra_folder_src):
        shutil.move(hydra_folder_src, hydra_folder_target)

    # Move the logs
    files = [f for f in os.listdir(work_dir) if f.endswith(".log")]
    for f in files:
        shutil.move(os.path.join(work_dir, f), os.path.join(output_dir, f))


@rank_zero_only
def _move_predictions_for_subprocesses(predictions_dir_src, predictions_dir_dst):
    if os.path.exists(predictions_dir_src):
        for f in os.listdir(predictions_dir_src):
            shutil.move(os.path.join(predictions_dir_src, f), os.path.join(predictions_dir_dst, f))
        shutil.rmtree(predictions_dir_src)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings
    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.
    Additionally, it saves:
    - Number of model parameters
    """
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["hydra_config"] = cfg
    # Add number of model parameters to logged information
    hparams["params"] = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "non_trainable": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }

    for key in hparams:
        if isinstance(hparams[key], DictConfig):
            hparams[key] = OmegaConf.to_container(hparams[key], resolve=True)

    # send hparams to all loggers
    for logger in trainer.loggers:
        # ToDo: The config's nested structure is not preserved by WandB. Why? Is this a bug fixed in newer versions?
        logger.log_hyperparams(hparams)


@rank_zero_only
def save_string_to_file(path: str, content: str, append_mode=True) -> None:
    """Save string to file in rank zero mode (only on the master process in multi-GPU setup)."""
    mode = "a+" if append_mode else "w+"
    with open(path, mode) as file:
        file.write(content)


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def get_absolute_path(path):
    """Get absolute path (relative to the original working directory) from a (potentially) relative path."""
    if not os.path.isabs(path):
        return os.path.join(hydra.utils.get_original_cwd(), path)
    return path


def unflatten_dict(dictionary: dict) -> dict:
    result_dict = dict()
    for key, value in dictionary.items():
        parts = key.split("/")
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict


def rec_dict_update(d, u):
    """Performs a multilevel overriding of the values in dictionary d with the values of dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = rec_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def write_gzipped_jsonlines(path_to_file, data, mode="w"):
    with gzip.open(path_to_file, mode) as fp:
        json_writer = jsonlines.Writer(fp)
        json_writer.write_all(data)


def read_gzipped_jsonlines(path_to_file):
    with gzip.open(path_to_file, "r") as fp:
        json_reader = jsonlines.Reader(fp)
        return list(json_reader)


def read_jsonlines(path_to_file):
    with open(path_to_file, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonlines(path_to_file, data, mode="w"):
    with jsonlines.open(path_to_file, mode) as writer:
        writer.write_all(data)


def get_list_of_dicts(dict_of_lists):
    """
    Converts a dictionary of lists to a list of dictionaries

    Parameters
    ----------
    dict_of_lists: A dict of lists, each element of the list corresponding to one item.
             For example: {'id': [1,2,3], 'val': [72, 42, 32]}

    Returns
    -------
    A list of dicts of individual items.
    For example: [{'id': 1, 'val': 72}, {'id': 2, 'val': 42}, {'id': 3, 'val': 32}]

    """
    keys = dict_of_lists.keys()
    values = [dict_of_lists[key] for key in keys]
    items = [dict(zip(keys, item_vals)) for item_vals in zip(*values)]
    return items


def chunk_elements(elements, n_items_per_group):
    """

    Parameters
    ----------
    elements: elements to be grouped, [s1, s2, s3, s3, s5, s6]
    n_items_per_group: integer if all groups are of equal size, or a list of varying group lengths

    Returns
    -------
    A list of grouped sequences.

    Example 1:
    elements=[s1,s2,s3,s3,s5,s6]
    n_items_per_group=3
    returns: [[s1,s2,s3], [s4,s5,s6]]

    Example 2:
    elements=[s1,s2,s3,s3,s5,s6]
    n_items_per_group=[2,4]
    returns: [[s1,s2], [s3,s4,s5,s6]]

    """
    if isinstance(n_items_per_group, int):
        assert len(elements) % n_items_per_group == 0
        n_items_per_group = [n_items_per_group for _ in range(len(elements) // n_items_per_group)]

    grouped_sequences = []
    start_idx = 0
    for n_items in n_items_per_group:
        grouped_sequences.append(elements[start_idx : start_idx + n_items])
        start_idx += n_items

    return grouped_sequences


def log_df(path, df, logger):
    if isinstance(logger, LoggerCollection):
        loggers = logger
    else:
        loggers = [logger]

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.log({path: df})
