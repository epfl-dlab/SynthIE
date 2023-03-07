import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import jsonlines
import math
import numpy as np

from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from .general_helpers import unflatten_dict


@rank_zero_only
def upload_outputs_to_wandb(hparams_to_log, output_dir, logger):
    if isinstance(logger, LoggerCollection):
        loggers = logger
    else:
        loggers = [logger]

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            output_files = os.listdir(output_dir)
            output_files = [os.path.relpath(os.path.join(output_dir, f)) for f in output_files]

            logger.experiment.save(f"{output_dir}/*", base_path=".", policy="now")
            logger.experiment.config["output_files"] = output_files
            logger.experiment.config.update(hparams_to_log, allow_val_change=True)


def restore_outputs_from_wandb(wandb_run_path, exp_dir):
    import wandb

    api = wandb.Api()
    wrun = api.run(wandb_run_path)
    for file in wrun.config["output_files"]:
        if not os.path.isfile(os.path.join(exp_dir, file)):
            wandb.restore(file, run_path=wandb_run_path, root=exp_dir)


def restore_results_from_wandb(wandb_run, exp_dir):
    for f in wandb_run.files():
        if f.name == "results.json":
            f.download(root=exp_dir, replace=True)


def read_outputs(outputs_dir):
    items_dict = defaultdict(dict)
    for filename in os.listdir(outputs_dir):
        if not filename.endswith(".jsonl.gz"):
            continue

        input_file_path = os.path.join(outputs_dir, filename)
        with gzip.open(input_file_path, "r+") as fp:
            reader = jsonlines.Reader(fp)
            for element in reader:
                assert "id" in element
                # when running inference using ddp, due to non-even splits, a few datapoints might be duplicated
                # assert element["id"] not in items_dict
                # however we will always consider only one prediction (the last one)
                items_dict[element["id"]].update(element)

    items = [items_dict[_id] for _id in sorted(items_dict.keys())]
    return items


def read_results(exp_dir):
    results_path = os.path.join(exp_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    return results


def write_results(exp_dir, results):
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as outfile:
        json.dump(results, outfile)


def _get_relative_path_to_logs(path):
    parts = path.split("/")
    for i, p in enumerate(parts):
        if p == "logs":
            break
    return "/".join(parts[i:])


def get_predictions_dir_path(output_dir, create_if_not_exists=True):
    if output_dir is not None:
        predictions_folder = os.path.join(output_dir, "predictions")
    else:
        predictions_folder = "predictions"

    if create_if_not_exists:
        Path(predictions_folder).mkdir(parents=True, exist_ok=True)

    return predictions_folder


def prepare_data_for_experiment(wandb_run_path, work_dir, log_func):
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)

    wandb_run_config = unflatten_dict(run.config)
    wandb_run_hydra_config = wandb_run_config["hydra_config"]
    exp_dir = _get_relative_path_to_logs(wandb_run_hydra_config["output_dir"])
    abs_exp_dir = os.path.join(work_dir, exp_dir)

    if os.path.isdir(abs_exp_dir) and os.path.isdir(os.path.join(abs_exp_dir, "predictions")):
        log_func(f"Experiment directory already exists: {abs_exp_dir}")
    else:
        if os.path.isdir(abs_exp_dir):
            log_func(f"Experiment directory ({abs_exp_dir}) already exists, but the predictions were not found found.")
        else:
            log_func(f"Experiment directory was not found at: {abs_exp_dir}")
        log_func(f"Synchronizing with the data from WandB at: {abs_exp_dir}")
        # Recreates the predictions directory at the same location relative to the project directory as the original run
        restore_outputs_from_wandb(wandb_run_path, work_dir)

    log_func(f"Loading the existing results")
    restore_results_from_wandb(run, abs_exp_dir)
    return wandb_run_config, wandb_run_hydra_config, abs_exp_dir


def _get_rel_sf2dp_indices(dataset, consider_prediction_triplets):
    rel_sf2dp_indices = defaultdict(set)

    for idx, sample in enumerate(dataset):
        target_triplets = sample["target_triplets"]
        for s, r, o in target_triplets:
            rel_sf2dp_indices[r].add(idx)

        if consider_prediction_triplets:
            prediction_triplets = sample["prediction_triplets"]
            for s, r, o in prediction_triplets:
                rel_sf2dp_indices[r].add(idx)

    return rel_sf2dp_indices


def _get_bucket_idx2dp_indices(rel_sf2dp_indices, bucket_idx2reference_rels_sfs):
    bucket_idx2dp_indices = {}

    for bucket_idx, ref_rels_sfs in bucket_idx2reference_rels_sfs.items():
        dp_indices = set().union(*[rel_sf2dp_indices[rel_sf] for rel_sf in ref_rels_sfs])
        bucket_idx2dp_indices[bucket_idx] = dp_indices

    return bucket_idx2dp_indices


def get_macro_metrics_computation_metadata(dataset, consider_prediction_triplets=True):
    rel_sf2dp_indices = _get_rel_sf2dp_indices(dataset, consider_prediction_triplets)

    bucket_idx2reference_rels_sfs = {bucket_idx: set([rel_sf]) for bucket_idx, rel_sf in enumerate(rel_sf2dp_indices)}
    bucket_idx2dp_indices = _get_bucket_idx2dp_indices(rel_sf2dp_indices, bucket_idx2reference_rels_sfs)

    return {
        "rel_sf2dp_indices": rel_sf2dp_indices,
        "bucket_idx2reference_rels_sfs": bucket_idx2reference_rels_sfs,
        "bucket_idx2dp_indices": bucket_idx2dp_indices,
    }


class RelationCentricBucketing:
    def __init__(self, train_dataset, base=2):
        self.train_dataset = train_dataset

        rel2freq = train_dataset.rel2freq
        max_freq = max(rel2freq.values())
        bin_edges, bin_labels = self.get_bin_edges(max_freq, base, return_labels=True)

        bucket_idx2reference_rels_sfs = defaultdict(set)

        for rel_sf in rel2freq:
            bucket_idx = self.get_bucket_idx_for_occ_count(bin_edges, rel2freq[rel_sf])
            bucket_idx2reference_rels_sfs[bucket_idx].add(rel_sf)

        self.bucket_idx2reference_rels_sfs = bucket_idx2reference_rels_sfs

        self.rel2freq = rel2freq
        self.bucket_idx2freq = {
            bucket_idx: sum([rel2freq[rel_sf] for rel_sf in rel_sfs])
            for bucket_idx, rel_sfs in bucket_idx2reference_rels_sfs.items()
        }
        self.bucket_idx2num_rels = {
            bucket_idx: len(rel_sfs) for bucket_idx, rel_sfs in bucket_idx2reference_rels_sfs.items()
        }
        self.bucket_idx2label = {bucket_idx: bin_labels[bucket_idx] for bucket_idx in bucket_idx2reference_rels_sfs}

    @staticmethod
    def get_bin_edges(_max, base, return_labels=False):
        """Returns [0, base^0, base^1, ..., base^k] such that k is the smallest number for which _max < base^k"""
        bin_edges = [0]

        for power in range(math.ceil(np.log(_max + 1) / np.log(base)) + 1):
            bin_edges.append(base**power)

        if bin_edges[-1] == _max:
            power += 1
            bin_edges.append(2**power)

        if not return_labels:
            return bin_edges

        bin_labels = ["None"]
        for i in range(1, len(bin_edges) - 1):
            bin_labels.append(f"${base}^{{{i - 1}}}$")

        return bin_edges, bin_labels

    @staticmethod
    def get_bucket_idx_for_occ_count(bin_edges, value):
        """Returns the k for which bin_edges[k] <= value < bin_edges[k+1]"""
        assert value >= bin_edges[0]
        assert value < bin_edges[-1]

        for i in range(0, len(bin_edges) - 1):
            if value < bin_edges[i + 1]:
                return i

        raise Exception("The value is outside of the range covered by the bin edges")


def get_rel_centric_bucket_metrics_computation_metadata(
    train_dataset, output_dataset, consider_prediction_triplets=True, base=2
):
    relation_centric_bucketing = RelationCentricBucketing(train_dataset, base)
    bucket_idx2reference_rels_sfs = relation_centric_bucketing.bucket_idx2reference_rels_sfs

    rel_sf2dp_indices = _get_rel_sf2dp_indices(output_dataset, consider_prediction_triplets)
    bucket_idx2dp_indices = _get_bucket_idx2dp_indices(rel_sf2dp_indices, bucket_idx2reference_rels_sfs)

    return {
        "rel_sf2dp_indices": rel_sf2dp_indices,
        "bucket_idx2reference_rels_sfs": bucket_idx2reference_rels_sfs,
        "bucket_idx2dp_indices": bucket_idx2dp_indices,
        "rel2freq": relation_centric_bucketing.rel2freq,
        "bucket_idx2freq": relation_centric_bucketing.bucket_idx2freq,
        "bucket_idx2num_rels": relation_centric_bucketing.bucket_idx2num_rels,
        "bucket_idx2label": relation_centric_bucketing.bucket_idx2label,
    }


def get_num_target_triplets_centric_bucket_metrics_computation_metadata(output_dataset):
    bucket_idx2dp_indices = {}
    for idx, dp in enumerate(output_dataset):
        num_triplets = len(dp["target_triplets"])
        if num_triplets not in bucket_idx2dp_indices:
            bucket_idx2dp_indices[num_triplets] = set()

        bucket_idx2dp_indices[num_triplets].add(idx)

    bucket_idx2label = {bucket_idx: str(bucket_idx) for bucket_idx in bucket_idx2dp_indices}

    return {
        "bucket_idx2dp_indices": bucket_idx2dp_indices,
        "bucket_idx2label": bucket_idx2label,
    }


def get_dataset_id(from_cfg: bool, **kwargs):
    if from_cfg:
        dataset_cfg = kwargs["dataset_cfg"]
        name = dataset_cfg["name"]
        split = dataset_cfg["load_dataset_params"]["split"]
        filter_on_num_tokens = dataset_cfg["load_dataset_params"]["filter_on_num_tokens"]
        linearization_id = dataset_cfg["linearization_class_id"]
        constrained_world = dataset_cfg["constrained_world"]
    else:
        name = kwargs["name"]
        split = kwargs["split"]
        filter_on_num_tokens = kwargs["filter_on_num_tokens"]
        linearization_id = kwargs["linearization_id"]
        constrained_world = kwargs["constrained_world"]

    return (
        f"name-{name}_"
        f"split-{split}_"
        f"filtering-{filter_on_num_tokens}_"
        f"linearization-{linearization_id}_"
        f"constrained_world-{constrained_world}"
    )


def get_percentile_based_ci(scores, confidence_level):
    alpha = (1 - confidence_level) / 2

    interval = alpha, 1 - alpha

    def percentile_fun(a, q):
        return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(scores, interval[0] * 100)
    ci_u = percentile_fun(scores, interval[1] * 100)
    return ci_l, np.mean(scores), ci_u


def get_std_based_ci(scores):
    std = np.std(scores)
    mean = np.mean(scores)

    ci_l = mean - 1.96 * std
    ci_u = mean + 1.96 * std
    return ci_l, mean, ci_u


class Results:
    def __init__(self, exp_dir):
        self.data = read_results(exp_dir)

    @staticmethod
    def _get_score(data, metric_id, per_bucket, dataset_id=None):
        if dataset_id is None:
            score_obj = data[metric_id]["score"]
        else:
            score_obj = data[metric_id][dataset_id]["score"]

        if per_bucket:
            return score_obj[0]

        return score_obj[1]

    def get_score(self, metric_id, per_bucket, dataset_id=None):
        return self._get_score(self.data, metric_id, per_bucket, dataset_id)

    def get_bootstrap_runs_scores(self, metric_id, dataset_id, per_bucket):
        if dataset_id:
            seed2result = self.data[metric_id][dataset_id]["bootstrap_runs"]
        else:
            seed2result = self.data[metric_id]["bootstrap_runs"]

        if per_bucket:
            return {seed: result[0] for seed, result in seed2result.items()}

        return {seed: result[1] for seed, result in seed2result.items()}

    def _get_scores_per_seed(self, metric_id, dataset_id, n_bootstrap_samples, per_bucket=False):
        seed2result = self.get_bootstrap_runs_scores(metric_id, dataset_id, per_bucket)

        if not per_bucket:
            results = [
                result for seed, result in sorted(seed2result.items(), key=lambda x: int(x[0]))[:n_bootstrap_samples]
            ]

            if len(results) < n_bootstrap_samples:
                raise ValueError(f"{n_bootstrap_samples} bootstrap results for {metric_id} are not available")

            return results

        # ~~~ Precautionary check ~~~
        bucket_indices = None
        for seed, results in seed2result.items():
            if bucket_indices is None:
                bucket_indices = set(results.keys())
            else:
                assert bucket_indices == set(results.keys()), "The buckets across all bootstrap runs aren't the same"

        ci_dict = {}
        for bucket_idx in bucket_indices:
            results = [
                results[bucket_idx]
                for seed, results in sorted(seed2result.items(), key=lambda x: x[0])[:n_bootstrap_samples]
            ]

            if len(results) < n_bootstrap_samples:
                raise ValueError(f"{n_bootstrap_samples} bootstrap results for {metric_id} are not available")

            ci_dict[bucket_idx] = results

        return ci_dict

    def get_metadata(self, metric_id, dataset_id=None):
        if dataset_id is None:
            return self.data[metric_id]["metadata"]

        return self.data[metric_id][dataset_id]["metadata"]

    def get_percentile_based_ci(
        self, metric_id, confidence_level, n_bootstrap_samples, dataset_id=None, per_bucket=False
    ):
        """Returns the `confidence_level`% confidence interval based on the empirical dist. of the bootstrap samples."""

        if per_bucket:
            ci_dict = self._get_scores_per_seed(metric_id, dataset_id, n_bootstrap_samples, per_bucket=True)
            return {key: get_percentile_based_ci(results, confidence_level) for key, results in ci_dict.items()}

        results = self._get_scores_per_seed(metric_id, dataset_id, n_bootstrap_samples, per_bucket=False)
        return get_percentile_based_ci(results, confidence_level)

    def get_std_based_ci(self, metric_id, n_bootstrap_samples, dataset_id=None, per_bucket=False):
        """Returns the 95% confidence interval based on the standard deviation of the bootstrap samples"""

        if per_bucket:
            ci_dict = self._get_scores_per_seed(metric_id, dataset_id, n_bootstrap_samples, per_bucket=True)
            return {key: get_std_based_ci(results) for key, results in ci_dict.items()}

        results = self._get_scores_per_seed(metric_id, dataset_id, n_bootstrap_samples, per_bucket=False)
        return get_std_based_ci(results)
