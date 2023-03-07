from src.utils import hydra_custom_resolvers
import hydra
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
from queue import Queue

import concurrent
import os

from typing import List, Dict, Union
import wandb
import pytorch_lightning as pl
from tqdm import tqdm

import src.utils.general_helpers as general_helpers
import src.utils.evaluation_helpers as evaluation_helpers
from src.utils.evaluation_helpers import Results
from src import utils

DEBUG = False
log = utils.get_pylogger(__name__)


def get_score_from_metric(
    cfg, output_dataset, bucket_metadata_dict, dp_centric_bucket_metadata_dict, metric_alias, seed, device=None
):
    assert not (
        bucket_metadata_dict and dp_centric_bucket_metadata_dict
    ), "Only one of bucket_metadata_dict and dp_centric_bucket_metadata_dict should be provided"

    # Load metric
    if "device" in cfg.metric[metric_alias]:
        # Load the metric to a specific device
        pass
    else:
        metric = hydra.utils.instantiate(cfg.metric[metric_alias], _recursive_=True)

    # Calculate score
    if bucket_metadata_dict:
        corpus_score = metric.compute_from_dataset(output_dataset, bucket_metadata_dict=bucket_metadata_dict, seed=seed)
    elif dp_centric_bucket_metadata_dict:
        corpus_score = metric.compute_from_dataset(
            output_dataset, dp_centric_bucket_metadata_dict=dp_centric_bucket_metadata_dict, seed=seed
        )
    else:
        corpus_score = metric.compute_from_dataset(output_dataset, seed=seed)

    return corpus_score


def _instantiate_output_dataset_instances_queue(output_dataset_cfg, num_workers):
    output_dataset_instances_queue = Queue(num_workers)
    for _ in range(num_workers):
        output_dataset_instances_queue.put(hydra.utils.instantiate(output_dataset_cfg, _recursive_=False))

    return output_dataset_instances_queue


def get_bootstrap_run_scores(
    cfg,
    results,
    bucket_metadata_dict,
    dp_centric_bucket_metadata_dict,
    starting_seed,
    num_workers=1,
    output_dataset_instances_queue=None,
):
    bootstrap_run_scores = results.get("bootstrap_runs", {})

    run_scores_for_ci = []

    if output_dataset_instances_queue is None:
        output_dataset_instances_queue = _instantiate_output_dataset_instances_queue(cfg.output_dataset, num_workers)

    # Use one instance of the metric for each worker
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(cfg.bootstrap_n)):
            seed = starting_seed + i
            output_dataset = output_dataset_instances_queue.get()

            # ~~~ Read the precomputed result for the seed (if it has already computed) ~~~
            if seed in bootstrap_run_scores:
                if not cfg.get("silent", False):
                    log.info(f"Score for seed {seed} was already computed.")
                run_scores_for_ci.append(bootstrap_run_scores[seed])
                output_dataset_instances_queue.put(output_dataset)
                continue
            elif str(seed) in bootstrap_run_scores:
                if not cfg.get("silent", False):
                    log.info(f"Score for seed {seed} was already computed.")
                run_scores_for_ci.append(bootstrap_run_scores[str(seed)])
                output_dataset_instances_queue.put(output_dataset)
                continue

            # ~~~ Compute the score for the specific seed ~~~
            if not cfg.get("silent", False):
                log.info(f"Computing the score for seed {seed}.")

            if DEBUG:
                bootstrap_run_scores[seed] = get_score_from_metric(
                    cfg=cfg,
                    output_dataset=output_dataset,
                    bucket_metadata_dict=bucket_metadata_dict,
                    dp_centric_bucket_metadata_dict=dp_centric_bucket_metadata_dict,
                    metric_alias=results["alias"],
                    seed=seed,
                    device=f"cuda:{(seed - starting_seed) % num_workers}",
                )
            else:
                future = executor.submit(
                    get_score_from_metric,
                    cfg=cfg,
                    output_dataset=output_dataset,
                    bucket_metadata_dict=bucket_metadata_dict,
                    dp_centric_bucket_metadata_dict=dp_centric_bucket_metadata_dict,
                    metric_alias=results["alias"],
                    seed=seed,
                    device=f"cuda:{(seed - starting_seed) % num_workers}",  # ToDo: This should be handled with a queue
                )
                bootstrap_run_scores[seed] = future.result()

            # ~~~ Log the score (if not executing silently) ~~~
            if not cfg.get("silent", False):
                if isinstance(bootstrap_run_scores[seed], tuple):
                    score = bootstrap_run_scores[seed][1]
                else:
                    score = bootstrap_run_scores[seed]
                log.info(f"Score for seed {seed}: {score * 100:.2f}%.")

            # ~~~ Add the score to the list of score that will be used to compute the confidence interval ~~~
            run_scores_for_ci.append(bootstrap_run_scores[seed])
            output_dataset_instances_queue.put(output_dataset)

    # ~~~ Update the cache of precomputed results if results for more runs were computed ~~~
    if len(results.get("bootstrap_runs", {})) < len(bootstrap_run_scores):
        results["bootstrap_runs"] = bootstrap_run_scores

    return run_scores_for_ci


def run_process_predictions(cfg: DictConfig) -> Dict[str, Dict[str, Union[str, float, List[float]]]]:
    """Contains the code for running evaluation on an output file.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Dict[str, Dict[str, Union[str, float, List[float]]]]: Dictionary containing the results of the evaluation.
    """
    assert cfg.output_dir is not None, "Path to the directory in which the predictions will be written must be given"
    cfg.output_dir = general_helpers.get_absolute_path(cfg.output_dir)
    log.info(f"Output directory: {cfg.output_dir}")

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    api = wandb.Api()
    run = api.run(cfg.wandb_run_path)

    wandb_run_config, wandb_run_hydra_config, abs_exp_dir = evaluation_helpers.prepare_data_for_experiment(
        cfg.wandb_run_path, cfg.work_dir, log.info
    )
    results = evaluation_helpers.read_results(abs_exp_dir)

    if cfg.get("override", False):
        log.info("Overriding the existing results.")
        results = {}
    else:
        log.info(results)

    log.info(f"Instantiating the output dataset and the metrics")
    linearization_class_id = wandb_run_hydra_config["datamodule"].get("linearization_class_id", None)
    if cfg.get("linearization_class_id", None):
        # Override the linearization class id if it is given in the config
        linearization_class_id = cfg.linearization_class_id
        log.info(f"Overriding the linearization class id with the value from the config `{linearization_class_id}`")
    elif linearization_class_id is None:
        # Left for backward compatibility
        log.info("Linearization class ID not specified. Using the default one `fully_expanded_et`")
        linearization_class_id = "fully_expanded_et"
    log.info(f"Linearization class ID: {linearization_class_id}")
    cfg.output_dataset.linearization_class_id = linearization_class_id
    cfg.output_dataset.data_dir = os.path.join(abs_exp_dir, "predictions")

    if cfg.get("datamodule", None):
        cfg.datamodule.dataset_parameters["train"]["dataset"]["linearization_class_id"] = linearization_class_id
        cfg.datamodule.dataset_parameters["train"]["dataset"][
            "linearization_class_id_for_filtering"
        ] = linearization_class_id

    output_dataset = hydra.utils.instantiate(cfg.output_dataset, _recursive_=False)
    metrics = hydra.utils.instantiate(cfg.metric, _recursive_=True)

    macro_metadata_dict = None

    train_cfg = None
    train_dataset = None
    rel_centric_bucket_metadata_dict = None
    dp_centric_bucket_metadata_dict = None

    log.info(f"Calculating corpus level metrics")
    for metric_id, metric in metrics.items():
        if metric.name in results and len(results[metric.name]) > 0:
            log.info(f"Skipped -- {metric.name} -- as it is already present in the results json.")
        else:
            results[metric.name] = {}
            results[metric.name]["score"] = metric.compute_from_dataset(output_dataset)
            results[metric.name]["alias"] = metric_id
            score = Results._get_score(results, metric.name, per_bucket=False)
            log.info(f"[{metric.name}] Results: {score * 100:.2f}%")

        if cfg.get("compute_macro_metrics", False):
            if macro_metadata_dict is None:
                macro_metadata_dict = evaluation_helpers.get_macro_metrics_computation_metadata(
                    output_dataset, consider_prediction_triplets=True
                )

            metric_name = "macro_" + metric.name
            if metric_name in results and len(results[metric_name]) > 0:
                log.info(f"Skipped -- {metric_name} -- as it is already present in the results json.")
            else:
                results[metric_name] = {}
                results[metric_name]["score"] = metric.compute_from_dataset(
                    output_dataset, bucket_metadata_dict=macro_metadata_dict
                )
                results[metric_name]["alias"] = metric_id
            score = Results._get_score(results, metric_name, per_bucket=False)
            log.info(f"[{metric_name}] Results: {score * 100:.2f}%")

        if cfg.get("compute_rel_centric_buckets_metrics", False):
            if rel_centric_bucket_metadata_dict is None:
                # ~~~ Load the train_dataset ~~~
                if train_dataset is None:
                    train_dataset = hydra.utils.instantiate(
                        cfg.datamodule.dataset_parameters["train"]["dataset"], tokenizer=None
                    )
                    train_cfg = cfg.datamodule.dataset_parameters["train"]["dataset"]
                rel_centric_bucket_metadata_dict = (
                    evaluation_helpers.get_rel_centric_bucket_metrics_computation_metadata(
                        train_dataset=train_dataset,
                        output_dataset=output_dataset,
                        consider_prediction_triplets=True,
                        base=2,
                    )
                )

            metric_name = "rel_centric_" + metric.name
            dataset_id = evaluation_helpers.get_dataset_id(dataset_cfg=train_cfg, from_cfg=True)
            if metric_name in results and dataset_id in results[metric_name]:
                log.info(f"Skipped -- {metric_name} [{dataset_id}] -- as it is already present in the results json.")
            else:
                results[metric_name] = results.get(metric_name, {})
                results[metric_name][dataset_id] = {}
                results[metric_name][dataset_id]["score"] = metric.compute_from_dataset(
                    output_dataset, bucket_metadata_dict=rel_centric_bucket_metadata_dict
                )
                results[metric_name][dataset_id]["alias"] = metric_id
                results[metric_name][dataset_id]["metadata"] = {
                    "train_cfg": OmegaConf.to_container(train_cfg, resolve=True)
                }

                serializable_metadata = deepcopy(rel_centric_bucket_metadata_dict)
                for key, value in serializable_metadata.items():
                    if isinstance(value, set):
                        serializable_metadata[key] = list(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, set):
                                serializable_metadata[key][k] = list(v)
                results[metric_name][dataset_id]["metadata"].update(serializable_metadata)

            score = Results._get_score(results, metric_name, dataset_id=dataset_id, per_bucket=False)
            log.info(f"[{metric_name}--{dataset_id}] Results: {score * 100:.2f}%")

        if cfg.get("compute_num_target_triplets_centric_buckets_metrics", False):
            if dp_centric_bucket_metadata_dict is None:
                dp_centric_bucket_metadata_dict = (
                    evaluation_helpers.get_num_target_triplets_centric_bucket_metrics_computation_metadata(
                        output_dataset=output_dataset,
                    )
                )

            metric_name = "num_target_triplets_centric_" + metric.name
            if metric_name in results and len(results[metric_name]) > 0:
                log.info(f"Skipped -- {metric_name} -- as it is already present in the results json.")
            else:
                results[metric_name] = {}
                results[metric_name]["score"] = metric.compute_from_dataset(
                    output_dataset, dp_centric_bucket_metadata_dict=dp_centric_bucket_metadata_dict
                )
                results[metric_name]["alias"] = metric_id

                serializable_metadata = deepcopy(dp_centric_bucket_metadata_dict)
                for key, value in serializable_metadata.items():
                    if isinstance(value, set):
                        serializable_metadata[key] = list(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            if isinstance(v, set):
                                serializable_metadata[key][k] = list(v)
                results[metric_name]["metadata"] = serializable_metadata

            score = Results._get_score(results, metric_name, per_bucket=False)
            log.info(f"[{metric_name}] Results: {score * 100:.2f}%")

        # Update the results json file
        evaluation_helpers.write_results(cfg.output_dir, results)

    output_dataset_instances_queue = None
    if cfg.get("bootstrap_n", None):
        bootstrap_n = cfg.bootstrap_n
        confidence_level = cfg.confidence_level

        log.info(
            f"Getting bootstrap samples and constructing intervals "
            f"at a {confidence_level * 100:.2f} confidence level using {bootstrap_n} samples."
        )

        for metric_name in results:
            # ~~~ Compute (or retrieve from cache) the scores for the bootstrap runs ~~~
            if output_dataset_instances_queue is None:
                output_dataset_instances_queue = _instantiate_output_dataset_instances_queue(
                    cfg.output_dataset, cfg.num_workers
                )
            parameters = {
                "cfg": cfg,
                "results": results[metric_name],
                "starting_seed": cfg.seed,
                "num_workers": cfg.num_workers,
                "output_dataset_instances_queue": output_dataset_instances_queue,
            }

            if metric_name.startswith("macro_"):
                parameters["bucket_metadata_dict"] = macro_metadata_dict
                parameters["dp_centric_bucket_metadata_dict"] = None
            elif metric_name.startswith("rel_centric_"):
                parameters["bucket_metadata_dict"] = rel_centric_bucket_metadata_dict
                parameters["dp_centric_bucket_metadata_dict"] = None
                parameters["results"] = results[metric_name][dataset_id]
            elif metric_name.startswith("num_target_triplets_centric_"):
                parameters["bucket_metadata_dict"] = None
                parameters["dp_centric_bucket_metadata_dict"] = dp_centric_bucket_metadata_dict
            else:
                parameters["bucket_metadata_dict"] = None
                parameters["dp_centric_bucket_metadata_dict"] = None
            bootstrap_run_scores = get_bootstrap_run_scores(**parameters)

            # ~~~ [Sanity check -- applied on the micro and macro scores] Construct the percentile based ci ~~~
            scores = [score[1] for score in bootstrap_run_scores]
            lower, mean_perc_based, upper = evaluation_helpers.get_percentile_based_ci(scores, confidence_level)
            log.info(
                f"[{metric_name}] Percentile based confidence interval: "
                f"[{lower * 100:.2f}, {mean_perc_based * 100:.2f}, {upper * 100:.2f}]"
            )

            # ~~~ Construct the standard deviation based ci ~~~
            lower, mean_std_based, upper = evaluation_helpers.get_std_based_ci(scores)
            log.info(
                f"[{metric_name}] Standard deviation based confidence interval: "
                f"[{lower * 100:.2f}, {mean_std_based * 100:.2f}, {upper * 100:.2f}]"
            )

            assert mean_perc_based == mean_std_based

    log.info(f"Experiment directory: {abs_exp_dir}")
    log.info(f"Writing the results to disk...")
    # Save the results to the experiment directory
    evaluation_helpers.write_results(abs_exp_dir, results)
    log.info(f"Uploading the results to wandb...")
    # Save the results to wandb
    run.upload_file(os.path.join(abs_exp_dir, "results.json"), root=abs_exp_dir)


@hydra.main(version_base="1.2", config_path="configs", config_name="process_predictions_root")
def main(hydra_config: DictConfig):
    utils.run_task(hydra_config, run_process_predictions)


if __name__ == "__main__":
    main()
