from abc import ABC
from typing import List, Set, Union, Tuple, Dict

import numpy as np
import torch
from torchmetrics import Metric


class IEAbstractTorchMetric(Metric, ABC):
    full_state_update = True

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_predicted", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    @staticmethod
    def _process_test_sample(pred_triples: set, target_triples: set):
        num_matched = len(target_triples.intersection(pred_triples))
        num_predicted = len(pred_triples)
        num_target = len(target_triples)

        return num_matched, num_predicted, num_target

    @classmethod
    def _process_lists(
        cls, preds: List[Set[Tuple[str, str, str]]], targets: List[Set[Tuple[str, str, str]]]
    ) -> Tuple[float, float, float]:
        num_correct = 0
        num_predicted = 0
        num_target = 0

        for p, t in zip(preds, targets):
            n_matched, n_predicted, n_target = cls._process_test_sample(pred_triples=p, target_triples=t)

            num_correct += n_matched
            num_predicted += n_predicted
            num_target += n_target

        return num_correct, num_predicted, num_target

    def update(self, preds: List[Set[Tuple[str, str, str]]], targets: List[Set[Tuple[str, str, str]]]):
        assert len(preds) == len(targets)

        num_correct = []
        num_predicted = []
        num_target = []

        for t, p in zip(targets, preds):
            n_matched, n_predicted, n_target = self._process_test_sample(pred_triples=p, target_triples=t)

            num_correct.append(n_matched)
            num_predicted.append(n_predicted)
            num_target.append(n_target)

        num_correct = torch.tensor(num_correct).long()
        num_predicted = torch.tensor(num_predicted).long()
        num_target = torch.tensor(num_target).long()

        self.total_correct += torch.sum(num_correct)
        self.total_predicted += torch.sum(num_predicted)
        self.total_target += torch.sum(num_target)

    @staticmethod
    def _compute(correct, predicted, target, use_tensor=False) -> Union[float, torch.Tensor]:
        raise NotImplementedError()

    def compute_from_dataset(
        self, dataset, seed=None, bucket_metadata_dict=None, dp_centric_bucket_metadata_dict=None
    ) -> Union[float, Dict[str, float]]:

        assert not (
            bucket_metadata_dict and dp_centric_bucket_metadata_dict
        ), "Only one of bucket_metadata_dict and dp_centric_bucket_metadata_dict should be provided"

        if bucket_metadata_dict is not None:
            per_bucket_performance = self._compute_from_dataset_per_bucket_relation_centric(
                dataset, bucket_metadata_dict=bucket_metadata_dict, seed=seed
            )

            return per_bucket_performance, np.mean(list(per_bucket_performance.values()))

        if dp_centric_bucket_metadata_dict is not None:
            per_bucket_performance = self._compute_from_dataset_per_bucket_dp_centric(
                dataset, bucket_metadata_dict=dp_centric_bucket_metadata_dict, seed=seed
            )

            return per_bucket_performance, np.mean(list(per_bucket_performance.values()))

        return None, self._compute_from_dataset_micro(dataset, seed=seed)

    def _compute_from_dataset_micro(self, dataset, seed=None) -> float:
        if seed is not None:
            # ~~~ Get a bootstrap dataset, but keep track of the original data ~~~
            original_data = dataset.data
            dataset.data = dataset.get_bootstrapped_data(seed=seed)

        # ~~~ Retrieve the predictions and the targets ~~~
        preds = [item["prediction_triplets"] for item in dataset]
        targets = [item["target_triplets"] for item in dataset]

        # ~~~ Compute performance ~~~
        num_correct, num_predicted, num_target = self._process_lists(preds, targets)
        score = self._compute(num_correct, num_predicted, num_target)

        if seed is not None:
            # restore the original dataset
            dataset.data = original_data

        return score

    def _compute_from_dataset_per_bucket_relation_centric(self, dataset, bucket_metadata_dict, seed=None):
        original_data = dataset.data

        bucket_idx2score = {}
        bucket_idx2reference_rels_sfs = bucket_metadata_dict["bucket_idx2reference_rels_sfs"]
        bucket_idx2dp_indices = bucket_metadata_dict["bucket_idx2dp_indices"]

        for bucket_idx in bucket_idx2reference_rels_sfs.keys():
            # ~~~ Filter the dataset to the triplets concerning relations pertaining to the current bucket ~~~
            dataset.data = dataset.get_filtered_data(
                relations_to_consider=bucket_idx2reference_rels_sfs[bucket_idx],
                entities_to_consider=None,
                indices_to_consider=bucket_idx2dp_indices[bucket_idx],
            )

            # ~~~ Get a bootstrap dataset from the filtered version ~~~
            if seed is not None:
                dataset.data = dataset.get_bootstrapped_data(seed=seed)

            # ~~~ Retrieve the predictions and the targets ~~~
            preds = [item["prediction_triplets"] for item in dataset]
            targets = [item["target_triplets"] for item in dataset]

            # ~~~ Compute performance ~~~
            num_correct, num_predicted, num_target = self._process_lists(preds, targets)
            score = self._compute(num_correct, num_predicted, num_target)

            bucket_idx2score[bucket_idx] = score
            dataset.data = original_data

        return bucket_idx2score

    def _compute_from_dataset_per_bucket_dp_centric(self, dataset, bucket_metadata_dict, seed=None):
        original_data = dataset.data

        bucket_idx2score = {}
        bucket_idx2dp_indices = bucket_metadata_dict["bucket_idx2dp_indices"]

        for bucket_idx in bucket_idx2dp_indices.keys():
            # ~~~ Filter the dataset to the datapoints pertaining to the current bucket ~~~
            dataset.data = [dataset.data[idx] for idx in bucket_idx2dp_indices[bucket_idx]]

            # ~~~ Get a bootstrap dataset from the filtered version ~~~
            if seed is not None:
                dataset.data = dataset.get_bootstrapped_data(seed=seed)

            # ~~~ Retrieve the predictions and the targets ~~~
            preds = [item["prediction_triplets"] for item in dataset]
            targets = [item["target_triplets"] for item in dataset]

            # ~~~ Compute performance ~~~
            num_correct, num_predicted, num_target = self._process_lists(preds, targets)
            score = self._compute(num_correct, num_predicted, num_target)

            bucket_idx2score[bucket_idx] = score
            dataset.data = original_data

        return bucket_idx2score
