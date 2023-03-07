from collections import Counter
from typing import Union, Set, List

import jsonlines
import gzip
import os

from src.datamodules.abstract import AbstractDataset, AbstractPLDataModule, AbstractOutputDataset
from src.utils.constrained_generation_utils import read_constrained_world
from src.utils.evaluation_helpers import read_outputs, get_dataset_id

import src.utils as utils
import src.utils.ordering_heuristic_utils as ordering_heuristic_utils
from tqdm import tqdm


if __name__ == "__main__":
    log = utils.get_pylogger(__name__, stdout=True)
else:
    log = utils.get_pylogger(__name__)


class IEGenericDataset(AbstractDataset):
    def __init__(self, **params):
        super().__init__(params)
        self.tokenizer = params.get("tokenizer")

        self.data = None
        self.entities_to_keep = None
        self.relations_to_keep = None
        self.num_filtered_datapoints_constrained_world = 0
        self.num_filtered_datapoints_input_tokens = 0
        self.num_filtered_datapoints_target_tokens = 0

        self.linearization_class = utils.get_linearization_class(params["linearization_class_id"])
        self.linearization_class_for_filtering = utils.get_linearization_class(
            params["linearization_class_id_for_filtering"]
        )

        self._load_data(load_dataset_params=params["load_dataset_params"])

        self.statistics = {}
        self.ent2freq = None
        self.rel2freq = None
        self.compute_dataset_statistics(params.get("compute_frequency_dicts", False))

        if params.get("verify_triplet_ordering", False):
            log.info("Verifying ordering of triplets...")
            self.are_triplets_with_same_subject_consecutive(self.data)

        if params.get("count_datapoints_with_unk_in_target", False):
            num_datapoints_with_unk_in_target = self.count_datapoints_with_unk_in_target()
            log.info(f"Number of datapoints with unk in target: {num_datapoints_with_unk_in_target}")

    def count_datapoints_with_unk_in_target(self):
        num_datapoints_with_unk_in_target = 0
        for dp in tqdm(self, "Counting datapoints with <unk> among the target ids..."):
            target_ids = dp["target_ids"]

            if self.tokenizer.unk_token_id in target_ids:
                num_datapoints_with_unk_in_target += 1

        return num_datapoints_with_unk_in_target

    @staticmethod
    def _get_triplet_surface_form(triplet):
        return triplet["subject"]["surfaceform"], triplet["predicate"]["surfaceform"], triplet["object"]["surfaceform"]

    @staticmethod
    def _get_triplets_surface_form(triplets):
        return [IEGenericDataset._get_triplet_surface_form(triplet) for triplet in triplets]

    @staticmethod
    def get_triplets_surface_form(dp):
        return IEGenericDataset._get_triplets_surface_form(dp["triplets"])

    @staticmethod
    def _get_num_tokens(text, tokenizer):
        return len(tokenizer(text)["input_ids"])

    @staticmethod
    def _get_num_tokens_input(dp, tokenizer):
        num_tokens_dict = dp.get("num_tokens_dict", {})

        # if the number of tokens is already computed, return it
        if "text" in num_tokens_dict:
            return num_tokens_dict["text"]

        # otherwise, compute it and store it
        num_tokens_dict["text"] = IEGenericDataset._get_num_tokens(dp["text"], tokenizer)
        dp["num_tokens_dict"] = num_tokens_dict
        return num_tokens_dict["text"]

    @staticmethod
    def _get_linearized_target(dp, tokenizer, linearization_class):
        target_dict = dp.get("target_dict", {})

        if linearization_class.identifier not in target_dict:
            triplets = IEGenericDataset.get_triplets_surface_form(dp)
            target_text, target_ids = linearization_class.triplet_list_to_text(triplets, tokenizer)
            target_dict[linearization_class.identifier] = (target_text, target_ids)
            dp["target_dict"] = target_dict

        return target_dict[linearization_class.identifier]

    @staticmethod
    def _get_num_tokens_target(dp, tokenizer, linearization_class):
        num_tokens_dict = dp.get("num_tokens_dict", {})

        # if the number of tokens is already computed, return it
        if linearization_class.identifier in num_tokens_dict:
            return num_tokens_dict[linearization_class.identifier]

        # otherwise, compute it and store it
        target_text, target_ids = IEGenericDataset._get_linearized_target(dp, tokenizer, linearization_class)
        num_tokens_dict[linearization_class.identifier] = len(target_ids)
        dp["num_tokens_dict"] = num_tokens_dict
        return num_tokens_dict[linearization_class.identifier]

    def _are_num_tokens_within_bounds(self, obj):
        # Input tokens
        num_tokens_input = self._get_num_tokens_input(obj, self.tokenizer)

        if num_tokens_input > self.params["max_num_tokens_input"]:
            self.num_filtered_datapoints_input_tokens += 1
            return False

        # Target tokens
        num_tokens_target = self._get_num_tokens_target(obj, self.tokenizer, self.linearization_class_for_filtering)

        if num_tokens_target > self.params["max_num_tokens_target"]:
            self.num_filtered_datapoints_target_tokens += 1
            return False

        # Both within bounds
        return True

    def _load_data(self, load_dataset_params):
        self._read_constrained_world()

        log.info(
            f"Loading the data with: "
            f"constrained world `{self.params.get('constrained_world') if self.constrained_world else 'None'}` -- "
            f"filter_on_num_tokens `{load_dataset_params['filter_on_num_tokens']}` -- "
            f"linearization_class_for_filtering `{self.linearization_class_for_filtering.identifier}` -- "
            f"apply_ordering_heuristic `{load_dataset_params['apply_ordering_heuristic']}`"
        )

        path = os.path.join(load_dataset_params["data_dir"], f"{load_dataset_params['split']}.jsonl")
        if load_dataset_params["gzipped"]:
            path += ".gz"

        if load_dataset_params["gzipped"]:
            with gzip.open(path, "r") as f:
                num_datapoints = sum(1 for line in f)
            stream = gzip.open(path, "r")
        else:
            with open(path, "r") as f:
                num_datapoints = sum(1 for line in f)
            stream = open(path, "r")
        json_reader = jsonlines.Reader(stream)

        self.data = []
        for obj in tqdm(json_reader, total=num_datapoints, desc=f"Loading the data from: {path}"):
            obj["text"] = obj["text"].strip()
            if not self.constrained_world or self._include_datapoint(obj):
                # the datapoint is within the world of interest, or the world is not constrained
                if load_dataset_params.get("apply_ordering_heuristic", False):
                    # apply ordering heuristic
                    ordering_heuristic_utils._apply_ordering_heuristic_to_datapoint(obj)

                if not load_dataset_params["filter_on_num_tokens"] or self._are_num_tokens_within_bounds(obj):
                    # the datapoint's input and linearized target do not exceed the max number of tokens
                    # or filtering on the number of tokens is not applied
                    if (
                        load_dataset_params.get("include_ids", None)
                        and obj["id"] not in load_dataset_params["include_ids"]
                    ):
                        # loading of specific datapoints is requested, and this datapoint is not in the list
                        continue

                    self.data.append(obj)

            if self.params.get("debug", False) and len(self.data) >= self.params["debug_k"]:
                break

        stream.close()

        log.info(f"Loaded {len(self.data)} datapoints from {path}")
        if self.constrained_world:
            log.info(
                f"[Constrained world filtering] Filtered {self.num_filtered_datapoints_constrained_world} datapoints"
            )
        if load_dataset_params["filter_on_num_tokens"]:
            log.info(f"[# tokens filtering -- input] Filtered {self.num_filtered_datapoints_input_tokens} datapoints")
            log.info(f"[# tokens filtering -- target] Filtered {self.num_filtered_datapoints_target_tokens} datapoints")

    def _read_constrained_world(self):
        world_id = self.params.get("constrained_world")
        path_to_constrained_world_dir = self.params.get("path_to_constrained_world_dir")
        self.constrained_world = world_id is not None or path_to_constrained_world_dir is not None

        if not self.constrained_world:
            return

        self.entities_to_keep, self.relations_to_keep = read_constrained_world(
            constrained_world_id=self.params.get("constrained_world"),
            path_to_constrained_world_dir=self.params.get("path_to_constrained_world_dir"),
            constrained_worlds_dir=self.params.get("constrained_worlds_dir"),
        )

    def _include_datapoint(self, obj):
        if all(entity["uri"] in self.entities_to_keep for entity in obj["entities"]) and all(
            entity["uri"] in self.relations_to_keep for entity in obj["relations"]
        ):
            return True
        self.num_filtered_datapoints_constrained_world += 1
        return False

    @staticmethod
    def are_triplets_with_same_subject_consecutive(data):
        for idx, dp in enumerate(data):
            triplets = IEGenericDataset.get_triplets_surface_form(dp)
            if ordering_heuristic_utils._are_triplets_with_same_subjects_consecutive(triplets):
                continue
            log.info(
                "DP with:\n"
                "IDX:{}\n"
                "TEXT:{}\n"
                "Has triplets with non-consecutive subjects: {}".format(idx, dp["text"], triplets)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dp = self.data[idx]

        target_text, target_ids = self._get_linearized_target(dp, self.tokenizer, self.linearization_class)

        return {"id": dp["id"], "text": dp["text"], "target": target_text, "target_ids": target_ids}

    def compute_dataset_statistics(self, compute_frequency_dicts=False):
        entity_sets = []
        for dp in self.data:
            entity_sets.append(
                set(
                    [triplet["subject"]["surfaceform"] for triplet in dp["triplets"]]
                    + [triplet["object"]["surfaceform"] for triplet in dp["triplets"]]
                )
            )
        flat_entity_sets = [entity for entity_set in entity_sets for entity in entity_set]

        relation_sets = [set([triplet["predicate"]["surfaceform"] for triplet in dp["triplets"]]) for dp in self.data]
        flat_relation_sets = [rel for rel_set in relation_sets for rel in rel_set]

        if compute_frequency_dicts:
            self.ent2freq = Counter(flat_entity_sets)  # The number of datapoints in which the entity appears
            self.rel2freq = Counter(flat_relation_sets)  # The number of datapoints in which a relation occurs

        stats = {
            "num_datapoints": len(self.data),
            "num_triplets": sum([len(dp["triplets"]) for dp in self.data]),
            "num_unique_entities": len(set(flat_entity_sets)),
            "num_unique_relations": len(set(flat_relation_sets)),
        }

        log.info(f"Dataset statistics: {stats}")

    def get_dataset_id(self):
        return get_dataset_id(self.params)


class IEGenericOutputDataset(AbstractOutputDataset):
    def __init__(self, data=None, **params):
        super().__init__(params)
        linearization_class_id = params.get("linearization_class_id")

        self.linearization_class = utils.get_linearization_class(linearization_class_id)

        self.data = None

        if data is None:
            self._load_data()
        else:
            self._process_data_argument(data)

        self.compute_dataset_statistics()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _load_data(self):
        data = read_outputs(self.params["data_dir"])

        for dp in data:
            dp["target_triplets"] = self.get_text_triples(self.get_targets(dp))
            dp["prediction_triplets"] = self.get_text_triples(self.get_predictions(dp))

        self.data = data
        log.info(
            "[Output DS] Loaded the predictions for %d datapoints from %s", len(self.data), self.params["data_dir"]
        )

    def _process_data_argument(self, data):
        for dp in data:
            dp["target_triplets"] = self.get_text_triples(dp["target"])
            dp["prediction_triplets"] = self.get_text_triples(dp["prediction"])

        self.data = data
        log.info("[Output DS] Got %d datapoints from the `data` passed to the constructor", len(self.data))

    @staticmethod
    def _filter_triplets(triplets, relations_to_consider, entities_to_consider):
        filtered_triplets = set()
        for s, r, o in triplets:
            # ~~~ Filtering on relations ~~~
            if not (relations_to_consider is None or r in relations_to_consider):
                continue

            # ~~~ Filtering on entities ~~~
            if not (entities_to_consider is None or s in entities_to_consider):
                continue

            # ~~~ Adding the triplet if it passes the filters ~~~
            filtered_triplets.add((s, r, o))

        return filtered_triplets

    def compute_dataset_statistics(self):
        relations_p = [triplet[1] for dp in self.data for triplet in dp["prediction_triplets"]]
        relations_t = [triplet[1] for dp in self.data for triplet in dp["target_triplets"]]
        entities_p = [triplet[0] for dp in self.data for triplet in dp["prediction_triplets"]] + [
            triplet[2] for dp in self.data for triplet in dp["prediction_triplets"]
        ]
        entities_t = [triplet[0] for dp in self.data for triplet in dp["target_triplets"]] + [
            triplet[2] for dp in self.data for triplet in dp["target_triplets"]
        ]

        stats = {
            "num_datapoints": len(self.data),
            "num_triplets": {"P": len(relations_p), "T": len(relations_t)},
            "num_unique_entities": {"P": len(set(entities_p)), "T": len(set(entities_t))},
            "num_unique_relations": {"P": len(set(relations_p)), "T": len(set(relations_t))},
        }

        log.info(f"[Output DS] Dataset statistics: {stats}")

    def get_filtered_data(
        self,
        relations_to_consider: Union[Set, List, None],
        entities_to_consider: Union[Set, List, None],
        indices_to_consider: Union[Set, List, None] = None,
    ):
        if isinstance(relations_to_consider, list):
            relations_to_consider = set(relations_to_consider)
        if isinstance(entities_to_consider, list):
            entities_to_consider = set(entities_to_consider)
        if indices_to_consider is None:
            indices_to_consider = range(len(self.data))

        filtered_data = []
        for idx in indices_to_consider:
            dp = self.data[idx]
            prediction_triplets = self._filter_triplets(
                dp["prediction_triplets"], relations_to_consider, entities_to_consider
            )
            target_triplets = self._filter_triplets(dp["target_triplets"], relations_to_consider, entities_to_consider)

            assert indices_to_consider is None or (len(prediction_triplets) > 0 or len(target_triplets) > 0)

            if len(dp["target_triplets"]) > 0 or len(dp["prediction_triplets"]) > 0:
                filtered_data.append({"prediction_triplets": prediction_triplets, "target_triplets": target_triplets})

        return filtered_data

    def get_text_triples(self, text, verbose=False, return_set=True) -> Union[Set[tuple], List[tuple]]:
        return self.linearization_class.text_to_triplet_list(text=text, verbose=verbose, return_set=return_set)


class IEGenericDataModule(AbstractPLDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    # params = {"data_dir": "/home/martin/SynthIE_main/logs/inference/runs/inf_sc_fully_synthetic_gcp_datamodule-rebel_world-genie_t5_tokenizeable_split-test_small_constraint-free_lp-0.6/2023-01-27_17-57-07/predictions",
    #           "linearization_class_id": "subject_collapsed",
    #           "seed": 123
    #           }
    # output_dataset = IEGenericOutputDataset(**params)

    import argparse

    from transformers import T5Tokenizer
    from pprint import pprint

    data_dir = "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=data_dir)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--linearization_class_id", type=str, required=True)
    parser.add_argument("--linearization_class_id_for_filtering", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--constrained_world", type=str, default="genie")
    parser.add_argument("--path_to_constrained_world_dir", type=str, default=None)
    parser.add_argument("--constrained_worlds_dir", type=str, default=os.path.join(data_dir, "constrained_worlds"))
    parser.add_argument("--filter_on_num_tokens", action="store_true")
    parser.add_argument("--apply_ordering_heuristic", action="store_true")
    parser.add_argument("--max_num_tokens_input", type=int, default=256)
    parser.add_argument("--max_num_tokens_target", type=int, default=256)
    args = parser.parse_args()

    if args.constrained_world.lower() == "none" or args.constrained_world.lower() == "null":
        args.constrained_world = None

    args.load_dataset_params = {
        "data_dir": os.path.join(args.data_dir, args.dataset_name),
        "split": args.split,
        "filter_on_num_tokens": args.filter_on_num_tokens,
        "apply_ordering_heuristic": args.apply_ordering_heuristic,
    }
    dataset_name = args.dataset_name
    split = args.split

    # import argparse
    #
    # from transformers import T5Tokenizer
    # from pprint import pprint
    #
    # data_dir = "data"
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default=data_dir)
    # parser.add_argument("--dataset_name", type=str, required=True)
    # parser.add_argument("--split", type=str, required=True)
    # parser.add_argument("--linearization_class_id", type=str, required=True)
    # parser.add_argument("--linearization_class_id_for_filtering", type=str, required=True)
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--debug_k", type=int, default=12)
    # parser.add_argument("--seed", type=int, default=123)
    # parser.add_argument("--constrained_world", type=str, default="genie")
    # parser.add_argument("--path_to_constrained_world_dir", type=str, default=None)
    # parser.add_argument("--constrained_worlds_dir", type=str, default=os.path.join(data_dir, "constrained_worlds"))
    # parser.add_argument("--filter_on_num_tokens", action="store_true")
    # parser.add_argument("--apply_ordering_heuristic", action="store_true")
    # parser.add_argument("--max_num_tokens_input", type=int, default=256)
    # parser.add_argument("--max_num_tokens_target", type=int, default=256)
    # args = parser.parse_args()
    #
    # if args.constrained_world.lower() == "none" or args.constrained_world.lower() == "null":
    #     args.constrained_world = None
    #
    # args.load_dataset_params = {
    #     "data_dir": os.path.join(args.data_dir, args.dataset_name),
    #     "split": args.split,
    #     "filter_on_num_tokens": args.filter_on_num_tokens,
    #     "apply_ordering_heuristic": args.apply_ordering_heuristic,
    # }
    # dataset_name = args.dataset_name
    # split = args.split
    #
    # del args.data_dir
    # del args.dataset_name
    # del args.split
    # del args.filter_on_num_tokens
    # del args.apply_ordering_heuristic
    #
    # pprint(vars(args))
    #
    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    # args.tokenizer = tokenizer
    #
    # dataset = IEGenericDataset(**vars(args))
    #
    # # data = dataset.data
    # # print("Number of datapoints:", len(data))
    # # print("Number of triplets:", sum([len(dp['triplets']) for dp in data]))
    # #
    # # subjects = set([triplet['subject']['surfaceform'] for dp in data for triplet in dp['triplets']])
    # # objects = set([triplet['object']['surfaceform'] for dp in data for triplet in dp['triplets']])
    # # relations = set([triplet['predicate']['surfaceform'] for dp in data for triplet in dp['triplets']])
    # #
    # # print("Number of unique subjects:", len(subjects))
    # # print("Number of unique objects:", len(objects))
    # # print("Number of unique entities:", len(subjects.union(objects)))
    # # print("Number of unique relations:", len(relations))
    #
    # pprint(dataset[0])
    # pprint(dataset.data[0])
    # pprint(dataset.data[0]["text"])
    # pprint(IEGenericDataset.get_triplets_surface_form(dataset.data[0]))

    # dataset.are_triplets_with_same_subject_consecutive(dataset.data)
