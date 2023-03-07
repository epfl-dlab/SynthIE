import json
import numpy as np
import os
import jsonlines
from tqdm import tqdm

import src.utils as utils
from src.utils.general_helpers import read_jsonlines, write_jsonlines

if __name__ == "__main__":
    log = utils.get_pylogger(__name__, stdout=True)
else:
    log = utils.get_pylogger(__name__)


def get_data(data_dir, split):
    path_to_file = os.path.join(data_dir, split + ".jsonl")
    data = read_jsonlines(path_to_file)
    return data


def write_data(data, data_dir, split):
    path_to_file = os.path.join(data_dir, split + ".jsonl")
    write_jsonlines(path_to_file, data, mode="w")


def get_start_index_parts(text, sf_parts):
    max_subseq_len = len(sf_parts)

    for curr_len in range(max_subseq_len, 0, -1):
        for i in range(max_subseq_len - curr_len + 1):
            subseq = " ".join(sf_parts[i : i + curr_len])
            try:
                return text.index(subseq)
            except ValueError:
                pass

    return None


def _get_start_index(dp, sf):
    for entity in dp["entities"]:
        if entity["surfaceform"] == sf:
            idx = entity["mention_start_index"]

    if idx is None:
        return float("inf")

    return idx


def _apply_ordering_heuristic_to_datapoint(dp):
    input_text = dp["text"].lower()
    entities = dp["entities"]
    entity_surfaceforms_parts = [entity["surfaceform"].lower().split("_") for entity in entities]
    mention_start_index = [get_start_index_parts(input_text, sf_parts) for sf_parts in entity_surfaceforms_parts]
    for idx, entity in zip(mention_start_index, entities):
        entity["mention_start_index"] = idx

    # apply_ordering
    sub_obj_sf_pairs = [
        (triplet["subject"]["surfaceform"], triplet["object"]["surfaceform"]) for triplet in dp["triplets"]
    ]
    sub_obj_start_idx_pairs = np.array(
        [(_get_start_index(dp, sub_sf), _get_start_index(dp, obj_sf)) for sub_sf, obj_sf in sub_obj_sf_pairs]
    ).T
    sub_ent_name = [triplet["subject"]["surfaceform"] for triplet in dp["triplets"]]
    ordered_indices = np.lexsort((sub_obj_start_idx_pairs[1, :], sub_ent_name, sub_obj_start_idx_pairs[0, :]))
    dp["triplets"] = [dp["triplets"][idx] for idx in ordered_indices]


def _apply_ordering_heuristic_to_data(data, verbose, num_examples_to_show):
    for dp in tqdm(data, "Locating the mention start indices."):
        _apply_ordering_heuristic_to_datapoint(dp)

    # report on non-resolved entities
    if verbose:
        _apply_ordering_heuristics_info(data, num_examples_to_show=num_examples_to_show)


def _apply_ordering_heuristics_info(data, num_examples_to_show):
    datapoints_affected = []
    num_triplets_affected = 0
    for dp in data:
        nan_surface_forms = set()

        for entity in dp["entities"]:
            if entity["mention_start_index"] is None:
                nan_surface_forms.add(entity["surfaceform"])

        if len(nan_surface_forms) == 0:
            continue

        datapoints_affected.append(dp)

        if num_examples_to_show > 0:
            log.info("------------------")
            log.info("Input text:", dp["text"])
            log.info("Not resolved entities:", nan_surface_forms)

        for triplet in dp["triplets"]:
            if (
                triplet["subject"]["surfaceform"] in nan_surface_forms
                or triplet["object"]["surfaceform"] in nan_surface_forms
            ):
                num_triplets_affected += 1
                if num_examples_to_show > 0:
                    log.info(triplet)

        num_examples_to_show -= 1

    # get all mention_start_indices
    mention_start_indices = np.array([entity["mention_start_index"] for dp in data for entity in dp["entities"]])
    # get number and portion of NaNs
    num_nans = np.sum(mention_start_indices == None)
    portion_nans = num_nans / len(mention_start_indices)

    # get total number of triplets
    num_total_triplets = 0
    for dp in data:
        num_total_triplets += len(dp["triplets"])

    log.info(
        f"[Ordering heuristic] Number of affected: entities ({num_nans} -- {portion_nans:.2%}), "
        f"triplets ({num_triplets_affected} -- {num_triplets_affected / num_total_triplets:.2%}), "
        f"datapoints ({len(datapoints_affected)} -- {len(datapoints_affected) / len(data):.2%})"
    )


def apply_ordering_heuristic(data_dir, split, output_split, verbose, num_examples_to_show, output_dir=None):
    if output_dir is None:
        output_dir = data_dir

    log.info(f"Applying ordering heuristic to the split `{split}` in `{data_dir}`...")
    data = get_data(data_dir, split)

    # process the data
    _apply_ordering_heuristic_to_data(data, verbose, num_examples_to_show)

    if output_split is not None:
        write_data(data, output_dir, output_split)

    return data


def _are_triplets_with_same_subjects_consecutive(triplets):
    past_subjects = set()
    curr_subject = None

    for triplet in triplets:
        if curr_subject is None:
            curr_subject = triplet[0]

        if triplet[0] != curr_subject:
            if triplet[0] in past_subjects:
                # the new subject has already been seen before
                return False
            past_subjects.add(triplet[0])
            curr_subject = triplet[0]

    return True


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    data_dir = "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=data_dir)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_split", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--num_examples_to_show", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset_name)
    del args.dataset_name

    pprint(vars(args))
    apply_ordering_heuristic(**vars(args))
