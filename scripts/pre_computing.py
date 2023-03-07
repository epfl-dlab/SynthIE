import argparse
import os

from transformers import T5Tokenizer
from tqdm import tqdm

import src.utils as utils
from src.datamodules import IEGenericDataset
from src.utils.general_helpers import write_gzipped_jsonlines, write_jsonlines

if __name__ == "__main__":
    data_dir = "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=data_dir)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--input_file_is_gzipped", action="store_true")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--apply_ordering_heuristic", action="store_true")
    parser.add_argument("--lc_ids", nargs="+", type=str, required=True)
    parser.add_argument("--output_data_dir", type=str, required=True)
    parser.add_argument("--include_target_dict", action="store_true")
    parser.add_argument("--save_zipped", action="store_true")
    parser.add_argument("--save_non_zipped", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_k", type=int, default=12)
    args = parser.parse_args()

    args.constrained_world = None
    args.load_dataset_params = {
        "data_dir": os.path.join(args.data_dir, args.dataset_name),
        "split": args.split,
        "filter_on_num_tokens": False,
        "apply_ordering_heuristic": args.apply_ordering_heuristic,
        "gzipped": args.input_file_is_gzipped,
    }
    # ~~~ Required by the constructor but will not be used ~~~
    args.seed = 123
    args.linearization_class_id = args.lc_ids[0]
    args.linearization_class_id_for_filtering = args.lc_ids[0]
    args.path_to_constrained_world_dir = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    output_data_dir = args.output_data_dir
    dataset_name = args.dataset_name
    split = args.split

    save_zipped = args.save_zipped
    save_non_zipped = args.save_non_zipped

    del args.output_data_dir
    del args.data_dir
    del args.dataset_name
    del args.split
    del args.input_file_is_gzipped
    del args.apply_ordering_heuristic
    del args.save_zipped
    del args.save_non_zipped

    linearization_classes_to_consider = args.lc_ids
    include_target_dict = args.include_target_dict
    del args.lc_ids
    del args.include_target_dict

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    args.tokenizer = tokenizer

    path = os.path.join(args.load_dataset_params["data_dir"], f"{args.load_dataset_params['split']}.jsonl")
    num_datapoints = sum(1 for line in open(path))
    dataset = IEGenericDataset(**vars(args))

    for dp in tqdm(dataset.data, total=num_datapoints, desc=f"Running pre-computation"):
        dataset._get_num_tokens_input(dp, tokenizer)

        for linearization_class_id in linearization_classes_to_consider:
            dataset._get_num_tokens_target(dp, tokenizer, utils.get_linearization_class(linearization_class_id))

        if not include_target_dict:
            del dp["target_dict"]

    output_data_dir = os.path.join(output_data_dir, dataset_name)
    os.makedirs(output_data_dir, exist_ok=True)

    output_file_name = split
    if args.load_dataset_params["apply_ordering_heuristic"]:
        output_file_name += "_ordered"

    output_file_name += ".jsonl"
    if save_non_zipped:
        output_file_path = os.path.join(output_data_dir, output_file_name)
        print("Writing (non-zipped) data to:", output_file_path)
        write_jsonlines(output_file_path, dataset.data, mode="w")

    output_file_name += ".gz"
    if save_zipped:
        output_file_path = os.path.join(output_data_dir, output_file_name)
        print("Writing (zipped) data to:", output_file_path)
        write_gzipped_jsonlines(os.path.join(output_data_dir, output_file_name), dataset.data, mode="w")
