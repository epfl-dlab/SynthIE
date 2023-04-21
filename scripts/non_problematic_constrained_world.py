import argparse
import os

from transformers import AutoTokenizer
from tqdm import tqdm

import src.utils as utils
from src.datamodules import IEGenericDataset
from src.utils.general_helpers import write_gzipped_jsonlines, write_jsonlines
import src.utils.constrained_generation_utils as cg_utils


def is_problematic_name(name):
    ids = encode_func(name)
    decoded_name = tokenizer.decode(ids[:-1])
    return name != decoded_name


def get_problematic_names(names, encode_func):
    p_name2ids = {}
    for name in tqdm(names):
        ids = encode_func(name)
        decoded_name = tokenizer.decode(ids[:-1])
        if name != decoded_name:
            p_name2ids[name] = ids
    return p_name2ids


if __name__ == "__main__":
    constrained_worlds_dir = "data/constrained_worlds"
    path_to_relid2name_mapping = "data/id2name_mappings/relation_mapping.jsonl"
    path_to_entid2name_mapping = "data/id2name_mappings/entity_mapping.jsonl"
    tokenizer_full_name = "google/flan-t5-base"
    tokenizer_short_name = "t5"

    parser = argparse.ArgumentParser()
    parser.add_argument("--constrained_worlds_dir", type=str, default=constrained_worlds_dir)
    parser.add_argument("--path_to_relid2name_mapping", type=str, default=path_to_relid2name_mapping)
    parser.add_argument("--path_to_entid2name_mapping", type=str, default=path_to_entid2name_mapping)
    parser.add_argument("--constrained_world_id", type=str, required=True)
    parser.add_argument("--tokenizer_full_name", type=str, default=tokenizer_full_name)
    parser.add_argument("--tokenizer_short_name", type=str, default=tokenizer_short_name)


    parser.add_argument("--linearization_class_id", type=str, default="subject_collapsed")

    # parser.add_argument("--input_file_is_gzipped", action="store_true")
    args = parser.parse_args()

    linearization_class = utils.get_linearization_class(args.linearization_class_id)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_full_name)
    encode_func = lambda x: tokenizer(x)["input_ids"]

    entity_ids, relation_ids = cg_utils.read_constrained_world(
        constrained_world_id=args.constrained_world_id,
        path_to_constrained_world_dir=None,
        constrained_worlds_dir=args.constrained_worlds_dir,
    )

    relation_names = cg_utils.get_names_for_ids(
        relation_ids, path_to_relid2name_mapping, keep_spaces=linearization_class.keep_spaces_relations
    )
    entity_names = cg_utils.get_names_for_ids(
        entity_ids, path_to_entid2name_mapping, keep_spaces=linearization_class.keep_spaces_entities
    )

    # Get the problematic names
    p_name2token_ids = get_problematic_names(entity_names, encode_func)
    problematic_names = set(p_name2token_ids.keys())
    assert len(problematic_names) == len(p_name2token_ids)

    # Get the non-problematic names
    non_problematic_names = set(entity_names) - problematic_names

    # Get the Wikidata IDs for the non-problematic names
    non_problematic_entity_ids = cg_utils.get_ids_for_names(non_problematic_names, path_to_entid2name_mapping)

    # Create a new constrained world with the non-problematic names only
    output_constrained_world_id = f"{args.constrained_world_id}_{args.tokenizer_short_name}_tokenizeable"
    print("Saving the new constrained world to", os.path.join(args.constrained_worlds_dir, output_constrained_world_id))
    cg_utils.write_constrained_world(
        args.constrained_worlds_dir, output_constrained_world_id, non_problematic_entity_ids, relation_ids
    )
