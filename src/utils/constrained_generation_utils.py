import jsonlines
import json
import os

from pathlib import Path
from src.utils.linearization_utils import LinearizationType, remove_accents_from_str


def get_trie_from_strings(
    string_iterable,
    remove_leading_bos=False,
    output_folder_path=None,
    trie_name=None,
    tokenizer=None,
):
    from src.constrained_generation import Trie

    assert (output_folder_path is None and trie_name is None) or (
        output_folder_path is not None and trie_name is not None
    )
    from tqdm import tqdm

    if tokenizer is None:
        from transformers import T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    if remove_leading_bos:
        leading_bos = lambda x: x[1:]
    else:
        leading_bos = lambda x: x

    encode_func = lambda x: leading_bos(tokenizer(x)["input_ids"])
    trie = Trie([encode_func(uniq_name) for uniq_name in tqdm(sorted(string_iterable), desc="Building trie")])

    if output_folder_path is not None:
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        trie.dump(output_folder_path=output_folder_path, file_name=trie_name, string_iterable=string_iterable)

    return trie


def read_constrained_world(constrained_world_id=None, path_to_constrained_world_dir=None, constrained_worlds_dir=None):
    assert {constrained_world_id is None, path_to_constrained_world_dir is None} == {
        True,
        False,
    }, "Either specify a `constrained_world` or a path_to_constrained_world_dir, not both."

    if path_to_constrained_world_dir is None:
        path_to_constrained_world_dir = os.path.join(constrained_worlds_dir, constrained_world_id)

    with open(os.path.join(path_to_constrained_world_dir, "entities.json")) as json_file:
        entities = set(json.load(json_file))

    with open(os.path.join(path_to_constrained_world_dir, "relations.json")) as json_file:
        relations = set(json.load(json_file))

    return entities, relations


def write_constrained_world(path_to_constrained_worlds_dir, constrained_world_id, entity_ids, relation_ids):
    path_to_constrained_world_dir = os.path.join(path_to_constrained_worlds_dir, constrained_world_id)
    os.makedirs(path_to_constrained_world_dir, exist_ok=True)

    with open(os.path.join(path_to_constrained_world_dir, "relations.json"), "w") as json_file:
        if isinstance(relation_ids, set):
            relation_ids = list(relation_ids)
        json.dump(list(relation_ids), json_file)

    with open(os.path.join(path_to_constrained_world_dir, "entities.json"), "w") as json_file:
        if isinstance(entity_ids, set):
            entity_ids = list(entity_ids)
        json.dump(entity_ids, json_file)


def get_names_for_ids(ids, path_to_id2name_mapping, keep_spaces, remove_accents=False):
    with jsonlines.open(path_to_id2name_mapping) as reader:
        id2name_mapping = {obj["id"]: obj["en_label"] for obj in reader}

    names = [
        LinearizationType.normalize_spaces(id2name_mapping[_id], keep_spaces=keep_spaces)
        for _id in ids
        if _id in id2name_mapping
    ]

    if remove_accents:
        names = [remove_accents_from_str(name) for name in names]

    return names


def get_ids_for_names(names, path_to_name2id_mapping):
    with jsonlines.open(path_to_name2id_mapping) as reader:
        name2id_mapping = {obj["en_label"]: obj["id"] for obj in reader}

    return [name2id_mapping[name] for name in names if name in name2id_mapping]


def encode(text, tokenizer, keep_eos: bool):
    if keep_eos:
        raise NotImplementedError
        # return tokenizer.encode(text)

    return tokenizer.encode(text, add_special_tokens=False)
