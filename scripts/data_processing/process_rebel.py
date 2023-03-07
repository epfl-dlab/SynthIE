import jsonlines
import pickle
import pandas as pd
from utils import *


def map_qids_to_names(
    ent_qids, rel_qids, entity_file, relation_file, mapping_file, mapping_file_rel, save_english_dict=None
):
    mapping_df = pd.read_parquet(mapping_file)
    mapping_df_ent = mapping_df[mapping_df["wiki_db"] == "enwiki"]
    mapping_df_ent = mapping_df_ent[["item_id", "page_title"]]
    if save_english_dict is not None:
        english_dict = dict(zip(mapping_df_ent.item_id, mapping_df_ent.page_title))
        with open(save_english_dict, "wb") as f:
            pickle.dump(english_dict, f)
    mapping_df_ent = mapping_df_ent[mapping_df_ent["item_id"].isin(ent_qids)]

    entities = set(map(tuple, mapping_df_ent.to_numpy()))
    print("Number of entities after mapping: " + str(len(entities)))

    relations = set()
    with jsonlines.open(mapping_file_rel) as f:
        for obj in f:
            pid = obj["wikidata_id"]
            if pid in rel_qids:
                relations.add((pid, obj["information"]["en_title"]))
    print("Number of relations after mapping: " + str(len(relations)))
    with open(entity_file, "wb") as f:
        pickle.dump(entities, f)
    with open(relation_file, "wb") as f:
        pickle.dump(relations, f)


def get_relations_entities_rebel(
    path, rebel_entities=None, rebel_relations=None, entity_file=None, relation_file=None, save=True
):
    if rebel_entities is None:
        rebel_entities = set()

    if rebel_relations is None:
        rebel_relations = set()

    with jsonlines.open(path) as f:
        for obj in f:
            for triplet in obj["triples"]:
                subject_id = triplet["subject"]["uri"]
                predicate_id = triplet["predicate"]["uri"]
                object_id = triplet["object"]["uri"]
                assert predicate_id.startswith("P")
                if subject_id.startswith("Q"):
                    rebel_entities.add(subject_id)
                if object_id.startswith("Q"):
                    rebel_entities.add(object_id)
                rebel_relations.add(predicate_id)
        if save:
            with open(entity_file, "wb") as f:
                pickle.dump(rebel_entities, f)
            with open(relation_file, "wb") as f:
                pickle.dump(rebel_relations, f)
        return rebel_entities, rebel_relations


def extract_unit(text, entity):
    try:
        boundaries = entity["boundaries"]
        words = text[boundaries[1] :].split()
        unit = words[0]
        if unit[0:3] == "km²":
            unit = "square kilometre"
        else:
            unit = "".join(filter(str.isalpha, unit))
            if unit == "square" or unit == "sq":
                unit = "square " + process_word(words[1])
            else:
                unit = process_word(unit)
        return unit
    except:
        return ""
