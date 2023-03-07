from process_webnlg import get_relations_entities_webnlg
from process_rebel import get_relations_entities_rebel, map_qids_to_names
import os
import argparse
import pickle


def to_dict(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    data = dict(data)
    with open(file, "wb") as f:
        pickle.dump(data, f)


def get_relations_entities(
    rebel_folder,
    webnlg_folder,
    mapping_file,
    mapping_file_rel,
    entity_file,
    relation_file,
    english_dict_path=None,
    convert_to_dict=True,
):
    rebel_entities = None
    rebel_relations = None
    print("Processing Rebel")
    for file in os.listdir(rebel_folder):
        if file.endswith(".jsonl"):
            print("Processing " + str(file))
            path = os.path.join(rebel_folder, file)
            rebel_entities, rebel_relations = get_relations_entities_rebel(
                path, rebel_entities, rebel_relations, entity_file, relation_file
            )
            print("Number of relations: " + str(len(rebel_relations)))
            print("Number of entities: " + str(len(rebel_entities)))

    map_qids_to_names(
        rebel_entities, rebel_relations, entity_file, relation_file, mapping_file, mapping_file_rel, english_dict_path
    )

    print("Processing WebNLG")
    for file in os.listdir(webnlg_folder):
        if file.endswith(".xml"):
            print("Processing " + str(file))
            path = os.path.join(webnlg_folder, file)
            get_relations_entities_webnlg(path, entity_file, relation_file, entity_file, relation_file)

    if convert_to_dict:
        to_dict(entity_file)
        to_dict(relation_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebel", type=str, help="Path to Rebel dataset")
    parser.add_argument("--webnlg", type=str, help="Path to WebNLG dataset")
    parser.add_argument("--mapping_ent", type=str, help="Path to the file mapping entities")
    parser.add_argument("--mapping_rel", type=str, help="Path to the file mapping relations")
    parser.add_argument("--entity_file", type=str, help="Path to save entities")
    parser.add_argument("--relation_file", type=str, help="Path to save relations")
    parser.add_argument("--english", type=str, help="Path to save english mapping")
    args, unknown = parser.parse_known_args()
    get_relations_entities(
        args.rebel, args.webnlg, args.mapping_ent, args.mapping_rel, args.entity_file, args.relation_file, args.english
    )
