from process_webnlg import *
from process_rebel import *
import argparse
from datetime import date
import dateutil.parser as parser

np.random.seed(0)

relations_to_check = ["P31", "P361", "P17", "P625", "P2044", "P279", "P2579", "P30", "P2046", "P36", "P1082"]


def process_webnlg(path, path_to_save, relations_path=None, filter=False):
    index_ = 0
    if not (relations_path is None):
        relations_dict = read_relations_dict(relations_path)
    with jsonlines.open(path_to_save, "w") as f:
        tree = ET.parse(path)
        benchmark = tree.getroot()
        entries = benchmark[0]
        for entry in entries:
            obj = {}
            triplets = []
            lexes = []
            for lex in entry.iter("lex"):
                lex_text = lex.text.strip()
                lex_quality = lex.get("comment")
                lexes.append({"text": lex_text, "quality": lex_quality})

            lex = choose_lex(lexes)
            for triplet in entry.iter("modifiedtripleset"):
                mtriplet = triplet[0]
                mtriplet_text = mtriplet.text.strip()
                mcode = mtriplet[0].text.strip()
                ids = mcode.split(" | ")
                sforms = mtriplet_text.split(" | ")
                if (ids[1] in relations_to_check) or (not filter):
                    if not ids[2].startswith("Q"):
                        sforms[2], relation_type = extract_literal(lex["text"], sforms[2], ids[1])
                        if len(sforms[2]) == 0:
                            continue
                    else:
                        relation_type = "entity"
                    if not (relations_path is None):
                        if ids[1] in relations_dict:
                            relations_dict[ids[1]].add(relation_type)
                        else:
                            relations_dict[ids[1]] = set()
                            relations_dict[ids[1]].add(relation_type)
                    triplet = {}
                    subject_ent = {"uri": ids[0], "surfaceform": sforms[0]}
                    object_ent = {"uri": ids[2], "surfaceform": sforms[2]}
                    relation = {"uri": ids[1], "surfaceform": sforms[1]}
                    triplet["subject"] = subject_ent
                    triplet["predicate"] = relation
                    triplet["object"] = object_ent
                    triplets.append(triplet)
            if len(triplets) > 0:
                obj["triplets"] = triplets
                obj["text"] = lex
                obj["id"] = index_
                obj = map_names(obj)
                if len(obj) > 0:
                    index_ += 1
                    f.write(obj)

    if not (relations_path is None):
        save_relations_dict(relations_dict, relations_path)


def process_rebel(path, path_to_save, relations_path=None, filter=False):
    index_ = 0
    if not (relations_path is None):
        relations_dict = read_relations_dict(relations_path)
    with jsonlines.open(path_to_save, "w") as f:
        with jsonlines.open(path) as jsonl_file:
            for idx, obj_ in enumerate(jsonl_file):
                obj_list = process_rebel_row(obj_)
                for obj in obj_list:
                    triplets_to_save = []
                    triplets = obj["triplets"]
                    for triplet in triplets:
                        if (triplet["predicate"]["uri"] in relations_to_check) or (not filter):
                            if not triplet["object"]["uri"].startswith("Q"):
                                try:
                                    datatype = triplet["object"]["uri"].split("#")[1]
                                except:
                                    continue
                                if datatype == "decimal":
                                    unit = extract_unit(obj["text"], triplet["object"])
                                    sform = triplet["object"]["surfaceform"]
                                    if sform.startswith("+"):
                                        sform = sform[1:]
                                    if len(unit) > 0:
                                        sform = sform + " " + unit
                                    triplet["object"]["surfaceform"] = sform
                                    if not (relations_path is None):
                                        if triplet['predicate']['uri'] in relations_dict:
                                            relations_dict[triplet['predicate']['uri']].add("quantity")
                                        else:
                                            relations_dict[triplet['predicate']['uri']] = set()
                                            relations_dict[triplet['predicate']['uri']].add("quantity")
                                elif datatype == "dateTime":
                                    ids = triplet["object"]["surfaceform"].split()
                                    try:
                                        if len(ids) == 1:
                                            # year
                                            date_obj = date(int(triplet["object"]["surfaceform"]), 1, 1)
                                            value = date_obj.strftime("%Y")
                                            triplet["object"]["surfaceform"] = value
                                        elif len(ids) == 2:
                                            # month + year
                                            date_obj = parser.parse(triplet["object"]["surfaceform"])
                                            value = date_obj.strftime("%B %Y")
                                            triplet["object"]["surfaceform"] = value
                                        else:
                                            date_obj = parser.parse(triplet["object"]["surfaceform"])
                                            value = date_obj.strftime("%d %B %Y")
                                            triplet["object"]["surfaceform"] = value
                                    except:
                                        continue
                                    if not (relations_path is None):
                                        if triplet['predicate']['uri'] in relations_dict:
                                            relations_dict[triplet['predicate']['uri']].add("date")
                                        else:
                                            relations_dict[triplet['predicate']['uri']] = set()
                                            relations_dict[triplet['predicate']['uri']].add("date")
                            else:
                                if not (relations_path is None):
                                    if triplet['predicate']['uri'] in relations_dict:
                                        relations_dict[triplet['predicate']['uri']].add("entity")
                                    else:
                                        relations_dict[triplet['predicate']['uri']] = set()
                                        relations_dict[triplet['predicate']['uri']].add("entity")
                            triplets_to_save.append(triplet)
                    if len(triplets_to_save) > 0:
                        obj["triplets"] = triplets_to_save
                        obj["id"] = index_
                        obj = map_names(obj)
                        if len(obj) > 0:
                            index_ += 1
                            f.write(obj)

    if not (relations_path is None):
        save_relations_dict(relations_dict, relations_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the current data")
    parser.add_argument("--save_path", type=str, help="Path where to save data")
    parser.add_argument("--dataset", type=str, help="Type of dataset to process")
    parser.add_argument("--filter", action="store_true", help="Whether to filter out the dataset")
    parser.add_argument("--relations_path", type=str, help="Path to the relation-rule file")
    args, unknown = parser.parse_known_args()

    if args.dataset == "rebel":
        process_rebel(args.path, args.save_path, args.relations_path, args.filter)
    elif args.dataset == "webnlg":
        process_webnlg(args.path, args.save_path, args.relations_path, args.filter)
    else:
        raise NotImplementedError
