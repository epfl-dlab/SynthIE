import os.path
import re
import json
import pickle

prefix_map = {"k": "kilo", "m": "mili", "c": "centi", "d": "deci", "G": "giga", "M": "mega"}
unit_map = {
    "m": "metre",
    "s": "second",
    "g": "gram",
    "meter": "metre",
    "meterhigh": "metre",
    "V": "volt",
    "W": "watt",
    "h": "hour",
    "pc": "parsec",
    "Hz": "hertz",
}
unit_path = "/dlabdata1/josifosk/SynthIE_main/data/mappings/unitConversionConfig.json"
path_to_entity_mapping = "/dlabdata1/josifosk/SynthIE_main/data/mappings/english_mapping.pickle"
path_to_relation_mapping = "/dlabdata1/josifosk/SynthIE_main/data/mappings/relations.pickle"

def extract_unit_dict(unit_path):
    # process the JSON file
    unit_dict = {}
    with open(unit_path, "r") as f:
        json_dict = json.load(f)
    for key, d in json_dict.items():
        unit_dict[key] = d["label"]
    return unit_dict


def read_mappings(path_to_entity_mapping, path_to_relation_mapping, unit_path):
    with open(path_to_entity_mapping, "rb") as f:
        en_map = pickle.load(f)

    with open(path_to_relation_mapping, "rb") as f:
        rel_map = pickle.load(f)

    unit_dict = extract_unit_dict(unit_path)
    return en_map, rel_map, unit_dict


en_map, rel_map, unit_dict = read_mappings(path_to_entity_mapping, path_to_relation_mapping, unit_path)
unit_list = list(unit_dict.values())


def process_word(word):
    if word in unit_list:
        return word

        # plurals
    if word.endswith("es") and word[:-2] in unit_list:
        return word[:-2]
    if word.endswith("s") and word[:-1] in unit_list:
        return word[:-1]
    if len(word) >= 2:
        if word[0] in prefix_map:
            if word[1:] in unit_map:
                unit = prefix_map[word[0]] + unit_map[word[1:]]
            elif word[1:] in unit_list:
                unit = prefix_map[word[0]] + word[1:]
            else:
                return ""
            return unit
    elif word in unit_map:
        return unit_map[word]
    return ""


def extract_code_and_substring(x):
    return x["uri"], x["surfaceform"]


def extract_code_substring_and_boundaries(x):
    uri, surfaceform = extract_code_and_substring(x)
    boundaries = x["boundaries"]
    return uri, surfaceform, boundaries


def process_rebel_row(article):
    processed_data = []
    prev_len = 0
    if len(article["triples"]) == 0:
        return processed_data

    count = 0
    for text_paragraph in article["text"].split("\n"):
        if len(text_paragraph) == 0:
            continue

        sentences = re.split(r"(?<=[.])\s", text_paragraph)  # split on dot
        text = ""
        for sentence in sentences:
            text += sentence + " "
            if any(
                [
                    entity["boundaries"][0] < len(text) + prev_len < entity["boundaries"][1]
                    for entity in article["entities"]
                ]
            ):
                continue

            entities = sorted(
                [
                    entity
                    for entity in article["entities"]
                    if prev_len < entity["boundaries"][1] <= len(text) + prev_len
                ],
                key=lambda tup: tup["boundaries"][0],
            )

            code_triplets = []
            code_triplets_set = set()
            substring_triplets = []
            triplet_entity_boundaries = []

            entities_p = []
            triplets_p = []
            relations_p = []

            for int_ent, entity in enumerate(entities):
                triplets = sorted(
                    [
                        triplet
                        for triplet in article["triples"]
                        if triplet["subject"] == entity
                        and prev_len < triplet["subject"]["boundaries"][1] <= len(text) + prev_len
                        and prev_len < triplet["object"]["boundaries"][1] <= len(text) + prev_len
                    ],
                    key=lambda tup: tup["object"]["boundaries"][0],
                )

                if len(triplets) == 0:
                    continue

                for triplet in triplets:
                    s_code, s_substring, s_boundaries = extract_code_substring_and_boundaries(triplet["subject"])
                    p_code, p_name = extract_code_and_substring(triplet["predicate"])
                    o_code, o_substring, o_boundaries = extract_code_substring_and_boundaries(triplet["object"])

                    code_triplet = (s_code, p_code, o_code)
                    substring_triplet = (s_substring, p_name, o_substring)

                    if code_triplet not in code_triplets_set:
                        code_triplets_set.add(code_triplet)
                        code_triplets.append(code_triplet)
                        substring_triplets.append(substring_triplet)
                        triplet_entity_boundaries.append((s_boundaries, o_boundaries))

            for code_triplet, substring_triplet, boundaries in zip(
                code_triplets, substring_triplets, triplet_entity_boundaries
            ):
                sub_ent = {"uri": code_triplet[0], "surfaceform": substring_triplet[0], "boundaries": boundaries[0]}
                obj_ent = {"uri": code_triplet[2], "surfaceform": substring_triplet[2], "boundaries": boundaries[1]}
                rel = {"uri": code_triplet[1], "surfaceform": substring_triplet[1]}
                entities_p.append(sub_ent)
                entities_p.append(obj_ent)
                relations_p.append(rel)
                triplets_p.append({"subject": sub_ent, "predicate": rel, "object": obj_ent})

            prev_len += len(text)

            if len(code_triplets) == 0:
                text = ""
                continue

            text = re.sub("\s{2,}", " ", text)

            processed_obj = {
                "docid": f"{article['docid']}",
                "text": text,
                "entities": entities_p,
                "relations": relations_p,
                "triplets": triplets_p,
            }
            processed_data.append(processed_obj)
            count += 1
            text = ""
    return processed_data


def is_a_num(s):
    return s.replace(".", "", 1).replace("-", "", 1).isdigit()


def map_names(data):  # need to save also list of entities and relations
    data = {
        "id": data["id"],
        "triplets": data["triplets"],
        "text": data["text"],
    }
    triplets = []
    entities = []
    relations = []
    entities_set = set()
    relations_set = set()
    for curr_triplet in data["triplets"]:
        triplet = {}
        triplet["subject"] = {}
        triplet["object"] = {}
        triplet["predicate"] = {}

        if curr_triplet["subject"]["uri"] in en_map:
            triplet["subject"]["surfaceform"] = en_map[(curr_triplet["subject"]["uri"])]
        elif curr_triplet["subject"]["uri"].startswith("Q"):
            # to be thrown out
            continue
        else:
            # literal
            triplet["subject"]["surfaceform"] = curr_triplet["subject"]["surfaceform"]
        triplet["subject"]["uri"] = curr_triplet["subject"]["uri"]

        if curr_triplet["object"]["uri"] in en_map:
            triplet["object"]["surfaceform"] = en_map[(curr_triplet["object"]["uri"])]
        elif curr_triplet["object"]["uri"].startswith("Q"):
            # to be thrown out
            continue
        else:
            # literal
            triplet["object"]["surfaceform"] = curr_triplet["object"]["surfaceform"]
        triplet["object"]["uri"] = curr_triplet["object"]["uri"]

        triplet["predicate"]["surfaceform"] = rel_map[(curr_triplet["predicate"]["uri"])]
        triplet["predicate"]["uri"] = curr_triplet["predicate"]["uri"]

        if triplet["subject"]["uri"] not in entities_set:
            entities.append(triplet["subject"])
            entities_set.add(triplet["subject"]["uri"])
        if triplet["object"]["uri"] not in entities_set:
            entities.append(triplet["object"])
            entities_set.add(triplet["object"]["uri"])
        if triplet["predicate"]["uri"] not in relations_set:
            relations.append(triplet["predicate"])
            relations_set.add(triplet["predicate"]["uri"])

        triplets.append(triplet)
    if len(triplets) > 0:
        data["triplets"] = triplets
        data["entities"] = entities
        data["relations"] = relations
        return data
    else:
        return {}


def read_relations_dict(relations_path):
    if os.path.exists(relations_path):
        with open(relations_path, 'r') as fp:
            data = json.load(fp)
        for key, val in data.items():
            data[key] = set(val)
    else:
        data = {}
    return data


def save_relations_dict(relations_dict, relations_path):
    for key, val in relations_dict.items():
        relations_dict[key] = list(val)
    with open(relations_path, 'w') as fp:
        json.dump(relations_dict, fp)