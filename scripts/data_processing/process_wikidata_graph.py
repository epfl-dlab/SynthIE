import pandas as pd
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import networkx as nx
import pickle
from datetime import date
import numpy as np
import traceback
import argparse


DISAMBIGUATION = "Q4167410"
LIST = "Q13406463"
INTERNAL_ITEM = "Q17442446"
CATEGORY = "Q4167836"
literal_dict = {}
# literal_id = 0

nodes_ids = []
edges_ids = []
nodes = []
edges = []


def create_graph(nodes, edges):
    # nodes contain qid and name
    # edges contain sub_qid, obj_qid, pid and rel (name)
    graph = nx.Graph()
    nodes = [(elem["qid"], elem) for elem in nodes]
    edges = [(elem["sub_qid"], elem["obj_qid"], {"pid": elem["pid"], "name": elem["rel"]}) for elem in edges]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def extract_unit_dict(unit_path):
    # process the JSON file
    unit_dict = {}
    with open(unit_path, "r") as f:
        json_dict = json.load(f)
    for key, d in json_dict.items():
        unit_dict[key] = d["label"]
    return unit_dict


def assign_literal_id(literal):
    return literal
    # if literal is None:
    #     return None
    # global literal_dict
    # if literal in literal_dict:
    #     return literal_dict[literal]
    # lid = "L" + str(id(literal))
    # literal_dict[literal] = lid
    # return lid


def assign_complex_node_id(sub_qid, rel, obj_qid):
    return sub_qid + "_" + rel + "_" + obj_qid


def dd2dms(deg):
    d = int(deg)
    md = np.abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]


def transform_time(time_dict):  # TODO check how they handle timezones
    # extract the "(-+)yyyy-mm-dd as hours, minutes and seconds aren't used
    value = ""
    time_string = time_dict["time"].split("T")[0]
    bce = True if time_string[0] == "-" else False
    try:
        year, month, day = time_string.split("-")
    except Exception as e:
        return None
    # remove (-+) and zeros in front of the year
    year = year[1:].lstrip("0")
    precision = time_dict["precision"]
    if precision == 11:
        # day
        date_obj = date(int(year), int(month), int(day))
        value = date_obj.strftime("%d %B %Y")
    elif precision == 10:
        # month
        date_obj = date(int(year), int(month), 1)
        value = date_obj.strftime("%B %Y")
    elif precision == 9:
        # year
        date_obj = date(int(year), 1, 1)  # TODO ValueError: year 19356 is out of range
        value = date_obj.strftime("%Y")
    elif precision == 8:
        # decade
        value = str(year) + "s"
    elif precision == 7:
        century = int(year) / 100 + 1
        value = str(century) + ". century"
    elif precision == 6:
        mill = int(year) / 1000
        value = str(mill) + ". millennium"
    else:
        return None
    if bce:
        value += " BCE"
    return value


def transform_quantity(quantity_dict):
    value = quantity_dict["amount"]
    if value.startswith("+"):
        value = value[1:]
    unit = quantity_dict["unit"]
    if unit.startswith("http"):
        unit = unit.split("/")[-1]
        try:
            unit = unit_dict[unit]
        except Exception as e:
            unit = None
    else:
        unit = None
    if unit is not None:
        return value + " " + unit
    else:
        return value


def transform_coordinate(coordinate_dict):
    latitude = coordinate_dict["latitude"]
    longitude = coordinate_dict["longitude"]
    lat_dir = "N" if latitude > 0 else "S"
    lon_dir = "E" if longitude > 0 else "W"
    lat_d, lat_m, lat_s = dd2dms(np.abs(latitude))
    lon_d, lon_m, lon_s = dd2dms(np.abs(longitude))  # TODO check if it needs to be absolute val
    value = (
        str(lat_d)
        + "°"
        + str(lat_m)
        + "'"
        + str(lat_s)
        + '"'
        + lat_dir
        + ", "
        + str(lon_d)
        + "°"
        + str(lon_m)
        + "'"
        + str(lon_s)
        + '"'
        + lon_dir
    )
    return value


def extract_object(claim):
    try:
        if not ("datavalue" in claim):
            return None, None
        if claim["datatype"] == "wikibase-item":
            obj_qid = claim["datavalue"]["value"]["id"]  # for wikibase items
            if obj_qid in english_dict:
                val = english_dict[obj_qid]
            else:
                return None, None  # both of the entities have to be from English WP
        # object is a literal
        elif claim["datatype"] == "string":
            val = claim["datavalue"]["value"]
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "time":
            # convert time to the format
            val = transform_time(claim["datavalue"]["value"])
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "globe-coordinate":
            # convert the coordinate
            val = transform_coordinate(claim["datavalue"]["value"])
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "quantity":
            # convert the quantity
            val = transform_quantity(claim["datavalue"]["value"])
            obj_qid = assign_literal_id(val)
        elif claim["datatype"] == "external-id":
            # convert the external id
            val = claim["datavalue"]["value"]
            obj_qid = assign_literal_id(val)
        else:
            # filter out other types of literals
            return None, None
        return obj_qid, val
    except Exception as e:
        traceback.print_exc()
        return None, None


def process_qualifiers(
    claim, sub_qid, sub_name, relation_pid, relation_name, obj_qid, obj_name, rows, use_qualifiers=False
):
    if not use_qualifiers:
        rows.append(
            Row(
                simple_sub={"qid": sub_qid, "name": sub_name},
                simple_sub_qid=sub_qid,
                simple_obj={"qid": obj_qid, "name": obj_name},
                simple_obj_qid=obj_qid,
                simple_edge={"pid": relation_pid, "sub_qid": sub_qid, "rel": relation_name, "obj_qid": obj_qid},
                simple_edge_pid=sub_qid + "_" + relation_pid + "_" + obj_qid,
            )
        )
        return rows
    complex_node_id = assign_complex_node_id(sub_qid, relation_pid, obj_qid)
    if "qualifiers" in claim:
        for qualifier, qualifier_list in claim["qualifiers"].items():
            if qualifier in relation_dict:
                qualifier_name = relation_dict[qualifier]
                for qual_elem in qualifier_list:
                    complex_obj_qid, complex_obj_name = extract_object(qual_elem)  # TODO
                    if complex_obj_qid is not None and complex_obj_name is not None:
                        complex_edge = {
                            "pid": qualifier,
                            "sub_qid": complex_node_id,
                            "rel": qualifier_name,
                            "obj_qid": complex_obj_qid,
                        }
                        complex_sub = {
                            "qid": complex_node_id,
                            "complex_sub_qid": sub_qid,
                            "complex_sub_name": sub_name,
                            "complex_pid": relation_pid,
                            "complex_rel": relation_name,
                            "complex_obj_qid": obj_qid,
                            "complex_obj_name": obj_name,
                        }
                        complex_obj = {"qid": complex_obj_qid, "name": complex_obj_name}
                        complex_rel_qid = complex_node_id + "_" + qualifier + "_" + complex_obj_qid
                    else:
                        complex_sub, complex_obj, complex_edge = None, None, None
                        complex_rel_qid = None
                    rows.append(
                        Row(
                            simple_sub={"qid": sub_qid, "name": sub_name},
                            simple_sub_qid=sub_qid,
                            simple_obj={"qid": obj_qid, "name": obj_name},
                            simple_obj_qid=obj_qid,
                            simple_edge={
                                "pid": relation_pid,
                                "sub_qid": sub_qid,
                                "rel": relation_name,
                                "obj_qid": obj_qid,
                            },
                            simple_edge_pid=sub_qid + "_" + relation_pid + "_" + obj_qid,
                            complex_sub=complex_sub,
                            complex_sub_qid=complex_node_id,
                            complex_edge=complex_edge,
                            complex_edge_pid=complex_rel_qid,
                            complex_obj=complex_obj,
                            complex_obj_qid=complex_obj_qid,
                        )
                    )
            else:
                rows.append(
                    Row(
                        simple_sub={"qid": sub_qid, "name": sub_name},
                        simple_sub_qid=sub_qid,
                        simple_obj={"qid": obj_qid, "name": obj_name},
                        simple_obj_qid=obj_qid,
                        simple_edge={"pid": relation_pid, "sub_qid": sub_qid, "rel": relation_name, "obj_qid": obj_qid},
                        simple_edge_pid=sub_qid + "_" + relation_pid + "_" + obj_qid,
                        complex_sub=None,
                        complex_sub_qid=None,
                        complex_edge=None,
                        complex_edge_pid=None,
                        complex_obj=None,
                        complex_obj_qid=None,
                    )
                )

    return rows


def process_claim(claim, subject_qid, subject_name, prop, rows, use_qualifiers):
    rel_name = relation_dict[prop]
    obj_qid, obj_name = extract_object(claim["mainsnak"])
    if obj_qid is not None and obj_name is not None:
        # checking the qualifiers
        rows = process_qualifiers(
            claim, subject_qid, subject_name, prop, rel_name, obj_qid, obj_name, rows, use_qualifiers
        )
    return rows


def find_preferred(claims):
    normal_claims = []
    for claim in claims:
        if claim["rank"] == "preferred":
            return [claim]
        if claim["rank"] == "normal":
            normal_claims.append(claim)
    return normal_claims


def get_entity_info(line):
    rows = []
    try:
        if DISAMBIGUATION in line or LIST in line or INTERNAL_ITEM in line or CATEGORY in line:
            return []
        if line[:-1] is not None:
            row = json.loads(line[:-1])
        else:
            row = ""
        if len(row) == 0:
            return []
        if "type" in row and row["type"] == "item":
            subject_qid = row["id"]
            # filter the nodes without an article in English Wikipedia (and a name)
            # we only keep the triplets for which both nodes are either in EW or a literal
            try:
                subject_name = english_dict[subject_qid]
            except Exception as e:
                return []

            for prop, claims in row["claims"].items():
                if not (prop in relation_dict):  # relation has to be from Rebel/WebNLG
                    continue
                # check if there is a preferred claim
                # if yes, process just that claim, if not, process all of them
                claims_to_process = find_preferred(claims)
                for claim in claims_to_process:
                    rows = process_claim(claim, subject_qid, subject_name, prop, rows, False)
            return rows
        else:
            return []
    except Exception as e:
        traceback.print_exc()
        return []


def helper_func_nodes(x):
    if x is not None:
        if x["qid"] in nodes_ids:
            nodes.append(x)
            nodes_ids.remove(x["qid"])


def helper_func_edges(x):
    if x is not None:
        pid = x["sub_qid"] + "_" + x["pid"] + "_" + x["obj_qid"]
        if pid in edges_ids:
            edges.append(x)
            edges_ids.remove(pid)


def extract_graph_from_parquet(df, simple=True):
    if simple:
        sub_col = "simple_sub_qid"
        obj_col = "simple_obj_qid"
        edge_col = "simple_edge_pid"
    else:
        sub_col = "complex_sub_qid"
        obj_col = "complex_obj_qid"
        edge_col = "complex_edge_pid"
    global nodes_ids
    global nodes
    nodes_ids = list(pd.unique(df[[sub_col, obj_col]].values.ravel("K")))
    global edges_ids
    global edges
    edges_ids = list(pd.unique(df[edge_col]))
    df[sub_col[:-4]].apply(helper_func_nodes)
    df[obj_col[:-4]].apply(helper_func_nodes)
    df[edge_col[:-4]].apply(helper_func_edges)
    del df
    graph = create_graph(nodes, edges)
    return graph


def print_literals_stats(graph_simple):
    print("number of literals and entities")
    literals = 0
    entities = 0
    for u, v in graph_simple.nodes(data=True):
        if u.startswith("L"):
            literals += 1
        elif u.startswith("Q"):
            entities += 1
    print(literals)
    print(entities)


def load_dictionaries(args):
    with open(args.rel_path, "rb") as f:
        relation_dict = pickle.load(f)

    with open(args.en_map, "rb") as f:
        english_dict = pickle.load(f)

    unit_dict = extract_unit_dict(args.unit_path)
    return relation_dict, english_dict, unit_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rel_path", type=str, help="Path to the relations file")
    parser.add_argument("--en_map", type=str, help="Path to the mapping between qid and titles")
    parser.add_argument("--unit_path", type=str, help="Path to the json with relations that can be units")
    parser.add_argument("--dump_path", type=str, help="Path to the wikidata dump json file")
    parser.add_argument("--graph_path", type=str, help="Path to save the graph")
    parser.add_argument("--graph_df", type=str, help="Path where to save parquet containing graph information")
    args, unknown = parser.parse_known_args()

    relation_dict, english_dict, unit_dict = load_dictionaries(args)

    conf = (
        pyspark.SparkConf()
        .setMaster("local[*]")
        .setAll(
            [
                ("spark.driver.memory", "230g"),
                ("spark.driver.maxResultSize", "32G"),
                ("spark.local.dir", "/scratch/tmp/"),
                ("spark.yarn.stagingDir", "/scratch/tmp/"),
            ]
        )
    )

    # create the session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # create the context
    sc = spark.sparkContext
    wikidata_all = sc.textFile(args.dump_path)
    row = wikidata_all.flatMap(get_entity_info)
    all_rows = spark.createDataFrame(row)

    # saving and making networkx
    all_rows.write.mode("overwrite").parquet(args.graph_df)
    all_rows = all_rows.toPandas()
    graph_simple = extract_graph_from_parquet(all_rows)

    print("number of edges and nodes")
    print(len(graph_simple.edges()))
    print(len(graph_simple.nodes()))

    with open(args.graph_path, "wb") as f:
        pickle.dump(graph_simple, f)
