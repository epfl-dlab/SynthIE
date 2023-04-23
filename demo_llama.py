#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : demo_llama.py
# @Date : 2023-04-21-20-35
# @Project: SynthIE
# @AUTHOR : Saibo Geng
# @Desc :
import os
from pprint import pprint
from src.models.genie_llama import GenIELlamaPL



override_models_default_hf_generation_parameters = {
    "num_beams": 10,
    "num_return_sequences": 1,
    "return_dict_in_generate": True,
    "output_scores": True,
    "seed": 123,
    "length_penalty": 0.8
}

texts = [
    'The Journal of Colloid and Interface Science is a bibliographic review indexed in Scopus and published by Elsevier. Its main subject is chemical engineering, and it is written in the English language. It is based in the United States, and is owned by Elsevier, the same company that owns Scopus.']


model_7b = GenIELlamaPL(from_pretrained=True, pretrained_model_name_or_path=
"/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/7B",
                     linearization_class_id="fully_expanded", default_collator_parameters=
                     {"max_input_length": 250, "padding": "longest", "truncation": True},
                     inference={"hf_generation_params": {"num_beams": 10, "num_return_sequences": 10,
                                                         "early_stopping": False, "encoder_no_repeat_ngram_size": 0
                         , "no_repeat_ngram_size": 0, "temperature": 1.0, "length_penalty": 1.0, "max_new_tokens": 256}}
                     )

model_13b = GenIELlamaPL(from_pretrained=True, pretrained_model_name_or_path=
"/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/13B",
                     linearization_class_id="fully_expanded", default_collator_parameters=
                     {"max_input_length": 250, "padding": "longest", "truncation": True},
                     inference={"hf_generation_params": {"num_beams": 10, "num_return_sequences": 10,
                                                         "early_stopping": False, "encoder_no_repeat_ngram_size": 0
                         , "no_repeat_ngram_size": 0, "temperature": 1.0, "length_penalty": 1.0, "max_new_tokens": 256}}
                     )

model_30b = GenIELlamaPL(from_pretrained=True, pretrained_model_name_or_path=
"/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/30B",
                     linearization_class_id="fully_expanded", default_collator_parameters=
                     {"max_input_length": 250, "padding": "longest", "truncation": True},
                     inference={"hf_generation_params": {"num_beams": 10, "num_return_sequences": 10,
                                                         "early_stopping": False, "encoder_no_repeat_ngram_size": 0
                         , "no_repeat_ngram_size": 0, "temperature": 1.0, "length_penalty": 1.0, "max_new_tokens": 256}}
                     )

model_65b = GenIELlamaPL(from_pretrained=True, pretrained_model_name_or_path=
"/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/65B",
                     linearization_class_id="fully_expanded", default_collator_parameters=
                     {"max_input_length": 250, "padding": "longest", "truncation": True},
                     inference={"hf_generation_params": {"num_beams": 10, "num_return_sequences": 10,
                                                         "early_stopping": False, "encoder_no_repeat_ngram_size": 0
                         , "no_repeat_ngram_size": 0, "temperature": 1.0, "length_penalty": 1.0, "max_new_tokens": 256}}
                     )


"""Load constrained decoding module"""
from src.constrained_generation import IEConstrainedGeneration

DATA_DIR = "./data"
params = {}
params['constrained_worlds_dir'] = os.path.join(DATA_DIR, "constrained_worlds")
params[
    'constrained_world_id'] = "genie_llama_tokenizeable"  # specifies the folder name from which the constrained world is loaded
params['identifier'] = "genie_llama_tokenizeable"  # specifies the cache subfolder where the trie will be stored

params['path_to_trie_cache_dir'] = os.path.join(DATA_DIR, ".cache")
params['path_to_entid2name_mapping'] = os.path.join(DATA_DIR, "id2name_mappings", "entity_mapping.jsonl")
params['path_to_relid2name_mapping'] = os.path.join(DATA_DIR, "id2name_mappings", "relation_mapping.jsonl")

constraint_module = IEConstrainedGeneration.from_constrained_world(model=model_7b,
                                                                   linearization_class_id=model_7b.hparams.linearization_class_id,
                                                                   **params)

model_7b.constraint_module = constraint_module
model_13b.constraint_module = constraint_module
model_30b.constraint_module = constraint_module
model_65b.constraint_module = constraint_module

for model in [model_7b, model_13b, model_30b, model_65b]:
    output = model.sample(texts,
                          convert_to_triplets=True,
                          **override_models_default_hf_generation_parameters)
    print(model.hparams.pretrained_model_name_or_path.split("/")[-1])
    pprint(output['grouped_decoded_outputs'][0])

