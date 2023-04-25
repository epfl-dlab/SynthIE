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
from src.models import GenIEFlanT5PL
from src.constrained_generation import IEConstrainedGeneration

DATA_DIR = "./data"

override_models_default_hf_generation_parameters = {
    "num_beams": 1,
    "num_return_sequences": 1,
    "return_dict_in_generate": True,
    "output_scores": True,
    "seed": 123,
    "length_penalty": 0.8
}

texts= ["The first president of the United States was George Washington."]



##########################
#
#      Llama
#
##########################

model_7b = GenIELlamaPL(from_pretrained=True, pretrained_model_name_or_path="/home/saibo/Research/llama-7B",
                        linearization_class_id="fully_expanded", default_collator_parameters=
                        {"max_input_length": 24, "padding": "longest", "truncation": True},
                        inference={"hf_generation_params": {"num_beams": 1, "num_return_sequences": 1,
                                                            "early_stopping": False, "encoder_no_repeat_ngram_size": 0
                            , "no_repeat_ngram_size": 0, "temperature": 1.0, "length_penalty": 1.0,
                                                            "max_new_tokens": 24}})

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

output = model_7b.sample(texts,
                      convert_to_triplets=True,
                      return_generation_outputs=True,
                      **override_models_default_hf_generation_parameters)


##########################
#
#      T5
#
##########################


ckpt_name = "synthie_base_sc.ckpt"
path_to_checkpoint = os.path.join(DATA_DIR, 'models', ckpt_name)
model_t5 = GenIEFlanT5PL.load_from_checkpoint(checkpoint_path=path_to_checkpoint)
model_t5.to("cuda")

params = {}
params['constrained_worlds_dir'] = os.path.join(DATA_DIR, "constrained_worlds")
params[
    'constrained_world_id'] = "genie_t5_tokenizeable"  # specifies the folder name from which the constrained world is loaded
params['identifier'] = "genie_t5_tokenizeable"  # specifies the cache subfolder where the trie will be stored

params['path_to_trie_cache_dir'] = os.path.join(DATA_DIR, ".cache")
params['path_to_entid2name_mapping'] = os.path.join(DATA_DIR, "id2name_mappings", "entity_mapping.jsonl")
params['path_to_relid2name_mapping'] = os.path.join(DATA_DIR, "id2name_mappings", "relation_mapping.jsonl")

constraint_module = IEConstrainedGeneration.from_constrained_world(model=model_t5,
                                                                   linearization_class_id=model_t5.hparams.linearization_class_id,
                                                                   **params)

model_t5.constraint_module = constraint_module


output = model_t5.sample(texts,
                      convert_to_triplets=True,
                      return_generation_outputs=True,
                      **override_models_default_hf_generation_parameters)

###

