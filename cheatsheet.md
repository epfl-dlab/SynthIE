# cheatsheet

### Select GPU device

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE trainer.devices=\[1\]`

### provide llama path

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE trainer=cpu model.pretrained_model_name_or_path=/dlabdata1/llama_hf/13B`

### Build `llama_tokenizable`

`python -m scripts.non_problematic_constrained_world --tokenizer_full_name /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/7B --tokenizer_short_name llama --constrained_world_id genie`

### T5

[T5 training](https://huggingface.co/docs/transformers/model_doc/t5?highlight=t5#training)
- model._shift_right:https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model.forward.example

### Check preprocessing input

Preprocessing is handled by `generic_collator.py`, an intermedia between `tokenizer` and `model`.

Below line in `Genie_lan_t5.py`
```python
        # Get input_ids and attention masks
        if not input_is_processed_batch:
            input_data = self.collator.collate_input(input_data)
```




### Data for few-shot learning

- "The General Administration of Quality Supervision, Inspection and Quarantine was replaced by the State Administration for Market Regulation and is a government agency under the parent organization, the State Council of the People's Republic of China. Its headquarters is located in Haidian District, China."
- "Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter."
- "Swedish Open Cultural Heritage is a project developed by the Swedish National Heritage Board, which is mainly focused on cultural heritage. It produces Resource Description Framework as its product or material and uses XML, JSON, and JSON-LD as its file formats. XML was inspired by Standard Generalized Markup Language."
- "The NHL Stadium Series is a sport that consists of ice hockey."
- "Abhishek Pictures is a film production company based in Hyderabad."


- "[s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] replaced by [o] State_Administration_for_Market_Regulation [r] instance of [o] Government_agency [r] parent organization [o] State_Council_of_the_People's_Republic_of_China [r] country [o] China [r] headquarters location [o] Haidian_District [e]",
- "[s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [r] screenwriter [o] B._Babusivan [e]"
- "[s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [r] developer [o] Swedish_National_Heritage_Board [r] product or material produced [o] Resource_Description_Framework [r] file format [o] XML [r] file format [o] JSON [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e]"
- "[s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e]"
- "[s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e]"

prompt = """The General Administration of Quality Supervision, Inspection and Quarantine was replaced by the State Administration for Market Regulation and is a government agency under the parent organization, the State Council of the People's Republic of China. Its headquarters is located in Haidian District, China. -> [s] General_Administration_of_Quality_Supervision,_Inspection_and_Quarantine [r] replaced by [o] State_Administration_for_Market_Regulation [r] instance of [o] Government_agency [r] parent organization [o] State_Council_of_the_People's_Republic_of_China [r] country [o] China [r] headquarters location [o] Haidian_District [e];

Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter. -> [s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [r] screenwriter [o] B._Babusivan [e];

Swedish Open Cultural Heritage is a project developed by the Swedish National Heritage Board, which is mainly focused on cultural heritage. It produces Resource Description Framework as its product or material and uses XML, JSON, and JSON-LD as its file formats. XML was inspired by Standard Generalized Markup Language. -> [s] Swedish_Open_Cultural_Heritage [r] main subject [o] Cultural_heritage [r] developer [o] Swedish_National_Heritage_Board [r] product or material produced [o] Resource_Description_Framework [r] file format [o] XML [r] file format [o] JSON [r] file format [o] JSON-LD [e] [s] XML [r] inspired by [o] Standard_Generalized_Markup_Language [e];

The NHL Stadium Series is a sport that consists of ice hockey. -> [s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e];

Abhishek Pictures is a film production company based in Hyderabad. -> [s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e]; 

The Journal of Colloid and Interface Science is a bibliographic review indexed in Scopus and published by Elsevier. Its main subject is chemical engineering, and it is written in the English language. It is based in the United States, and is owned by Elsevier, the same company that owns Scopus. -> """


short_prompt = """
The NHL Stadium Series is a sport that consists of ice hockey. -> [s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e];

Abhishek Pictures is a film production company based in Hyderabad. -> [s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e]; 

The Journal of Colloid and Interface Science is a bibliographic review indexed in Scopus and published by Elsevier. Its main subject is chemical engineering, and it is written in the English language. It is based in the United States, and is owned by Elsevier, the same company that owns Scopus. -> """


new_prompt="""
- Vettaikaaran (2009 film) was originally written in the Tamil language, with B. Babusivan as the screenwriter.
- The NHL Stadium Series is a sport that consists of ice hockey.
- Abhishek Pictures is a film production company based in Hyderabad.
- The Journal of Colloid and Interface Science is a bibliographic review indexed in Scopus and published by Elsevier. Its main subject is chemical engineering, and it is written in the English language. It is based in the United States, and is owned by Elsevier, the same company that owns Scopus.

- [s] Vettaikaaran_(2009_film) [r] original language of film or TV show [o] Tamil_language [r] screenwriter [o] B._Babusivan [e]
- [s] NHL_Stadium_Series [r] sport [o] Ice_hockey [e]
- [s] Abhishek_Pictures [r] industry [o] Film_industry [r] headquarters location [o] Hyderabad [e]
- """

```python
from src.models.genie_llama import GenIELlamaPL
model_7b = GenIELlamaPL(from_pretrained=True, pretrained_model_name_or_path=
"/dlabdata1/llama_hf/7B",
                     linearization_class_id="subject_collapsed", default_collator_parameters=
                     {"max_input_length": 2048, "padding": "longest", "truncation": True},
                     inference={"hf_generation_params": {"num_beams": 10, "num_return_sequences": 10,
                                                         "early_stopping": False, "encoder_no_repeat_ngram_size": 0
                         , "no_repeat_ngram_size": 0, "temperature": 1.0, "length_penalty": 1.0, "max_new_tokens": 256}}
                     )
texts=[prompt]

override_models_default_hf_generation_parameters = {
    "num_beams": 10,
    "num_return_sequences": 1,
    "return_dict_in_generate": True,
    "output_scores": True,
    "seed": 123,
    "length_penalty": 0.8
}

output = model_7b.sample(texts,
                    return_generation_outputs=True,
                      convert_to_triplets=True,
                      **override_models_default_hf_generation_parameters)
print(model_7b.hparams.pretrained_model_name_or_path.split("/")[-1])
print(model_7b.tokenizer.batch_decode(output['generation_outputs'].sequences, skip_special_tokens=True))
print(output['grouped_decoded_outputs'][0])
```

### Llama+ constrained decoding 

# state_id="et_id", this is strange. it should be

```python
"sub_id": ["rel_id"],
"rel_id": ["obj_id"],
"obj_id": ["rel_id", "et_id"],
"et_id": ["sub_id"],
```