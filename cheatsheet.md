# cheatsheet

### Select GPU device

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE trainer.devices=\[1\]`

### provide llama path

`python run_inference.py +experiment/inference=$MODEL datamodule=$DATAMODULE trainer=cpu model.pretrained_model_name_or_path=/dlabdata1/llama_hf/13B`

### Build `llama_tokenizable`

`python -m scripts.non_problematic_constrained_world --tokenizer_full_name /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/share/models/llama_hf/7B --tokenizer_short_name llama --constrained_world_id genie`
