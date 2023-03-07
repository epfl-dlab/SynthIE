#!/bin/bash

url="https://huggingface.co/datasets/martinjosifoski/SynthIE/resolve/main"

# Unless a data_directory is passed as an argument, download the files in the `data` directory
path_to_data_dir=${1:-"data"}
echo "Downloading the data to '$path_to_data_dir'."

# Create it if it does not exist
mkdir -p $path_to_data_dir
cd $path_to_data_dir

###################
# Download Datasets
###################
# ~~~ Raw Data ~~~~

# SynText
dataset_dir="sdg_text_davinci_003"
mkdir $dataset_dir && cd $dataset_dir
for split in "val" "test" "test_small"
do
    wget "$url/$dataset_dir/$split.jsonl.gz"
done
cd ..

# SynCode
dataset_dir="sdg_code_davinci_002"
mkdir $dataset_dir && cd $dataset_dir
for split in "train" "val" "test" "test_small"
do
    wget "$url/$dataset_dir/$split.jsonl.gz"
done
cd ..

# Rebel
dataset_dir="rebel"
mkdir $dataset_dir && cd $dataset_dir
for split in "train" "val" "test" "test_small"
do
    wget "$url/$dataset_dir/$split.jsonl.gz"
done
cd ..

# ~~~ Processed Data (used in the paper) ~~~~

# The preprocessing consists of some pre-computation to speed up the data loading in the experiments:
# - triplets are ordered according to a heuristic detecting the constituent entities' appearence position in the text (cf. paper)
# - targets are pre-tokenized using the T5 tokenizer according to both linearizations the fully-expanded and subject collapsed linearization (cf. paper) and
# - the number of tokens in the input text and the linearized targets are recorded.

mkdir processed && cd processed

# SynText
dataset_dir="sdg_text_davinci_003"
mkdir $dataset_dir && cd $dataset_dir
for split in "val_ordered" "test_ordered" "test_small_ordered"
do
    wget "$url/processed/$dataset_dir/$split.jsonl.gz"
done
cd ..

# SynCode
dataset_dir="sdg_code_davinci_002"
mkdir $dataset_dir && cd $dataset_dir
for split in "train_ordered" "val_ordered" "test_ordered" "test_small_ordered"
do
    wget "$url/processed/$dataset_dir/$split.jsonl.gz"
done
cd ..

# Rebel
dataset_dir="rebel"
mkdir $dataset_dir && cd $dataset_dir
for split in "train" "val" "test" "test_small"  # Rebel is already ordered so the ordering heuristic is not applied
do
    wget "$url/processed/$dataset_dir/$split.jsonl.gz"
done
cd ..

cd ..
###################

########################################
# Download Constrained World Definitions
########################################
wget "$url/constrained_worlds.tar.gz"
tar -zxvf constrained_worlds.tar.gz && rm constrained_worlds.tar.gz
########################################

#######################################
# Download WikidataID2Name Dictionaries
#######################################
wget "$url/id2name_mappings.tar.gz"
tar -zxvf id2name_mappings.tar.gz && rm id2name_mappings.tar.gz
#######################################

echo "The data was download at '$path_to_data_dir'."
