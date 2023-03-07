#!/bin/bash

url="https://huggingface.co/martinjosifoski/SynthIE/resolve/main"

# Unless a model_directory is passed as an argument, download the files in the `data/models` directory
path_to_models_dir=${1:-"data/models"}
echo "Downloading the models to '$path_to_models_dir'."

#################################
##### Download Pre-Trained Models
#################################
mkdir -p $path_to_models_dir
cd $path_to_models_dir

# SynthIE models (trained on the SynthIE-code dataset)
echo "Downloading SynthIE-base-FE"
wget "$url/synthie_base_fe.ckpt"  # SynthIE-base-FE
echo "Downloading SynthIE-base-SC"
wget "$url/synthie_base_sc.ckpt"  # SynthIE-base-SC
echo "Downloading SynthIE-large-FE"
wget "$url/synthie_large_fe.ckpt"  # SynthIE-large-FE

# GenIE models (trained on the REBEL dataset)
echo "Downloading GenIE-base-FE"
wget "$url/genie_base_fe.ckpt"  # GenIE-base-FE
echo "Downloading GenIE-base-SC"
wget "$url/genie_base_sc.ckpt"  # GenIE-base-SC
#################################

echo "The model checkpoints were downloaded to '$path_to_models_dir'."
