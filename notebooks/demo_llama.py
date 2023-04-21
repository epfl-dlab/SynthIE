#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : demo_llama.py
# @Date : 2023-04-21-11-49
# @Project: SynthIE
# @AUTHOR : Saibo Geng
# @Desc :


# If you are using a different directory for your data, update the path below
DATA_DIR="../data"

# To download the data, uncomment the following line and run the cell
# !bash ../download_data.sh $DATA_DIR

#%%

import os
import sys
import gzip
import jsonlines

sys.path.append("../")


"""Load the Model (downloaded in the ../data/models directory)"""
from src.models import GenIELlamaPL

# ckpt_name = "synthie_base_sc.ckpt"
# path_to_checkpoint = os.path.join(DATA_DIR, 'models', ckpt_name)
model = GenIELlamaPL()
model.to("cuda");