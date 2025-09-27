#!/bin/bash
SAMPLING_CONFIG_FILE="<CONFIG_FILE_PATH>" # Path to the config file for evaluation ex: configs/hyperparameter_configs/inference/dit.yaml

python ctm_inference.py --config $SAMPLING_CONFIG_FILE