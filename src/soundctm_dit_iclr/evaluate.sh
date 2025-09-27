#!/bin/bash
GT_DIR="<TEST_DATSET_PATH>" # Path to the directory containing ground truth samples
GEN_DIR="<GENERATED_SAMPLES_PATH>" # Path to the directory containing generated samples
CLAP_MODEL_PATH="<CLAP_MODEL_PATH>" # Path to the directory containing the trained CLAP model (630k-audioset-best.pt)
REFERENCE_CSV="<TEST_CSV_PATH>" # Path to the csv file containing text audio pairs of test dataset
ID_REFERENCE_CSV="data/clap_test.csv" # Path to the csv file containing the reference youtube ids


python evaluate_metrics.py \
    --reference_dir $GT_DIR \
    --generated_dir $GEN_DIR \
    --reference_csv $REFERENCE_CSV \
    --reference_path_acid $ID_REFERENCE_CSV \
    --clap_model_path $CLAP_MODEL_PATH 
