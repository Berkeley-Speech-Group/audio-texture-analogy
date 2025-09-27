TRAINING_CONFIG_FILE="<CONFIG_FILE_PATH>" # Path to the config file for training ex: configs/hyperparameter_configs/training/dit.yaml
BIND_DIR="<ANY_DIRECTORY_TO_BE_MOUNTED>" # Path to the directory to be mounted for accessability
DATASET_PATH="<DATASET_PATH>"   # Path to the dataset directory ex: /dataset/audiocaps

TRAIN_IMAGE="<SINGULARITY_IMAGE_PATH>" # Path to the singularity image for training

singularity exec --bind $BIND_DIR:$BIND_DIR \
    --bind $DATASET_PATH:$PWD/data/audiocaps/ \
    --nv $TRAIN_IMAGE /bin/bash -c \
    "accelerate launch --main_process_port 29504 ctm_train.py --config $TRAINING_CONFIG_FILE"