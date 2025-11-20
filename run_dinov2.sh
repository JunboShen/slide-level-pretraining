#!/bin/bash

# Default dinov2 arguments
CONFIG_FILE="dinov2/configs/train/slide_encoder.yaml"
OUTPUT_DIR="./dinov2test"
LOCAL_RANK=0
# Slide encoder arguments
LOCAL_CROPS_NUMBER=8
GLOBAL_CROPS_SCALE=1.0 #origin 0.95, gives bugs for one particular image, so use 1.0
LOCAL_CROPS_SCALE=0.5
DATA_PATH=''
EXCLUDE_DATA_PATH=''

TILE_SIZE=256
MAX_WSI_SIZE=32768
MAX_TILES=8192
SHUFFLE_TILES=False
INPUT_DIM=1536
LATENT_DIM=768
GLOBAL_POOL=False
GC=32
DROPOUT=0.1
DROP_PATH_RATE=0.0

WANDB_API_KEY=""

# Run the wandb login
wandb login $WANDB_API_KEY
# Run the train.py with the default arguments
PYTHONPATH=. python dinov2/train/train.py \
  --local_crops_number $LOCAL_CROPS_NUMBER \
  --global_crops_scale $GLOBAL_CROPS_SCALE \
  --local_crops_scale $LOCAL_CROPS_SCALE \
  --data_path $DATA_PATH \
  --exclude_data_path $EXCLUDE_DATA_PATH \
  --tile_size $TILE_SIZE \
  --max_wsi_size $MAX_WSI_SIZE \
  --max_tiles $MAX_TILES \
  --input_dim $INPUT_DIM \
  --latent_dim $LATENT_DIM \
  --gc $GC \
  --dropout $DROPOUT \
  --drop_path_rate $DROP_PATH_RATE \
  --config-file $CONFIG_FILE \
  --output_dir $OUTPUT_DIR \
  --local-rank $LOCAL_RANK
