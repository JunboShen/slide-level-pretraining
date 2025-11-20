export CUDA_VISIBLE_DEVICES=3
# Task setting
TASKCFG=finetune_dinov2_cls_global/task_configs/mutation_5_gene_freeze.yaml
DATASETCSV=
ROOTPATH=
MAX_WSI_SIZE=32768 #Maximum WSI size in pixels for the longer side (width or height).
TILE_SIZE=256
# Model settings
PRETRAINED=
MODELARCH=slide_enc12l768d
TILEEMBEDSIZE=1536
LATENTDIM=768
# Training settings
EPOCH=10 
GC=32
BLR=0.004 
WD=0.01 
LD=0.95
FEATLAYER="9-10-11-12" 
DROPOUT=0.1
GLOBAL_POOL=False
FREEZE=True
#TODO: model_select  Originally last epoch
# Output settings
WORKSPACE=outputs/dinov2 
SPLITDIR=$WORKSPACE/splits
SAVEDIR=$WORKSPACE
EXPNAME=run_epoch-${EPOCH}_blr-${BLR}_wd-${WD}_ld-${LD}_feat-${FEATLAYER}_model-select-val

echo "Data directory set to $ROOTPATH"

python finetune_dinov2_cls_global/main.py --task_cfg_path ${TASKCFG} \
               --dataset_csv $DATASETCSV \
               --root_path $ROOTPATH \
               --model_arch $MODELARCH \
               --blr $BLR \
               --layer_decay $LD \
               --optim_wd $WD \
               --dropout $DROPOUT \
               --drop_path_rate 0.0 \
               --val_r 0.1 \
               --epochs $EPOCH \
               --input_dim $TILEEMBEDSIZE \
               --latent_dim $LATENTDIM \
               --feat_layer $FEATLAYER \
               --warmup_epochs 1 \
               --gc $GC \
               --model_select last_epoch \
               --lr_scheduler cosine \
               --folds 10 \
               --dataset_csv $DATASETCSV \
               --split_dir $SPLITDIR \
               --save_dir $SAVEDIR \
               --pretrained $PRETRAINED \
               --report_to tensorboard \
               --exp_name $EXPNAME \
               --max_wsi_size $MAX_WSI_SIZE \
               --freeze