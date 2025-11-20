Pretraining with folders "dino", "dinov2" uses cropping on the original slide, leaves some empty spaces in the cropped area (some coordinates don't have corresponding tile embeddings).

"dinov2_with_new_settings" pretrain with a different setting, first rearrange input tile embeddings in order into a square, then do the cropping on the square area. This way is more stable in training DINOv2 without break down in the middle of training. "dinov2_with_new_settings" adds gradient accumulation in training, with smoother convergence.

Pretraining DINO, run dino/run_main.sh

Pretraining DINOv2, run run_dinov2.sh , dinov2_with_new_settings/run_dinov2.sh

Finetune scripts: scripts/run_mutation_dino_freeze_4layers_cls_4layes_globalpool.sh, (Uses cls tokens from last 4 layesrs + global poolings from last 4 layesrs)

scripts/run_mutation_dinov2_freeze_4layers_cls_4layes_globalpool.sh, (Uses cls tokens from last 4 layesrs + global poolings from last 4 layesrs)

dinov2_with_new_settings/scripts_finetune_dinov2/run_mutation_dinov2_4layers_cls_input_global_pool.sh (Uses cls tokens from last 4 layesrs + global pooling on the input embeddings)
