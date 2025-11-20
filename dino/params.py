import argparse

import utils
def get_pretrain_params():

    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters

    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=1, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--global_crops_scale', type=float, default=0.95, help="""Scales of the
        global view. The scale is the relative size of height/width of the global view compared to the original image.""")
    parser.add_argument('--local_crops_scale', type=float, default=0.5, help="""Scales of the
        local views. The scale is the relative size of height/width of the local views compared to the original image.""")

    # Misc
    parser.add_argument('--data_path', default='/m-ent1/ent1/xuhw/TCGA-embed/GigaPath-giant-1B/h5_files/', type=str,
        help='Please specify path to the training data.')
    parser.add_argument('--exlude_data_path', default='./datasets/LUAD-5-gene_TCGA.csv', type=str,
        help='Please specify path to the excluded training data.')
    parser.add_argument('--output_dir', default="./test", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    

  
    parser.add_argument('--tile_size',      type=int, default=256, help='Tile size in pixels')
    parser.add_argument('--max_wsi_size',   type=int, default=32768, help='Maximum WSI size in pixels for the longer side (width or height).')
    parser.add_argument('--max_tiles',      type=int, default=8192, help='Maximum number of tiles to sample from a WSI')
    parser.add_argument('--shuffle_tiles', action='store_true', default=False, help='Shuffle tiles before sampling')

    parser.add_argument('--input_dim',      type=int, default=1536, help='Dimension of input tile embeddings')
    parser.add_argument('--latent_dim',     type=int, default=768, help='Hidden dimension of the slide encoder')
    #parser.add_argument('--feat_layer',     type=str, default='11', help='The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers')

    parser.add_argument('--global_pool',    action='store_true', default=False, help='Use global pooling, will use [CLS] token if False')
    
    parser.add_argument('--gc',             type=int, default=32, help='Gradient accumulation')
  
    parser.add_argument('--dropout',        type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Drop path rate')
   
    

    
    
    
    return parser.parse_args()