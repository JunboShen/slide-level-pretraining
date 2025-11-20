import torch
import os
from pathlib import Path

import sys
# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))
from dinov2.models.slide_encoder.slide_encoder import *
from dino.params import get_pretrain_params

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        #state_dict = state_dict["model"]
        #print keys in state_dict
        # print("Keys in state_dict")
        # for key in state_dict:
        #     print(key)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        #print the first item in the state_dict

        msg = model.load_state_dict(state_dict, strict= False)
    else:
        print ("Pretrained weights not found at {}".format(pretrained_weights))

args = get_pretrain_params()
model = slide_enc3l384d(**vars(args))
for name, param in model.named_parameters():
    print(name, param)
    break

load_pretrained_weights(model, "../dinov2/test/eval/training_599/teacher_checkpoint.pth", "teacher")
#print weights
for name, param in model.named_parameters():
    print(name, param)
    break

