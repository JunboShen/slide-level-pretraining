import torch

from torch import nn
from pathlib import Path
import torch.nn.functional as F

import sys
# For convinience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir.parent))
from slide_encoder.slide_encoder import *


# class Attention(nn.Module):
#     def __init__(self,
#         input_dim,
#         n_classes=2,
#         **kwargs,):
#         super(Attention, self).__init__()
#         self.M = 768 #500 #768
#         self.L = 128
#         self.ATTENTION_BRANCHES = 1

#         self.feature_extractor_part2 = nn.Sequential(
#             nn.Linear(input_dim, self.M),
#             nn.ReLU(),
#         )

#         self.attention = nn.Sequential(
#             nn.Linear(self.M, self.L),  # matrix V
#             nn.Tanh(),
#             nn.Linear(self.L, self.ATTENTION_BRANCHES)  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(self.M * self.ATTENTION_BRANCHES, n_classes),
#         )
    
#     def forward(self, H):
#         H = H.squeeze(0)
#         H = self.feature_extractor_part2(H)  # KxM
#         A = self.attention(H)  # KxATTENTION_BRANCHES
  
#         A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
  
#         A = F.softmax(A, dim=1)  # softmax over K
  
#         Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

#         logits = self.classifier(Z)

#         return logits



def reshape_input(imgs, coords, pad_mask=None):
    if len(imgs.shape) == 4:
        imgs = imgs.squeeze(0)
    if len(coords.shape) == 4:
        coords = coords.squeeze(0)
    if pad_mask is not None:
        if len(pad_mask.shape) != 2:
            pad_mask = pad_mask.squeeze(0)
    return imgs, coords, pad_mask

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
    else:
        print ("Pretrained weights not found at {}".format(pretrained_weights))

class ClassificationHead(nn.Module):
    """
    The classification head for the slide encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    model_arch: str
        The architecture of the slide encoder
    pretrained: str
        The path to the pretrained slide encoder
    freeze: bool
        Whether to freeze the pretrained model
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        feat_layer,
        n_classes=2,
        model_arch="slide_enc6l384d",
        pretrained="../dino/test_/checkpoint.pth",
        freeze=False,
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim * 2
        if model_arch == "slide_enc6l384d":
            self.slide_encoder = slide_enc6l384d(**kwargs)
        elif model_arch == "slide_enc12l768d":
            self.slide_encoder = slide_enc12l768d(**kwargs)
        else:
            raise ValueError("Invalid model architecture")
        #print weights
        for name, param in self.slide_encoder.named_parameters():
            print(name, param)
            break

        # load pretrained weights
        load_pretrained_weights(self.slide_encoder, pretrained, "teacher") #TODO:Originally teacher
         #print weights
        for name, param in self.slide_encoder.named_parameters():
            print(name, param)
            break
        

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        print('Number of classes: ', n_classes)
        # setup the classifier
        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])
        #self.classifier = Attention(input_dim=self.feat_dim, n_classes=n_classes)

    def forward(self, images: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        # inputs: [N, L, D]
        # print(images.shape)
        # print(coords.shape)
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3
        # forward GigaPath slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
        #convert img_enc from list to tensor
        # img_enc = torch.stack(img_enc)
        
        # #squeeze img_enc
        if len(img_enc.shape) == 3:
            img_enc = img_enc.squeeze(0)

        img_enc = [img_enc[i] for i in self.feat_layer]
        img_enc = torch.cat(img_enc, dim=-1)
  
        # classifier
        h = img_enc.reshape([-1, img_enc.size(-1)])
    
        logits = self.classifier(h)
        return logits


def get_model(**kwargs):
    model = ClassificationHead(**kwargs)
    return model


