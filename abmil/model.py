import torch
import torch.nn as nn
import torch.nn.functional as F
from .pos_embed import get_2d_sincos_pos_embed

    

    
class Attention(nn.Module):
    def __init__(self,
        input_dim,
        n_classes=2,
        slide_ngrids=1000, 
        **kwargs,):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(input_dim, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, n_classes),
        )
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids**2
        self.register_buffer('pos_embed', torch.zeros(1, num_patches, self.M), persistent=False)  # fixed sin-cos embedding

        self.initialize_weights()
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.slide_ngrids, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))  
    
    
    # def coords_to_pos(self, coords):
    #     """
    #     This function is used to convert the coordinates to the positional indices

    #     Arguments:
    #     ----------
    #     coords: torch.Tensor
    #         The coordinates of the patches, of shape [N, L, 2]
    #     output: torch.Tensor
    #         The positional indices of the patches, of shape [N, L]
    #     """
    #     coords_ = torch.floor(coords / 256.0)
    #     pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
    #     return pos.long() 

    def forward(self, H, coords):
        H = H.squeeze(0)

        H = self.feature_extractor_part2(H)  # KxM
        
        # pos = self.coords_to_pos(coords).squeeze(0)
    

        # H = H + self.pos_embed[:, pos, :].squeeze(0)

        A = self.attention(H)  # KxATTENTION_BRANCHES
  
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
  
        A = F.softmax(A, dim=1)  # softmax over K
  

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        logits = self.classifier(Z)

        return logits

'''    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A'''


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(768, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L,
                                     self.ATTENTION_BRANCHES)  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, H):
        H = H.squeeze(0)
        H = self.feature_extractor_part2(H)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U)  # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Attention(768, 2)
    model = model.to(device)
    #set random seed
    torch.manual_seed(0)
    
    #test the model with random input of shape (B, L, H)
    x = torch.randn(1, 32,768)


    #add to cuda
    x = x.to(device)
    print(x.shape)

    #random coords
    coords = torch.randint(0, 1000, (1, 32, 2)).to(device)
    print(coords.shape)

    #forward pass
    output = model(x, coords)
    print(output)
    print(output.shape)
    print("Model run successful")

