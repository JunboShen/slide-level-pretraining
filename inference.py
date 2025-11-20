import timm
from PIL import Image
from torchvision import transforms
import torch

# Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

img_path = "images/prov_normal_000_1.png"
sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

tile_encoder.eval()
with torch.no_grad():
    output = tile_encoder(sample_input).squeeze()
print(output)
print(output.shape)
#outputs shape is (1536,)
import gigapath
import gigapath.slide_encoder

slide_encoder = gigapath.slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)
slide_encoder = slide_encoder.to(torch.float16)
#to cuda

tile_embed = output.unsqueeze(0).unsqueeze(0).to(torch.float16)  # (1, 1, 1536)
coordinates = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float16)  # (1, 6), represents (x, y, z, width, height, level)
#put on cuda
device = torch.device("cuda")
tile_embed = tile_embed.to(device)
coordinates = coordinates.to(device)
slide_encoder = slide_encoder.to(device)



slide_encoder.eval()
with torch.no_grad():
    output = slide_encoder(tile_embed, coordinates)
print(output)
#output is a list, show its shape
print([o.shape for o in output])
