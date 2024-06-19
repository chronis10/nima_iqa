import torch
import torchvision.transforms as transforms
from PIL import Image

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load the model
model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                       regressor_dataset="kadid10k")    # You can choose any of the available datasets
model.eval().to(device)

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the full-scale image
img_path = "/Users/christoschronis/nima_iqa/samples/blur.jpg"
img = Image.open(img_path).convert("RGB")

# Get the half-scale image
img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

# Preprocess the images
img = preprocess(img).unsqueeze(0).to(device)
img_ds = preprocess(img_ds).unsqueeze(0).to(device)

# NOTE: here, for simplicity, we compute the quality score of the whole image.
# In the paper, we average the scores of the center and four corners crops of the image.

# Compute the quality score
with torch.no_grad(), torch.cuda.amp.autocast():
    score = model(img, img_ds, return_embedding=False, scale_score=True)

print(f"Image quality score: {score.item()}")