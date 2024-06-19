import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the pre-trained model."""
    model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                           regressor_dataset="kadid10k")  # You can choose any of the available datasets
    model.eval().to(device)
    return model

def preprocess_image(img):
    """Preprocess the input image."""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img).unsqueeze(0).to(device)
    return img

def get_half_scale_image(img):
    """Get the half-scale version of the input image."""
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
    return img_ds

def compute_quality_score(model, img, img_ds):
    """Compute the quality score of the image."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    return score.item()

if __name__ == "__main__":
    # Define the directory path
    
    image_paths = glob.glob('samples/*.jpg')
    model = load_model()
    for image_path in image_paths:
            
        # Load the full-scale image
        img = Image.open(image_path).convert("RGB")
        
        # Get the half-scale image
        img_ds = get_half_scale_image(img)
        
        # Preprocess the images
        img = preprocess_image(img)
        img_ds = preprocess_image(img_ds)
            
        score = compute_quality_score(model, img, img_ds)
            
        print(f"Predicted aesthetic score: {score} for image {image_path.split('/')[-1].split('.')[0].title()}")

