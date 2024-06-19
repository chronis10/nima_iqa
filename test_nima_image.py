
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from nima.model import NIMA
from nima.mobile_net_v2 import MobileNetV2
import glob

def load_model(weights_path):

    # Initialize the NIMA model
    model = NIMA()
    
    # Load the state dictionary from the .pth file
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # Ensure the keys match
    model_state_dict = model.state_dict()
    for key in list(state_dict.keys()):
        if key not in model_state_dict:
            print(f"Key {key} from state_dict does not match model keys, removing.")
            del state_dict[key]
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=True) 
    return model

def preprocess_image(image_path):
    """
    Preprocess the input image to the required format
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image_path, model):
    """
    Perform inference on an image
    """
    model.eval()  # Set the model to evaluation mode
    image = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(image)
    
    # Output is a probability distribution, we take the mean score as the prediction
    score = (torch.arange(1, 11) * output).sum(dim=1).item()
    
    return score

if __name__ == "__main__":
    # Path to the weights file, image

    weights_path = 'pretrain-model.pth'
    image_paths = glob.glob('samples/*.jpg')

    # Load the model
    model = load_model(weights_path)
    for image_path in image_paths:
        # Perform inference
        score = predict(image_path, model)        
        print(f"Predicted aesthetic score: {score} for image {image_path.split('/')[-1].split('.')[0].title()}")
    
