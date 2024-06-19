import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from nima.model import NIMA
from nima.mobile_net_v2 import MobileNetV2
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def preprocess_image(image):
    """
    Preprocess the input image to the required format
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(image, model):
    """
    Perform inference on an image
    """
    model.eval()  # Set the model to evaluation mode
    image = preprocess_image(image)
    
    with torch.no_grad():
        output = model(image)
    
    # Output is a probability distribution, we take the mean score as the prediction
    score = (torch.arange(1, 11) * output).sum(dim=1).item()
    
    return score

def analyze_video(video_path, model, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict score for the frame
        score = predict(frame, model)
        scores.append(score)
        
        # Overlay score on the frame
        cv2.putText(frame, f'Score: {score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the frame to the output video
        out.write(frame)
        
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{frame_count}, score: {score}")
        
        # Display the frame with the score
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return scores

def plot_scores(scores):
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label="Frame Score")
    plt.xlabel("Frame")
    plt.ylabel("Aesthetic Score")
    plt.title("Aesthetic Scores for Video Frames")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the weights file and video file
    weights_path = 'pretrain-model.pth'
    video_path = 'video_sample/drone_factory.mp4'
    output_video_path = 'output_video_with_scores.mp4'

    # Load the model
    model = load_model(weights_path)
    
    # Analyze the video
    scores = analyze_video(video_path, model, output_video_path)
    
    # Plot the scores
    plot_scores(scores)
