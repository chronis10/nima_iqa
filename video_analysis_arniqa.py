import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    width, height = img.size
    resize_transform = transforms.Resize((height // 2, width // 2))
    img_ds = resize_transform(img)
    return img_ds

def compute_quality_score(model, img, img_ds):
    """Compute the quality score of the image."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
    return score.item()

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

        # Convert the frame to PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Get the half-scale image
        img_ds = get_half_scale_image(img)

        # Preprocess the images
        img = preprocess_image(img)
        img_ds = preprocess_image(img_ds)

        # Compute the quality score
        score = compute_quality_score(model, img, img_ds)
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
    plt.ylabel("ARNIQA Score")
    plt.title("ARNIQA Scores for Video Frames")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Path to the weights file and video file
    video_path = 'video_sample/drone_factory_blur.mp4'
    output_video_path = 'output_video_with_scores.mp4'

    # Load the model
    model = load_model()
    
    # Analyze the video
    scores = analyze_video(video_path, model, output_video_path)
    
    # Plot the scores
    plot_scores(scores)
