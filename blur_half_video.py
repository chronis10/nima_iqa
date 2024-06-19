import cv2

def blur_frame(frame, kernel_size=(51, 51)):
    """
    Apply a Gaussian blur to the frame.
    """
    return cv2.GaussianBlur(frame, kernel_size, 0)

def process_video(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Blur the frame if it is in the second half of the video
        if frame_idx >= frame_count // 2:
            frame = blur_frame(frame)

        # Write the frame to the output video
        out.write(frame)
        
        frame_idx += 1
        print(f"Processed frame {frame_idx}/{frame_count}")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the input and output video files
    input_video_path = 'video_sample/drone_factory.mp4'
    output_video_path = 'video_sample/drone_factory_blur.mp4'

    # Process the video
    process_video(input_video_path, output_video_path)
