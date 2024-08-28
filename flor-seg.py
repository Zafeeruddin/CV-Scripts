import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
import numpy as np
from ultralytics import SAM

# Set up device and model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Florence-2 model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Load SAM model
sam_model = SAM("sam2_b.pt")

# Set up video capture and writer
video_path = "Golden Retriever Pup Makes Baby Cry But Says Sorry! (Cutest Ever!!).mp4"  # Specify your input video file path
cap = cv2.VideoCapture(video_path)
output_path = "output_video.mp4"  # Specify your output video file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Define vibrant colors (BGR format for OpenCV)
colors = [
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Yellow
    (0, 128, 255),  # Orange
    (255, 0, 128),  # Pink
    (0, 255, 0),    # Lime Green
    (128, 0, 255),  # Violet
]

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Prepare inputs for the Florence-2 model
    prompt = "<OD>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    # Generate predictions
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    
    # Decode and process results
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
    
    # Extract bounding boxes and labels
    bboxes = parsed_answer['<OD>']['bboxes']
    labels = parsed_answer['<OD>']['labels']
    
    # Draw bounding boxes and labels on the frame
    for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        color = colors[idx % len(colors)]
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(frame, start_point, end_point, color=color, thickness=2)
        label_position = (int(bbox[0]), int(bbox[1]) - 10)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2, lineType=cv2.LINE_AA)
    
    # Segment with bounding box prompt
    results = sam_model(frame, bboxes=bboxes)
    
    # Overlay segmentation results on the frame
    segmented_frame = results[0].plot()
    
    # Combine bounding boxes with segmentation results
    # Convert segmented_frame to BGR if it's not already
    if segmented_frame.shape[2] == 4:  # RGBA
        segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2BGR)

    # Add the bounding boxes and labels to the segmented frame
    for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        color = colors[idx % len(colors)]
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(segmented_frame, start_point, end_point, color=color, thickness=2)
        label_position = (int(bbox[0]), int(bbox[1]) - 10)
        cv2.putText(segmented_frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2, lineType=cv2.LINE_AA)
    
    # Write the final frame with bounding boxes, labels, and segmentation results to the output video
    out.write(segmented_frame)

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()