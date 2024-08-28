# Video Object Detection and Segmentation Script

This script processes a video file to detect objects and perform segmentation using the Florence-2 and SAM models. The detected objects are highlighted with bounding boxes and labels, and segmentation masks are overlaid on the video frames. The output is saved as a new video file.

## Requirements

Before running the script, ensure you have the following libraries installed:

- `torch`
- `transformers`
- `requests`
- `Pillow`
- `opencv-python`
- `numpy`
- `ultralytics`

You can install these dependencies using pip:

```bash
pip install torch transformers requests Pillow opencv-python numpy ultralytics
