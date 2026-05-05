import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open camera.")

# Get camera properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)

model_path = hf_hub_download(
    repo_id="jags/yolov8_model_segmentation-set",
    filename="face_yolov8n-seg2_60.pt",
    local_dir="./models",  # Optional: Save locally
)
model = YOLO(model_path)  # Loads as segmentation model

# Blur kernel size (adjust for stronger/weaker blur)
blur_kernel = (99, 99)  # Larger = more blur (odd numbers only)

print("Face segmentation blurring active! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Run face segmentation (masks + boxes)
    results = model(frame, verbose=False)  # Suppress logs for speed

    # Start with original frame
    annotated_frame = frame.copy()

    # Process each detected face
    if results[0].masks is not None:
        for mask_data in results[0].masks.data:
            mask = mask_data.cpu().numpy().astype(bool)  # Shape: (H, W)

            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
                mask = mask.astype(bool)

            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            blurred = cv2.GaussianBlur(frame, blur_kernel, 0)
            annotated_frame[mask_3ch] = blurred[mask_3ch]

    cv2.imshow("Live Face Segmentation Blurring", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
