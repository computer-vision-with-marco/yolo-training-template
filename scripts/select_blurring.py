import cv2
import numpy as np
from ultralytics import YOLO
import os

# ================== CONFIG ==================
MODEL_PATH = "YOLO-Human-Parse/weights/yolo-human-parse-epoch-125.pt"  # ← your working model
BLUR_STRENGTH = (101, 101)   # change for stronger/weaker blur
# ===========================================

# Check model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nDownload it first!")

model = YOLO(MODEL_PATH)

# Class names (in order)
class_names = ["Hair", "Face", "Neck", "Arm", "Hand", "Back", "Leg", "Foot", "Outfit", "Person", "Phone"]
num_classes = len(class_names)

# State: which parts are selected for blurring (True/False)
selected = [False] * num_classes
selected[2] = True   # Neck (torso) on by default
selected[8] = True   # Outfit (clothes) on by default → good torso coverage

# UI settings
checkbox_size = 20
padding = 10
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 2

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Create a blank panel on the left for checkboxes
panel_width = 300
total_width = w + panel_width
display = np.zeros((h, total_width, 3), dtype=np.uint8)

print("UI Ready! Click checkboxes to select body parts to blur. Press 'q' to quit.")


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < panel_width:
            for i in range(num_classes):
                cy = 40 + i * 35
                if 20 <= x <= 20 + checkbox_size and cy - 15 <= y <= cy + checkbox_size - 15:
                    selected[i] = not selected[i]
                    print(f"Toggled: {class_names[i]} → {'BLUR' if selected[i] else 'SHARP'}")


cv2.namedWindow("Body Part Blur Selector (Click checkboxes!)")
cv2.setMouseCallback("Body Part Blur Selector (Click checkboxes!)", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to fit
    frame_resized = cv2.resize(frame, (w, h))
    display[:] = 0
    display[:, panel_width:] = frame_resized

    # === Run segmentation ===
    results = model(frame_resized, verbose=False)[0]
    blurred_frame = frame_resized.copy()

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for mask, cls_id in zip(masks, classes):
            if not selected[cls_id]:
                continue

            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask_resized > 0.5
            mask_3ch = np.stack([mask_bool]*3, axis=-1)

            blurred = cv2.GaussianBlur(frame_resized, BLUR_STRENGTH, 0)
            blurred_frame[mask_3ch] = blurred[mask_3ch]

    # === Draw UI Panel ===
    for i, name in enumerate(class_names):
        y = 40 + i * 35
        color = (0, 255, 0) if selected[i] else (70, 70, 70)
        cv2.rectangle(display, (20, y - 15), (20 + checkbox_size, y + checkbox_size - 15), color, thickness)
        if selected[i]:
            cv2.rectangle(display, (25, y - 10), (20 + checkbox_size - 5, y + checkbox_size - 20), color, -1)
        cv2.putText(display, name, (50, y), font, font_scale, color, thickness)

    cv2.putText(display, "Click to toggle blur", (20, 20), font, 0.7, (0, 255, 255), 2)
    cv2.putText(display, "Green = BLURRED", (20, h - 20), font, 0.6, (0, 255, 0), 2)

    # === Paste blurred result ===
    display[:, panel_width:] = blurred_frame

    cv2.imshow("Body Part Blur Selector (Click checkboxes!)", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done! Thanks for using the blur selector.")
