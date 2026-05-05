import streamlit as st
import os
import subprocess
import tempfile
import zipfile
import shutil
import sys
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

sys.path.append("scripts")
from preprocessing import YOLODataPreprocessor
from scripts.dataset_explorer import load_dataset, compute_dataset_stats, get_sample_images, draw_annotations

# Page config
st.set_page_config(page_title="YOLO Training Template", page_icon="🚀", layout="wide")

# Sidebar navigation
st.sidebar.title("YOLO Training Template")
page = st.sidebar.radio(
    "Select Page", ["Training", "Inference", "Preprocessing", "Dataset Explorer", "Auto-labeling", "Export"]
)

if page == "Training":
    st.title("🚀 YOLO Model Training")

    # Dataset source
    dataset_source = st.radio("Dataset Source", ["Kaggle Dataset", "Upload Dataset"])

    if dataset_source == "Kaggle Dataset":
        dataset_handle = st.text_input(
            "Kaggle Dataset Handle",
            placeholder="e.g., jocelyndumlao/multi-weather-pothole-detection-mwpd",
        )
        nc = st.number_input("Number of Classes", min_value=1, value=1)
        names = st.text_input(
            "Class Names (comma-separated)", placeholder="e.g., Potholes,Cracks"
        )
    else:
        uploaded_file = st.file_uploader("Upload Dataset (ZIP file)", type=["zip"])
        nc = st.number_input("Number of Classes", min_value=1, value=1)
        names = st.text_input(
            "Class Names (comma-separated)", placeholder="e.g., Potholes,Cracks"
        )

    # Training parameters
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", min_value=1, max_value=1000, value=60)
        imgsz = st.slider(
            "Image Size", min_value=32, max_value=2048, value=512, step=32
        )
        batch = st.slider("Batch Size", min_value=1, max_value=128, value=32)
    with col2:
        device = st.selectbox("Device", ["0", "cpu"], index=0)
        project = st.text_input("Project Directory", value="runs/train")
        name = st.text_input("Experiment Name", value="yolo_train")

    # Preprocessing options
    st.subheader("Preprocessing")
    preprocess = st.checkbox("Run Preprocessing (Cleaning + Augmentation)")
    augment_only = False
    if preprocess:
        augment_only = st.checkbox("Augmentation Only (Skip Training)")

    # Weights
    weights = st.text_input(
        "Pretrained Weights Path (optional)",
        placeholder="e.g., runs/train/yolo_train/weights/best.pt",
    )
    resume = st.checkbox("Resume Training")

    if st.button("Start Training"):
        if not names:
            st.error("Please provide class names.")
            st.stop()

        # Build command
        cmd = ["python", "scripts/main.py"]

        if dataset_source == "Kaggle Dataset":
            if not dataset_handle:
                st.error("Please provide a Kaggle dataset handle.")
                st.stop()
            cmd.extend(["--dataset", dataset_handle])
        else:
            # Handle uploaded dataset
            if not uploaded_file:
                st.error("Please upload a dataset ZIP file.")
                st.stop()
            # Extract to temp dir
            temp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(temp_dir, "dataset.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                extract_path = os.path.join(temp_dir, "dataset")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)
                # Detect structure (simplified)
                dataset_path = None
                for root, dirs, files in os.walk(extract_path):
                    if "images" in dirs and "labels" in dirs:
                        dataset_path = root
                        break
                if not dataset_path:
                    st.error(
                        "Could not detect dataset structure. Ensure it has train/val/test with images/labels subdirs."
                    )
                    st.stop()
                # Create yaml
                import yaml

                data_yaml = {
                    "path": dataset_path,
                    "train": "images",  # Assume flat structure for uploaded
                    "val": "images",
                    "test": "images",
                    "nc": nc,
                    "names": [n.strip() for n in names.split(",")],
                }
                yaml_path = os.path.join(temp_dir, "data.yaml")
                with open(yaml_path, "w") as f:
                    yaml.dump(data_yaml, f)
                # If preprocess, run preprocessing
                if preprocess:
                    from preprocessing import YOLODataPreprocessor

                    preprocessor = YOLODataPreprocessor()
                    images_dir = os.path.join(dataset_path, "images")
                    labels_dir = os.path.join(dataset_path, "labels")
                    if os.path.exists(images_dir) and os.path.exists(labels_dir):
                        stats = preprocessor.preprocess_dataset(images_dir, labels_dir)
                        st.info(f"Preprocessing stats: {stats}")
                    else:
                        st.warning(
                            "Images or labels dir not found, skipping preprocessing."
                        )
                # Train with yaml
                cmd.extend(["--dataset", "dummy"])  # Placeholder, since we have yaml
                # Actually, modify to use yaml directly, but for now, assume we set the path
                # This is hacky; better to modify main.py to accept yaml path
                st.info(
                    "Uploaded dataset training not fully integrated. Please use Kaggle for now."
                )
                st.stop()
            finally:
                shutil.rmtree(temp_dir)

        cmd.extend(
            [
                "--nc",
                str(nc),
                "--names",
                names,
                "--epochs",
                str(epochs),
                "--imgsz",
                str(imgsz),
                "--batch",
                str(batch),
                "--device",
                device,
                "--project",
                project,
                "--name",
                name,
            ]
        )
        if weights:
            cmd.extend(["--weights", weights])
        if resume:
            cmd.append("--resume")
        if preprocess and dataset_source == "Kaggle Dataset":
            cmd.append("--preprocess")
            if "augment_only" in locals() and augment_only:
                cmd.append("--augment-only")

        # Run command
        st.info("Starting training...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            st.success("Training completed successfully!")
            st.text_area("Output", result.stdout, height=300)
        else:
            st.error("Training failed!")
            st.text_area("Error", result.stderr, height=300)

elif page == "Inference":
    st.title("🔍 YOLO Inference")

    model_file = st.file_uploader("Upload Model Weights", type=["pt"])
    input_type = st.radio("Input Type", ["Image", "Video"])

    if input_type == "Image":
        image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    elif input_type == "Video":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    conf = st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

    if st.button("Run Inference"):
        if not model_file:
            st.error("Please upload a model file.")
            st.stop()

        # Save model to temp
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_model:
            temp_model.write(model_file.getvalue())
            model_path = temp_model.name

        cmd = [
            "python",
            "scripts/inference.py",
            "--model",
            model_path,
            "--conf",
            str(conf),
            "--no-display",
        ]

        if input_type == "Image" and image_file:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                temp_img.write(image_file.getvalue())
                cmd.extend(["--input", temp_img.name])
        elif input_type == "Video" and video_file:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_vid:
                temp_vid.write(video_file.getvalue())
                cmd.extend(["--input", temp_vid.name])
        else:
            st.error("Please provide input.")
            st.stop()

        # Run command
        st.info("Running inference...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            st.success("Inference completed!")
            st.text_area("Output", result.stdout, height=200)
            # Display result
            if input_type == "Image":
                if os.path.exists("inference_result.jpg"):
                    st.image("inference_result.jpg", caption="Inference Result")
            elif input_type == "Video":
                if (
                    os.path.exists("inference_result.mp4")
                    and os.path.getsize("inference_result.mp4") > 0
                ):
                    with open("inference_result.mp4", "rb") as f:
                        video_bytes = f.read()
                    if video_bytes:
                        st.video(video_bytes)
                    else:
                        st.error("Video file is empty.")
                else:
                    st.error("Video file not found or empty.")
        else:
            st.error("Inference failed!")
            st.text_area("Error", result.stderr, height=200)

elif page == "Preprocessing":
    st.title("🛠️ Data Preprocessing")

    uploaded_dataset = st.file_uploader(
        "Upload Dataset ZIP (containing images/ and labels/)", type=["zip"]
    )
    config_file = st.file_uploader("Preprocessing Config (optional)", type=["yaml"])

    if st.button("Run Preprocessing"):
        if not uploaded_dataset:
            st.error("Please upload a dataset ZIP file.")
            st.stop()

        # Extract and preprocess
        temp_dir = tempfile.mkdtemp()
        try:
            zip_path = os.path.join(temp_dir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_dataset.getvalue())
            extract_path = os.path.join(temp_dir, "dataset")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            # Find images and labels dirs
            images_dir = None
            labels_dir = None
            for root, dirs, files in os.walk(extract_path):
                if "images" in dirs:
                    images_dir = os.path.join(root, "images")
                if "labels" in dirs:
                    labels_dir = os.path.join(root, "labels")
                if images_dir and labels_dir:
                    break

            if not images_dir or not labels_dir:
                st.error("Could not find images/ and labels/ directories in the ZIP.")
                st.stop()

            # Load config if provided
            config_path = None
            if config_file:
                config_path = os.path.join(temp_dir, "config.yaml")
                with open(config_path, "wb") as f:
                    f.write(config_file.getvalue())

            # Run preprocessing
            preprocessor = YOLODataPreprocessor(config_path)
            stats = preprocessor.preprocess_dataset(images_dir, labels_dir)
            st.success("Preprocessing completed!")
            st.json(stats)

        finally:
            shutil.rmtree(temp_dir)

elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")

    mode = st.radio("Mode", ["Explore Dataset", "Manual Annotation"], help="Choose exploration or manual annotation")

    if mode == "Explore Dataset":
        st.markdown("### Dataset Input")
        input_method = st.radio(
            "Input Method",
            ["Upload ZIP", "Local Path"],
            help="Choose how to provide the dataset"
        )

        dataset_path = None
        class_names = None

        if input_method == "Upload ZIP":
            uploaded_file = st.file_uploader(
                "Upload Dataset ZIP",
                type=["zip"],
                help="Upload a ZIP file containing images/ and labels/ directories"
            )

            if uploaded_file:
                # Create a persistent temp directory for this session
                if 'dataset_temp_dir' not in st.session_state:
                    st.session_state.dataset_temp_dir = tempfile.mkdtemp(prefix="yolo_dataset_")

                temp_dir = st.session_state.dataset_temp_dir
                zip_path = os.path.join(temp_dir, uploaded_file.name)
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                extract_path = os.path.join(temp_dir, "extracted")
                os.makedirs(extract_path, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

                dataset_path = extract_path
        else:
            dataset_path_input = st.text_input(
                "Dataset Path",
                help="Path to local dataset directory"
            )
            if dataset_path_input and os.path.exists(dataset_path_input):
                dataset_path = dataset_path_input

        # Class names input
        class_names_input = st.text_input(
            "Class Names (optional)",
            placeholder="class1, class2, class3",
            help="Comma-separated list of class names for better visualization"
        )
        if class_names_input:
            class_names = [name.strip() for name in class_names_input.split(',')]

        if dataset_path:
            try:
                # Load dataset
                with st.spinner("Loading dataset..."):
                    dataset_splits = load_dataset(dataset_path)

                # Debug info
                st.markdown("### Debug Info")
                subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                st.write(f"Detected subdirectories: {subdirs}")
                st.write(f"Dataset splits found: {list(dataset_splits.keys())}")

                if dataset_splits:
                    st.success("Dataset loaded successfully!")

                    # Dataset statistics
                    st.markdown("### Dataset Statistics")
                    labels_dir = None  # Assume labels are in same structure
                    stats = compute_dataset_stats(dataset_splits, labels_dir, class_names)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Images", stats['total_images'])
                    with col2:
                        st.metric("Total Annotations", stats['bbox_stats']['total_bboxes'])
                    with col3:
                        st.metric("Avg BBoxes/Image", f"{stats['bbox_stats']['avg_bbox_per_image']:.2f}")

                    # Split information
                    st.markdown("#### Split Distribution")
                    for split, count in stats['split_counts'].items():
                        st.write(f"**{split.title()}:** {count} images")

                    # Class distribution
                    if stats['class_counts']:
                        st.markdown("#### Class Distribution")
                        if class_names:
                            for class_id, count in stats['class_counts'].items():
                                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                                st.write(f"**{class_name}:** {count}")
                        else:
                            for class_id, count in stats['class_counts'].items():
                                st.write(f"**Class {class_id}:** {count}")

                    # Sample images
                    st.markdown("### Sample Images")
                    sample_size = st.slider("Number of sample images", 1, 20, 5)
                    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.0)

                    if st.button("Generate Samples"):
                        with st.spinner("Generating sample images..."):
                            samples = get_sample_images(
                                dataset_splits,
                                sample_size=sample_size,
                                labels_dir=labels_dir,
                                class_names=class_names,
                                conf_threshold=conf_threshold
                            )

                        if samples:
                             for i, sample in enumerate(samples):
                                 st.markdown(f"**Sample {i+1}:** {os.path.basename(sample['path'])}")
                                 st.image(sample['image'], use_column_width=True)
                                 if sample['bboxes']:
                                    st.write(f"Found {len(sample['bboxes'])} annotations")
                        else:
                            st.warning("No sample images could be generated.")
                else:
                    st.error("No valid dataset structure found. Make sure your dataset contains 'images/' directories in train/val/test folders or at root level.")

            except Exception as e:
                st.error(f"Error exploring dataset: {str(e)}")
                logging.error(f"Dataset exploration error: {str(e)}")

    elif mode == "Manual Annotation":
        st.markdown("### Manual Annotation")
        st.write("Upload images and manually add/edit bounding box annotations.")

        # Upload images
        uploaded_images = st.file_uploader(
            "Upload Images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            help="Select multiple images to annotate"
        )

        class_names_input = st.text_input("Class Names (comma-separated)", placeholder="class1, class2")
        class_names = [name.strip() for name in class_names_input.split(',')] if class_names_input else []

        if uploaded_images and class_names:
            # Select image
            image_names = [img.name for img in uploaded_images]
            selected_image = st.selectbox("Select Image to Annotate", image_names)

            if selected_image:
                img = [img for img in uploaded_images if img.name == selected_image][0]
                image = Image.open(img)
                img_array = np.array(image)

                # Use session state for annotations
                session_key = f"annotations_{selected_image}"
                canvas_counter_key = f"canvas_counter_{selected_image}"
                if session_key not in st.session_state:
                    # Load existing annotations if any
                    label_file_name = f"{selected_image.rsplit('.', 1)[0]}.txt"
                    st.session_state[session_key] = []
                    if os.path.exists(label_file_name):
                        with open(label_file_name, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    st.session_state[session_key].append({
                                        'class_id': int(parts[0]),
                                        'x': float(parts[1]),
                                        'y': float(parts[2]),
                                        'w': float(parts[3]),
                                        'h': float(parts[4])
                                    })
                if canvas_counter_key not in st.session_state:
                    st.session_state[canvas_counter_key] = 0

                existing_annotations = st.session_state[session_key]

                # Convert annotations to draw_annotations format
                bboxes_for_draw = []
                height, width = img_array.shape[:2]
                for ann in existing_annotations:
                    x_center = ann['x'] * width
                    y_center = ann['y'] * height
                    w = ann['w'] * width
                    h = ann['h'] * height
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    bboxes_for_draw.append({
                        'class_id': ann['class_id'],
                        'bbox': (x1, y1, x2, y2),
                        'confidence': 1.0
                    })

                # Display image with current annotations
                annotated_img = draw_annotations(img_array, bboxes_for_draw, class_names)
                st.image(Image.fromarray(annotated_img), caption=f"Current annotations: {len(existing_annotations)} bboxes", use_column_width=True)

                # Add new bbox with mouse
                st.markdown("#### Add New Bounding Box")
                new_class = st.selectbox("Select Class for New Box", class_names, key="new_class")

                # Canvas for drawing
                canvas_key = f"canvas_{selected_image}_{st.session_state[canvas_counter_key]}"
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill with transparency
                    stroke_width=2,
                    stroke_color="#FF0000",  # Red stroke
                    background_color="#EEEEEE",
                    background_image=image,  # Original PIL image
                    update_streamlit=True,
                    height=image.height,
                    width=image.width,
                    drawing_mode="rect",
                    point_display_radius=0,
                    key=canvas_key,
                )

                if st.button("Add Drawn Box"):
                    if canvas_result and canvas_result.json_data and 'objects' in canvas_result.json_data:
                        objects = canvas_result.json_data['objects']
                        img_width, img_height = image.size  # PIL is (width, height)
                        added_count = 0
                        for obj in objects:
                            if obj['type'] == 'rect':
                                left = float(obj['left'])
                                top = float(obj['top'])
                                width = float(obj['width'])
                                height = float(obj['height'])
                                # Convert to normalized YOLO format
                                x_center = (left + width / 2) / img_width
                                y_center = (top + height / 2) / img_height
                                w_norm = width / img_width
                                h_norm = height / img_height
                                class_id = class_names.index(new_class)
                                st.session_state[session_key].append({
                                    'class_id': class_id,
                                    'x': x_center,
                                    'y': y_center,
                                    'w': w_norm,
                                    'h': h_norm
                                })
                                added_count += 1
                        if added_count > 0:
                            st.success(f"Added {added_count} bounding box(es)!")
                            st.session_state[canvas_counter_key] += 1
                            st.rerun()
                        else:
                            st.warning("No rectangles found in drawing.")
                    else:
                        st.warning("Please draw a rectangle on the image first.")

                # List current annotations
                st.markdown("#### Current Annotations")
                for i, ann in enumerate(existing_annotations):
                    class_name = class_names[ann['class_id']] if ann['class_id'] < len(class_names) else f"Class {ann['class_id']}"
                    st.write(f"{i+1}. {class_name}: x={ann['x']:.3f}, y={ann['y']:.3f}, w={ann['w']:.3f}, h={ann['h']:.3f}")
                    if st.button(f"Remove {i+1}", key=f"remove_{i}"):
                        st.session_state[session_key].pop(i)
                        st.rerun()

                if st.button("Save Annotations"):
                    # Save to txt file
                    label_file_name = f"{selected_image.rsplit('.', 1)[0]}.txt"
                    annotations_lines = [f"{ann['class_id']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}" for ann in st.session_state[session_key]]
                    with open(label_file_name, "w") as f:
                        f.write("\n".join(annotations_lines))
                    st.success(f"Annotations saved to {label_file_name} ({len(st.session_state[session_key])} bboxes)")

    # Clear dataset button
    if st.button("Clear Uploaded Dataset"):
        if 'dataset_temp_dir' in st.session_state and os.path.exists(st.session_state.dataset_temp_dir):
            shutil.rmtree(st.session_state.dataset_temp_dir)
            del st.session_state.dataset_temp_dir
        st.success("Uploaded dataset cleared.")
        st.rerun()

elif page == "Export":
    st.title("📤 Model Export")

    model_file = st.file_uploader("Upload Model Weights", type=["pt"])
    export_format = st.selectbox("Export Format", ["NCNN"])

    if st.button("Export Model"):
        if not model_file:
            st.error("Please upload a model file.")
            st.stop()

        # Save model to temp
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_model:
            temp_model.write(model_file.getvalue())
            model_path = temp_model.name

        # Run export
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            if export_format == "NCNN":
                exported_path = model.export(format='ncnn')
                st.success("Model exported to NCNN!")
                # Zip the output directory
                if os.path.exists(exported_path):
                    zip_path = "exported_model.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        if os.path.isdir(exported_path):
                            for root, dirs, files in os.walk(exported_path):
                                for file in files:
                                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), exported_path))
                        else:
                            zipf.write(exported_path, os.path.basename(exported_path))
                    with open(zip_path, "rb") as f:
                        st.download_button("Download Exported Model", f, file_name="model_ncnn.zip")
                else:
                    st.error("Export failed: output not found.")
        except Exception as e:
            st.error(f"Export failed: {str(e)}")

elif page == "Auto-labeling":
    st.title("🏷️ Auto-labeling with GroundingDINO")

    input_folder = st.text_input("Input Images Folder")
    text_prompt = st.text_input(
        "Text Prompt (comma-separated classes)", placeholder="e.g., car, person, dog"
    )
    output_path = st.text_input("Output Path", value="auto_labeled_dataset")

    if st.button("Run Auto-labeling"):
        if not input_folder or not text_prompt:
            st.error("Please provide input folder and text prompt.")
            st.stop()

        cmd = [
            "python",
            "autolabeling/auto-label.py",
            "--input_folder",
            input_folder,
            "--text_prompt",
            text_prompt,
            "--output_path",
            output_path,
        ]

        # Run command
        st.info("Running auto-labeling...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            st.success("Auto-labeling completed!")
            st.text_area("Output", result.stdout, height=200)
        else:
            st.error("Auto-labeling failed!")
            st.text_area("Error", result.stderr, height=200)

