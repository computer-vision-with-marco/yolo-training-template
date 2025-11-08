import streamlit as st
import os
import subprocess
import tempfile
import zipfile
import shutil
import sys

sys.path.append("scripts")
from preprocessing import YOLODataPreprocessor

# Page config
st.set_page_config(page_title="YOLO Training Template", page_icon="🚀", layout="wide")

# Sidebar navigation
st.sidebar.title("YOLO Training Template")
page = st.sidebar.radio(
    "Select Page", ["Training", "Inference", "Preprocessing", "Auto-labeling", "Export"]
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

