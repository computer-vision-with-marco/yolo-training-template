# YOLO Training Template

This repository provides a template for training YOLO models on any Kaggle dataset and performing inference. It includes scripts for command-line use and a notebook-style script for interactive environments.

## Files

- [`scripts/main.py`](scripts/main.py): Command-line script for training YOLO on a Kaggle dataset, with options for preprocessing and model export.
- [`scripts/inference.py`](scripts/inference.py): Command-line script for running inference with a trained model.
- [`notebooks/yolo_template.ipynb`](notebooks/yolo_template.ipynb): Notebook template to run train a YOLO model and test it.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md): Contributing guidelines.
- [`docs/streamlit_app.md`](docs/streamlit_app.md): Documentation for the Streamlit app.
- [`example_datasets.md`](example_datasets.md): List of example Kaggle datasets for testing.
- [`requirements.txt`](requirements.txt): Dependencies for the project.
- [`streamlit_app.py`](streamlit_app.py): Streamlit web app for interactive model training and inference.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. For training: Run `python scripts/main.py --dataset <kaggle-handle> --nc <num-classes> --names <class-names>`
3. For training with preprocessing: Run `python scripts/main.py --dataset <kaggle-handle> --nc <num-classes> --names <class-names> --preprocess`
4. For training with NCNN export: Run `python scripts/main.py --dataset <kaggle-handle> --nc <num-classes> --names <class-names> --export-ncnn`
4. For inference: Run `python scripts/inference.py --model <model-path> --input <image/video/webcam>`
5. For the Streamlit app: Run `streamlit run streamlit_app.py` (includes pages for training, inference, preprocessing, auto-labeling, and model export)
6. For non-technical setup: Please see [docs/QUICKSTART-GUIDE.md](docs/QUICKSTART-GUIDE.md)

## Data Preprocessing

The template includes optional data preprocessing capabilities for cleaning and augmenting your dataset:

- **Data Cleaning**: Remove corrupted images, validate annotations, check bounding box validity
- **Data Augmentation**: Apply various transformations (flips, rotations, color adjustments, noise) while properly updating YOLO labels

### Preprocessing Options

- `--preprocess`: Run cleaning and augmentation before training
- `--augment-only`: Only run augmentation (creates augmented dataset without training)
- `--preprocess-config`: Specify custom preprocessing configuration file (default: preprocessing_config.yaml)

### Configuration

Edit `preprocessing_config.yaml` to customize preprocessing behavior:

```yaml
cleaning:
  remove_corrupted_images: true
  validate_annotations: true
  check_bbox_validity: true
  min_bbox_size: 1
  max_bbox_size_ratio: 0.9

augmentation:
  enabled: true
  augment_factor: 2  # Number of augmented versions per image
  transforms:
    horizontal_flip:
      p: 0.5  # Probability
    # ... other transforms
```

## Model Export

The template supports exporting trained YOLO models to different formats for deployment on various platforms.

### Export Options

- `--export-ncnn`: Export the trained model to NCNN format after training (suitable for edge hardware like mobile devices)

### Usage

To train and export to NCNN:

```bash
python scripts/main.py --dataset <kaggle-handle> --nc <num-classes> --names <class-names> --export-ncnn
```

### Streamlit App

The Streamlit app includes an "Export" page where you can upload a trained model and export it to NCNN format for download.

### Notes on NCNN

- NCNN is optimized for mobile and embedded devices, providing efficient inference with low latency.
- Exported models include `.param` and `.bin` files in a directory (e.g., `model_ncnn/`).
- Use ONNX Runtime or NCNN's inference engine for deployment.

## Contributing

We welcome contributions! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on how to contribute, report issues, and run the notebook on Google Colab.

