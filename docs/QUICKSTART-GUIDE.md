# Quickstart Guide
This guide is meant for non-developers who want to get hands on experience training and performing inference with ML models using [this template](../notebooks/yolo_template.ipynb). 

This will not cover any engineering topics necessary to _construct_ this repo; rather it will focus on the lowest-friction way to _run_ this repo.

This guide makes the following assumptions:
1. You're a non-developer experimenter. 
2. You have a basic understanding of what "training" and "inference" mean for an ML model.
3. You have a Gmail and Google Drive account that is already setup. 
4. You'll use a Jupyter notebook in Google Colab (no prior knowledge needed).
5. [#4](https://github.com/mfranzon/yolo-training-template/issues/4) has been solved.

Follow the steps in the order listed.

## Jupyter Notebook in Colab
First, you'll need to open the Jupyter project notebook in Google Colab. 

[Jupyter](https://jupyter.org/) is an interactive local development environment that lets you build and run code in different 'steps' and see the results immediately. Its a learning tool that is commonly used on ML projects. Since we want to skip all the steps of setting up locally, we can open and run this notebook in Google Colab. 

[Google Colab](https://colab.research.google.com/) is a web-based dev environment where you can run jupyter notebooks and avoid any local setup. Additionally, on google colab you get free (limited) access to Google compute resources which makes running these scripts really fast. 

**To open the jupyter notebook in google colab, you'll need to:**
1. Navigate to the notebook file on github in your browser. You can do this by opening [`yolo_template.ipynb`](../notebooks/yolo_template.ipynb) in a new browser window.
2. In the browser's address bar, change the domain from `github.com` to `colab.research.google.com/github`. The new address should look something like: `colab.research.google.com/github/mfranzon/yolo-training-template/blob/main/notebooks/yolo_template.ipynb`
3. Hit **Enter**. The notebook should now open inside Google Colab.
4. Optionally, to save a copy of this notebook in your Google Drive, go to `File` then `Save a copy in Drive`

## Colab Setup
Before running any code, you need to configure your Colab environment to use the appropriate resources. Namely a GPU for training/inference, and your Google Drive as a storage location. 

- By enabling a GPU in Colab, you'll cut training time from hours to a few minutes. 
- By mounting Google Drive to Colab, you'll be able to save your model and store annotated videos. 

**To enable GPU in Colab, you'll need to:**
1. With your Colab window open, click on the `Runtime` menu in the top left corner.
2. In the resulting menu, select `Change runtime type` to open the "Change runtime type" modal.
3. In "Hardware accelerator" section of the modal, select `T4 GPU`.
4. Click `Save` to store the changes to your environment. 

**To mount Google Drive in Colab, you'll need to:**
1. With your Colab window open, click on the **folder icon** in the left sidebar. This will load a `Files` pane. (Sign in to Google if prompted.)
2. Click the **Mount Drive icon** (folder with Google Drive logo) in the top bar of the `Files` pane. Doing so should add a new cell to the notebook.
3. On the newly added cell, click `Run` (**play button icon**) and follow the prompts to mount your Google Drive.
4. Your Google Drive should now appear in the `Files` pane as a new folder titled `drive`.


## Run Training
Before running your first training, you first need to configure your training parameters. 

Doing so will allow you to define what training data to use and how many epochs to train for (along with other elements that are out of the scope of this guide). 

**To setup training parameters, you'll need to:**
1. Choose a dataset from [`example_datasets.md`](../example_datasets.md). Take note of its `Handle`, `Classes`, and `Names`
2. In Cell 5 of the notebook (beginning with `# Example regarding potholes`), update the following parameters with appropriate values from your chosen dataset:
    - `dataset_handle = 'your/dataset/handle/here'` Enter the `Handle` from your chosen dataset.
    - `nc = 1` Enter the `Classes` value from your chosen dataset.
    - `names = []` Enter the `Names` from your chosen dataset
3. Also in Cell 5 of the notebook, set the number of `epochs` you'd like to train for. More epochs take longer (~2m per epoch), but result in a better model.
4. Leave the remaining parameters unchanged (`imgsz`, `batch`, `device`, `project`)

**To run the first training, you'll need to:**
1. Click `Run` on Cell 1:  This will install necessary dependencies to the Colab environment
2. Click `Run` on Cell 2. This will import required libraries into Colab environment
3. Click `Run` on Cell 3. This defines the training functions and makes them available in the Colab environment. 
4. Click `Run` on Cell 5. This sets the training parameters in memory so they can be referenced by other functions.
5. Click `Run` on Cell 6. This calls functions in Cell 3 using parameters in Cell 5 to download the dataset, and run the training. Once it completes, your trained model is stored at `/content/runs/train/yolo_train/weights/best.pt`

## Run inference
Now that you've trained your model you can use the trained model to detect objects in new images/videos. To do this you'll first need to upload images to run inference on, setup the inference parameters, and finally Run the inference. 

**To upload new images or videos, you'll need to:**
1. Create a dedicated "Inputs" folder in Google Drive for new images or videos. These will be the _inputs_ to the inference. 
    - Images must be of the following file types: `.jpg`, `.png`, `.jpeg`, `.bmp`, or `.tiff`
    - Videos must be of the following file types: `.mp4`, `.avi`, `.mov`, `.mkv`, or `.flv`
2. Create a dedicated "Outputs" folder in Google Drive for annotated images or videos. These will be the _outputs_ of the inference. 
3. In the left sidebar click the **folder icon** to open the `Files` pane.
4. In the `Files` pane find the `drive` folder representing your Google Drive. Ensure your inputs and outputs folders are visible.
    - If the `drive` folder is not visible you may need to go back to [Colab Setup](#colab-setup) to mount your Google Drive

**To setup inference parameters, you'll need to:**
1. From the `Files` pane, open the `drive` folder and browse to your "Inputs" folder containing new image or video files.
2. Choose an image or video and right click to open the menu. From the right-click menu, select 'Copy path'. Paste this value in Cell 7 as `input_source`
3. Navigate to your "outputs" folder, right click, select 'Copy path'. Paste this value in Cell 7 as `output_path`
4. Append a file name to the end of the `output_path`. This will be the name of your file, and must contain a supported file extension (like `.jpg` or `.mp4`)
5. Leave the remaining parameters unchanged (`model_path`, `conf_thresh`)


**To run inference on an image or video, you'll need to:**
1. Click `Run` on Cell 4. This defines the inference functions and makes them available in the Colab environment. 
2. Click `Run` on Cell 7. This sets the inference parameters in memory so they can be referenced by other functions.
2. Click `Run` on Cell 8. This calls functions in Cell 4 using parameters in Cell 7 to perform inference on selected files. Once it completes, your annotated files are stored in the Google Drive location you specified in Cell 7

To run subsequent inference on other files, you'll just need to update the parameters in Cell 7 and rerun Cells 7 & 8. 

---

Thanks for following this quickstart guide. I hope it helped you get started on your ML journey. 
