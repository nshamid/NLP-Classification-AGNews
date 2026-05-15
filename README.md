## Dataset Usage

This project uses the **AI-MedLeafX** dataset:  
**“AI-MedLeafX: A Large-Scale Computer Vision Dataset for Medicinal Plant Diagnosis”** available on Mendeley Data.

### Dataset Source
Dataset Link:  
https://data.mendeley.com/datasets/zz7r5y4dc6/1

### Dataset Description
The dataset contains medicinal plant leaf images categorized into healthy and diseased classes. It includes both original and augmented images from multiple medicinal plant species and disease categories.

### Using the Dataset in Kaggle

The dataset was manually downloaded from Mendeley Data and uploaded into Kaggle to be used as notebook input.

### Steps to Use the Dataset

1. Download the dataset from the Mendeley Data link above.

2. Extract the dataset files on your local computer.

3. Open Kaggle and create a new notebook.

4. Upload the dataset manually:
   - Open the **Data** panel on the right side.
   - Click **Add Input**.
   - Select **Upload**.
   - Upload the extracted dataset folder/files.

5. After the upload process is complete, Kaggle will automatically mount the dataset as an input directory.

6. Set the dataset path in the notebook. Example:

```python
dataset_path = "/kaggle/input/your-dataset-folder-name"

```

Notes
All three models (Vision Transformer, ResNet50, and MobileNetV3) use the same dataset source.
The dataset folder structure should not be modified to ensure compatibility with the data loading pipeline.
GPU acceleration such as NVIDIA Tesla T4 is recommended for model training.
