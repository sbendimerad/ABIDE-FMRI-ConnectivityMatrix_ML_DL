# ABIDE-FMRI-ConnectivityMatrix_ML_DL

This project focuses on training and evaluating machine learning models on the ABIDE dataset using fMRI connectivity matrices. The project includes scripts for data loading, model training, evaluation, and saving the trained models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ABIDE-FMRI-ConnectivityMatrix_ML_DL.git
   cd ABIDE-FMRI-ConnectivityMatrix_ML_DL
    ```

2. Create a conda environment and activate it:
   ```sh
    conda create --name abide_env python=3.10
    conda activate abide_env
    ```

3. Install the required dependencies:
   ```sh
    pip install -r requirements.txt
    ```

4. To train and evaluate the models, launch the main.ipynb Jupyter notebook and follow the instructions provided in the notebook.

## Project Structure
   ```sh
ABIDE-FMRI-ConnectivityMatrix_ML_DL/
│
├── data/                           # Directory for storing data files
│   ├── abide.hdf5                  # HDF5 file containing the ABIDE dataset
│   └── phenotypes/                 # Directory for storing phenotypic data
│       └── Phenotypic_V1_0b_preprocessed1.csv
│
├── models/                         # Directory for storing trained models
│
├── [download_data.py](http://_vscodecontentref_/1)                # Script for downloading the data
├── [prepare_data.py](http://_vscodecontentref_/2)                 # Script for preparing the data
│
├── [classifier_ml.py](http://_vscodecontentref_/3)                # Script for training classifiers
├── [evaluate_classifier_ml.py](http://_vscodecontentref_/4)       # Script for evaluating classifiers
│
├── [main.ipynb](http://_vscodecontentref_/5)                      # Jupyter notebook for launching the project experimentations
│
├── [utils.py](http://_vscodecontentref_/6)                        # Utility functions for data loading and preprocessing
│
├── [requirements.txt](http://_vscodecontentref_/7)                # List of required dependencies
└── [README.md](http://_vscodecontentref_/8)                       # Project README file   
 ```
