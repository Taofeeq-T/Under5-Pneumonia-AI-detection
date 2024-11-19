# Under5-Pneumonia-AI-detection

# Detection of Pneumonia in Children using AI

This repository contains code for detecting pneumonia in children under-5 years through chest radiographs (CXR) using a VGG19-based deep learning model. The project focuses on addressing pneumonia detection in resource-limited settings, particularly in Nigeria.

This codebase supports the study **"Detection of Pneumonia in Children through Chest Radiographs using Artificial Intelligence in a Low-Resource Setting: A Pilot Study"**. The preprint will be provided soon with detailed insights on performance and methodology.

## Repository Structure
- **data/**: The training and internal test data from UC San Diego associated with the study is found at *http://dx.doi.org/10.17632/rscbjbr9sj.2* The external test data collected in Nigeria is found at *https://doi.org/10.5281/zenodo.14185822*
- **src/**: Source code for data preprocessing, model building, training, fine-tuning, and evaluation.
  - **preprocessing/**: Scripts for dataset preparation.
  - **model/**: Scripts for model architecture, training, fine-tuning, and hyperparameter tuning.
  - **visualization/**: Scripts for generating plots and confusion matrices.


## Instructions for Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Taofeeq-T/Under5-Pneumonia-AI-detection
   
   
