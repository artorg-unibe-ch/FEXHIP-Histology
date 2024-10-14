# FEXHIP-Histology
Repository used for the histology part of the FEXHIP project abbreviated FEXHIS
Linked to the article "Automatic Segmentation of Cortical Bone Microstructure: Application and Analysis of Three Proximal Femur Sites"

## Starting
- Install Anaconda
- Create virtual environment with `conda env create -f filename.yml`. "filename.yml" is "Linux_Requirements.yml" if working on a linux station or "WSL_Requirements.yml" if working with Windows subsystem for linux.
- Activate the virtual environment with conda activate FEXHIS
- Test installation by running "Test.ipynb" JupyterLab Notebook

## Overview
- Step 0: Draw lines to define quadrants (inferior, superior, anterior, and posterior) based on uCT scans and slice geometry
- Step 1: Design a model to perform automatic segmentation based on manual segmentations performed by multiple operators
- Step 2: Select regions of interest (ROIs) to perform automatic segmentation
- Step 3: Segment selected ROIs
- Step 4: Analysis of segmentation results (Extract data)
- Step 5: Perform statistics