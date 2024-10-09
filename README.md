# Multi-body self-salibration (CVT)
This repository contains the code of the Multi-body self calibration paper at BMVC 2022.

## Description of contents
This repository contains the following files and folders:
* **Dataset** <br>
    - optimization <br> Already extracted fundamental matrices for demos on Robust Initialization + Optimization
    - raw <br> Images from the datasets introduced in the original paper and the supplementary materials
    - sturm <br> Already extracted fundamental matrices for demos on Robust Initialization
    - synthetic <br> Data for synthetic demos
* **BMVC** <br>
    - src <br> Implementation of the algorithm introduced in the original paper
    - exp_pipeline_real.m <br> Demo for Robust Initialization + Optimization on real datasets
    - exp_pipeline_synthetic.m <br> Demo for Robust Initialization + Optimization on synthetic datasets
    - exp_sturm_real.m <br> Demo for Robust Initialization on real datasets
    - exp_sturm_synthetic.m <br> Demo for Robust Initialization on synthetic datasets
* **MotionSegmentation** <br>
Python code for motion segmentation with SURF feature descriptors and t-linkage


## Instructions
Run the scripts in the BMVC folder to test our Robust Initialization step and our Optimization.

Scripts were tested using MATLAB R2021b.

Download the Computer Vision Toolkit dependency from http://www.diegm.uniud.it/fusiello/demo/toolkit/ and add the ComputerVisionToolkit folder to the path before launching the scripts.
