# Project Overview

This repository contains the data, scripts, and methods for analyzing the results of two experiments for the paper *Probabilistic In-Plane Detection for Mid-Air Virtual Surface Interactions*. 

Please make sure the raw data file path before running the script. 
You can generate plots and 3D-finger trajectory using the script below

## Folder Structure

- **row_data**: Contains the raw data for the first experiment.
- **taskname_frame**: Includes the row numbers and labels indicating whether each data point is intentional or not for each participant in the first experiment.
- **extracted_datafile_taskname**: Contains extracted rows related to velocity and deviation calculations for the first experiment.
- **experiment_data**: Contains data and questionnaires for the second experiment.
- **Experiment_Result_CSV**: Contains post-processed data aggregated for the second experiment.

## Scripts and Functions

- **draw3D_TaskX.py**: Draws the finger trajectory for the three tasks in the experiment.
- **InPlaneVelocity.py**: Classifies data using tangential velocity.
- **velocity.py**: Uses Naive Bayes and perpendicular velocity for classification.
- **deviation.py**: Uses Naive Bayes and deviation for classification.
- **NaiveBayes_Task.py**: Classifies data for each task using Naive Bayes.
- **NaiveBayes_TwoFeature.py**: Uses two features for classification with Naive Bayes, based on the `final_aggregated_data_TwoFeature.csv`.
- **NaiveBayes.py**: Classifies data using only the z-axis velocity with Naive Bayes, based on the `final_aggregated_velocity_data.csv`.


