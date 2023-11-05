# AI-in-healthcare
## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [About the codes and datasets](#installation)

### General Info
***
This project is Motor Imagery Classification using Multitask Learning. Spectrograms (generated from time series EEG data from BCI Competition 2a dataset) are given as input to the model.

## Technologies
***
A list of technologies used within the project:
*Packages used can be accessed in the two requirements files in this folder.
## About the codes and datasets
***
Two files can be found in the folder - Main_Code.ipynb and Spectrogram_generation.py. The latter is the script which generates spectrograms from the BCIC dataset. The former is the main code which performs classification.
"spectrogram_generated_dataset" contains the spectrograms generated using Spectrogram_generation.py. The original EEG time series dataset is also provided in "BCICIV_2a" for those interested in re-generating the spectrograms.
Then, the Main_Code.ipynb can be run with spectrograms as input data.
