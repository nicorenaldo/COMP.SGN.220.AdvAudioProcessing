### Final Project WOrk

This is a final project works for COMP.SGN.220 Advanced Audio Processing class

### Contributor
- Nico Renaldo
- Umair Raihan

### Description

This projects involves audio analysis and classification for music genre classification. The data used are [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/code) that are publicly available on Kaggle. The classification are done using a NN model with CNN for feature extraction and LSTM for the temporal dynamics in the spectrogram.

### Structure of the Files

- `training.ipynb` : Notebook for training the models
- `data_generation.py` : Converting the raw audio files to chunks of extracted features with the corresponding label in pickle format.
- `dataset.py` : Load the extracted features to PyTorch Dataset classes
- `feature_extraction.py` : Utility functions for extracting features from the raw audio file
- `model.py` : NN Model for classification
- `utils.py` : Misc Functions

### Notes for Running the Project

The dataset included inside the repositories are only a fraction / samples of the full dataset. Before training the model, make sure to extract the features and split it to train-val-test sets using the scripts inside `data_generation.py`.
