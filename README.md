### Final Project WOrk

This is a final project works for COMP.SGN.220 Advanced Audio Processing class

### Contributor
- Nico Renaldo
- Umair Raihan

### Description

This projects involves audio analysis and classification for music genre classification. The data used are [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/code) that are publicly available on Kaggle. The classification are done using a NN model with GRU and CNN.

### Structure of the Files

- `temp.ipynb` : 
- `feature_extraction.py` : Functions for extracting
- `data_generation.py` : Converting the raw audio files to chunks of extracted features with the corresponding label in pickle format.
- `dataset.py` : Load the extracted features to PyTorch Dataset classes
- `model.py` : NN Model
- `utils.py` : Misc Functions