#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from typing import List, MutableMapping, Union, Tuple
from pathlib import Path
import pathlib
import os
import numpy as np
import librosa


__docformat__ = 'reStructuredText'
__all__ = ['get_files_from_dir_with_pathlib',
           'get_audio_files_from_subdirs',
           'get_audio_file_data',
           'to_audio'
           ]


def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())


def get_audio_files_from_subdirs(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the audio files in the subdirectories of `dir_name`.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the audio files in the subdirectories `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return [Path(dirpath) / Path(filename) for dirpath, _, filenames in os.walk(dir_name)
            for filename in filenames
            if filename[-4:] == '.wav']


def get_audio_file_data(audio_file: Union[str, pathlib.Path], sr: int = None) \
        -> Tuple[np.ndarray, float]:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: Tuple[numpy.ndarray, float]
    """
    return librosa.core.load(path=audio_file, sr=sr, mono=True)


def write_pickle(
        file: str, data: MutableMapping[str, Union[np.ndarray, int]]) -> None:
    """Serializes the features and classes.

    :param file: File to dump the serialized features
    :type file: str
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    with open(file, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def split_into_sequences(spec: np.ndarray, seq_len: int) \
        -> List[np.ndarray]:
    """Splits the spectrum `spec` into sequences of length `seq_len`.

    :param spec: Spectrum to be split into sequences.
    :type spec: numpy.ndarray
    :param seq_len: Length of the sequences.
    :type seq_len: int
    :return: List of sequences.
    :rtype: list[numpy.ndarray]
    """
    # Discard if extra frames
    n_frames = spec.shape[1]
    n_sequences = n_frames // seq_len
    return np.array([spec[:, i * seq_len:(i + 1) * seq_len] for i in range(n_sequences)])


def move_files(files: List[pathlib.Path], dest_dir: pathlib.Path) -> None:
    """Moves the files to the `dest_dir`.

    :param files: Files to move.
    :type files: list[pathlib.Path]
    :param dest_dir: Destination directory.
    :type dest_dir: pathlib.Path
    """
    dest_dir.mkdir(exist_ok=True)
    for file in files:
        file.rename(dest_dir / file.name)

# EOF
