
from typing import List

import pathlib
import numpy as np
import librosa

from feature_extraction import extract_spectrogram
from utils import write_pickle


def add_noise(y: np.ndarray, snr_db: float = 20.0, mode: str = 'normalize') -> np.ndarray:
    # Add noise to the signal
    noise = np.random.normal(0, 1, len(y))
    signal_power = np.sum(y ** 2) / len(y)

    target_noise = signal_power / (10 ** (snr_db / 10))
    current_noise = np.sum(noise ** 2) / len(noise)

    noise *= np.sqrt(target_noise / current_noise)
    y_noised = y + noise

    return y_noised.astype(np.float32)

def add_pitch_shift(y: np.ndarray, n_steps: float, sr: int = 44100) -> np.ndarray:
    # Perform pitch shift to the input signal
    y_shift = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    
    return y_shift

def augment_data(audio_data: np.ndarray, 
                 file_name: str, 
                 label, 
                 target_dir: pathlib.Path, 
                 generate_noise = True, 
                 generate_pshifted = True) -> None:
    
    if generate_noise:
        y_noised = add_noise(audio_data)
        stft = extract_spectrogram(y_noised)
        
        features_and_metadata = {
            'features': stft,
            'label': label
        }

        write_pickle(
            str(target_dir / f'{file_name}_noised.pkl'),
            features_and_metadata)
                    

    if generate_pshifted:
        y_shifted = add_pitch_shift(audio_data, 4.0)
        stft = extract_spectrogram(y_shifted)

        features_and_metadata = {
            'features': stft,
            'label': label
        }

        write_pickle(
            str(target_dir / f'{file_name}_pshifted.pkl'),
            features_and_metadata)
