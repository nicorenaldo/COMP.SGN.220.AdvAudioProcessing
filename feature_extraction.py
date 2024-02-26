from typing import Optional

import numpy as np
import librosa


def extract_spectrogram(audio_signal: np.ndarray,
                        n_fft: Optional[int] = 2048,
                        hop_length: Optional[int] = 1024,
                        window: Optional[str] = 'hamm') \
        -> np.ndarray:
    """Extracts and returns the magnitude spectrogram from the `audio_signal` signal.

    :param audio_signal: Audio signal.
    :type audio_signal: numpy.ndarray
    :param n_fft: STFT window length (in samples), defaults to 2048.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 1024.
    :type hop_length: Optional[int]
    :param window: Window type, defaults 'hamm'.
    :type window: Optional[str]
    :return: Magnitude of the short-time Fourier transform of the audio signal [shape=(n_frames, n_bins)].
    :rtype: numpy.ndarray
    """

    return np.abs(librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length, window=window))
