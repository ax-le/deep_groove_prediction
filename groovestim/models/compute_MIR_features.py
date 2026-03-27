"""
Compute "embeddings" from hand-crafted MIR features.

This is the baseline model: no deep learning, just a set of MIR
descriptors read from a pre-computed CSV.
"""

import pandas as pd

import groovestim.utils.default_path as default_path
from groovestim.utils.io_utils import get_song_ids

# MIR feature columns (numeric, no NaN)
_MIR_FEATURE_COLS = [
    'event_density', 'rms', 'rms_sd', 'spectral_flux',
    'band_1_flux', 'band_2_flux', 'band_3_flux', 'band_4_flux',
    'band_5_flux', 'band_6_flux', 'band_7_flux', 'band_8_flux',
    'band_9_flux', 'band_10_flux',
    'pulse_clarity', 'pulse_clarity_attack',
]


def load_MIR_features(audio_folder_path=default_path.path_audio_folder):
    """
    Return MIR-features for each song.

    Parameters
    ----------
    audio_folder_path : str
        Path to the audio folder (used only to derive song IDs).

    Returns
    -------
    song_embeddings : dict
        Mapping ``song_id → numpy array`` of shape ``(1, n_features)``.
    """
    song_ids = get_song_ids(audio_folder_path)

    mir_df = pd.read_csv(default_path.path_mir_features_csv, delimiter=",")

    song_embeddings = {}
    for song_id in song_ids:
        row = mir_df.loc[mir_df['id'] == song_id]
        song_embeddings[song_id] = row[_MIR_FEATURE_COLS].values

    return song_embeddings, song_ids
