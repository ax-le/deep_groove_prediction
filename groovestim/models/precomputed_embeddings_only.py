"""
Return embeddings for both text and audio data, assuming they were already computed.
Used for models where the venv was different (typically model not loaded on HF).
"""

# %% Imports
import torch

from groovestim.models.get_embeddings import device_general
import groovestim.utils.default_path as default_path
from groovestim.utils.io_utils import load_audio_file

# %% General model handling
def load_model(checkpoint_name="default", cache_dir_huggingface=default_path.cache_dir_huggingface, modality="audio"):
    return None, None, checkpoint_name

# %% Audio embedding
def embed_this_audio(audio_file_path, model, processor):
    raise NotImplementedError("Placeholder function for precomputed embeddings. Cannot compute embeddings. TODEBUG.")
    
# %% Text embedding
def embed_this_text(text, model, processor):
    raise NotImplementedError("Placeholder function for precomputed embeddings. Cannot compute embeddings. TODEBUG.")