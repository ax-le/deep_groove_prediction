"""
This script is used to compute the audio embeddings for the MERT model.
"""

# %% Imports
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel

from groovestim.models.get_embeddings import device_general
import groovestim.utils.default_path as default_path
from groovestim.utils.io_utils import load_audio_file

# %% General model handling
def load_model(checkpoint_name="default", cache_dir_huggingface=default_path.cache_dir_huggingface, device=device_general):
    """
    Load the MERT HuggingFace model and feature extractor.
    """
    #checkpoint_name # MERT-v1-330M # MERT-v1-95M # MERT-v0-public # MERT-v0
    # Checkpoint for the model
    if checkpoint_name == "default":
        checkpoint_name = "MERT-v1-95M"

    model = AutoModel.from_pretrained(f"{cache_dir_huggingface}/m-a-p/{checkpoint_name}", trust_remote_code=True, cache_dir=cache_dir_huggingface)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        f"{cache_dir_huggingface}/m-a-p/{checkpoint_name}", trust_remote_code=True, cache_dir=cache_dir_huggingface
    )

    model.to(device)
    model.eval()

    return model, processor, checkpoint_name

# %% Audio embedding
def embed_this_audio(audio_file_path, model, processor, device=device_general):
    """
    Embed an audio file using the MERT model.
    """
    # Load and preprocess the audio (MERT expects the processor's sampling rate)
    resample_rate = processor.sampling_rate
    waveform, _ = load_audio_file(audio_file_path, target_sr=resample_rate)

    # Tokenize the audio
    inputs = processor(waveform, sampling_rate=resample_rate, return_tensors="pt")

    # Cast the inputs for CPU or GPU inference  
    for key, value in inputs.items():
        inputs[key] = value.to(device)

    # Run the model
    with torch.no_grad():
        model_output = model(**inputs, output_hidden_states=True)

    # Time-averaged hidden states from all layers: (n_layers, D)
    all_layer_hidden_states = torch.stack(model_output.hidden_states).squeeze()
    time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
    return time_reduced_hidden_states