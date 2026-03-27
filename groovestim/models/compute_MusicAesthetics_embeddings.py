"""
This script is used to compute the audio embeddings for the MERT model.
"""

# %% Imports
import numpy as np
import torch
from transformers import WhisperModel, WhisperProcessor

from groovestim.models.get_embeddings import device_general
from groovestim.models.additional_files.Music_Aesthetics_model_architecture import MusicAestheticsModel # Downloaded from the HF repo (model_architecture.py)
import groovestim.utils.default_path as default_path
from groovestim.utils.io_utils import load_audio_file

# %% General model handling
def load_model(checkpoint_name="default", cache_dir_huggingface=default_path.cache_dir_huggingface, device=device_general):
    """
    Load the Music Aesthetics HuggingFace model and feature extractor.
    """
    # Download and load weights
    # 1. The Audio Encoder (The Music Whisper Model)
    WHISPER_REPO = f"{cache_dir_huggingface}/laion/music-whisper"
    # 2. This Aesthetics Model
    AESTHETICS_REPO = f"{cache_dir_huggingface}/laion/music-aesthetics"

    processor = WhisperProcessor.from_pretrained(WHISPER_REPO)
    # We only need the encoder part of Whisper
    whisper = WhisperModel.from_pretrained(WHISPER_REPO).encoder.to(device)
    whisper.eval()

    # Initialize the architecture
    model = MusicAestheticsModel().to(device)
    
    # Download and load weights
    # 1. Load Shared Bottleneck
    bt_path = f"{AESTHETICS_REPO}/stage1_bottleneck.pt" # hf_hub_download(repo_id=AESTHETICS_REPO, filename="stage1_bottleneck.pt")
    state_dict = torch.load(bt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # 2. Load Expert Heads
    for metric in model.metrics:
        head_path = f"{AESTHETICS_REPO}/expert_{metric}.pt" # hf_hub_download(repo_id=AESTHETICS_REPO, filename=f"expert_{metric}.pt")
        model.heads[metric].load_state_dict(torch.load(head_path, map_location=device))
    
    model.eval()
    return (model, whisper), processor, checkpoint_name

# %% Audio embedding
def embed_this_audio(audio_file_path, model, processor, device=device_general):
    """
    Embed an audio file using the MERT model.
    """
    aesthetic_model, whisper = model

    audio, sr = load_audio_file(audio_file_path, target_sr=16000)

    target_len = sr * 30 # Audio should be 30s long
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        audio = audio[start : start + target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))
        
    # 2. Extract Whisper Features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        # Get last hidden state from encoder
        outputs = whisper(inputs.input_features.to(device))
        last_hidden = outputs.last_hidden_state # (1, 1500, 768)
        
        # 3. Apply Feature Pooling (Expert Model Logic)
        # Reshape to (1, 10 segments, 150 frames, 768 dim)
        feats = last_hidden.view(1, 10, 150, 768)
        
        mean_pool = torch.mean(feats, dim=2)        
        max_pool = torch.max(feats, dim=2).values   
        min_pool = torch.min(feats, dim=2).values   
        
        # Concat -> Flatten -> (23040,)
        concat = torch.cat([mean_pool, max_pool, min_pool], dim=2)
        embedding = concat.view(-1).unsqueeze(0) # Add batch dim

    # 4. Predict Scores
    with torch.no_grad():
        outputs = aesthetic_model(embedding)
    
    results = {k: v.item() for k, v in outputs.items()}
    
    # Calculate Average Global Score
    avg_score = sum(results.values()) / len(results)
    results["Overall_Aesthetics"] = avg_score

    emb = torch.tensor(list(results.values())).unsqueeze(0).cpu().numpy()
    
    return emb