"""
This script is used to compute the audio embeddings for the MERT model.
"""

# %% Imports
import torch
from transformers import AutoModel
from einops import rearrange # To be installed

from groovestim.models.get_embeddings import device_general
import groovestim.utils.default_path as default_path
from groovestim.utils.io_utils import load_audio_file

# %% General model handling
def load_model(checkpoint_name="default", cache_dir_huggingface=default_path.cache_dir_huggingface, device=device_general):
    """
    Load the AudioMAE HuggingFace model and feature extractor.
    """
    # Only one AudioMAE model on HF
    model = AutoModel.from_pretrained(f"{cache_dir_huggingface}/hance-ai/audiomae", trust_remote_code=True, cache_dir=f"{cache_dir_huggingface}/tmp")
    processor = None

    model.to(device)
    model.eval()

    return model, processor, checkpoint_name

# %% Audio embedding
def embed_this_audio(song_path, model, processor):
    """
    Embed the song.
    """
    def embed_one_chunk(audio_chunk):
        """
        The audio will be splitted in chunks of 10s. HEnce, define the function per chunk. They will be averaged in the end.
        """
        if len(audio_chunk.shape) == 1:
            audio_chunk = audio_chunk[None,:] # Adding a channel dim, important for the model.
        melspec = model.encoder.waveform_to_melspec(audio_chunk)  # (length, n_freq_bins) = (1024, 128)
        melspec = melspec[None,None,:,:]  # (1, 1, length, n_freq_bins) = (1, 1, 1024, 128)
        z = model.encoder.forward_features(melspec.to("cuda")).cpu()  # (b, 1+n, d); d=768
        z = z[:,1:,:]  # (b n d); remove [CLS], the class token

        b, c, w, h = melspec.shape  # w: temporal dim; h:freq dim
        wprime = round(w / model.encoder.patch_embed.patch_size[0])  # width in the latent space
        hprime = round(h / model.encoder.patch_embed.patch_size[1])  # height in the latent space

        # reconstruct the temporal and freq dims
        z = rearrange(z, 'b (w h) d -> b d h w', h=hprime)  # (b d h' w')

        # remove the batch dim
        z = z[0].mean(-1).reshape(-1)  # (d h')

        return z  # (d h')

    waveform, sr = load_audio_file(song_path, target_sr=16000) # Found in the hf model
    wavs = torch.tensor(waveform).unsqueeze(0).to(device_general)
    # Separate the audio in 10 seconds chunks
    chunk_length = 10 * sr
    chunks = [wavs[:, i:i+chunk_length] for i in range(0, wavs.shape[1], chunk_length)]
    
    list_of_embeddings = []
    for chunk in chunks:
        if chunk.shape[1] < 16000: # Less than a second, avoid this chunk. Arbitrary.
            continue
        list_of_embeddings.append(embed_one_chunk(chunk))

    # Average the embeddings of the chunks
    final_emb = torch.mean(torch.stack(list_of_embeddings), dim=0, keepdim=True)
    return final_emb.detach().cpu().numpy()