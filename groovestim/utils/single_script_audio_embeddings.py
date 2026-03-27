
import pathlib
import torch
import numpy as np
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T


model_name = #TODO
checkpoint_name = #TODO

cache_dir_save_embeddings = '/Brain/private/a23marmo/projects/groove_study/cache/embeddings'
audio_folder_path = "/Brain/private/a23marmo/projects/groove_study/data/songs"
cache_dir_huggingface = f'/Brain/public/models/{model_name}'
model_path = f'/Brain/public/models/{checkpoint_name}'

# Device for inference (shared across all embedding modules)
device_general = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.flac', '.ogg', '.aac', '.wma'}

def load_model(checkpoint_name="default", cache_dir_huggingface=cache_dir_huggingface):
    # TODO
    return model, processor, checkpoint_name


def embed_this_audio(song_path, model, processor):
    # model = model.to(device).eval()
    waveform, sr = load_audio_file(song_path, target_sr=16000)
    wavs = torch.tensor(waveform).unsqueeze(0).to(device_general)

    # TODO

    return embedding.detach().cpu().numpy()


def extract_song_id(file_path):
    """
    Extract a song ID from the file path.

    Uses the stem (filename without extension) and takes the ``[-12:-1]``
    slice to match the ID format used in the MIR-features dataframe.
    """
    stem = pathlib.Path(file_path).stem
    return stem[-12:-1]

# Cache for resamplers to avoid recreating them for each audio file
_resamplers = {}

def load_audio_file(audio_file_path, target_sr=48000):
    """
    Load an audio file, resample to *target_sr* if needed, and convert to mono.

    Parameters
    ----------
    audio_file_path : str or pathlib.Path
        Path to the audio file.
    target_sr : int
        Target sampling rate.  Defaults to 48000 (CLAP).  Use 24000 for MERT.

    Returns
    -------
    waveform : torch.Tensor
        1-D waveform tensor (mono).
    target_sr : int
        The sampling rate of the returned waveform.
    """
    waveform, base_sr = torchaudio.load(audio_file_path)

    # Resample if needed (cached resampler per source / target rate pair)
    if base_sr != target_sr:
        key = (base_sr, target_sr)
        if key not in _resamplers:
            _resamplers[key] = T.Resample(base_sr, target_sr)
        waveform = _resamplers[key](waveform)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=False)
    else:
        waveform = waveform.squeeze()

    return waveform, target_sr


def compute_or_load_audio_embeddings(
    folder_path,
    model_name,
    model,
    processor,
    embed_fn,
    checkpoint_name,
    cache_dir=cache_dir_save_embeddings,
    verbose=True,
):
    """
    Generic audio-embedding computation with on-disk caching.

    Loops over all audio files in *folder_path*, computes embeddings using
    the provided *embed_fn*, and caches them as ``.npz`` files.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Directory containing audio files.
    model_name : str
        Model name (e.g. ``"CLAP"``, ``"MERT"``), used for the cache sub-dir.
    model : torch.nn.Module
        The pre-loaded model.
    processor : object
        The pre-loaded processor / tokenizer.
    embed_fn : callable
        ``embed_fn(song_path, model, processor) -> embedding``.
    checkpoint_name : str
        Checkpoint identifier, used for the cache sub-dir.
    cache_dir : str or pathlib.Path
        Root cache directory for saved embeddings.

    Returns
    -------
    audio_embeddings : dict
        Mapping ``song_id → numpy embedding array``.
    song_ids_list : list
        List of song IDs extracted from audio filenames in *folder_path*.
    """
    folder = pathlib.Path(folder_path)

    # Filter to audio files only
    song_list = sorted([
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ])

    audio_embeddings = {}

    # Path of the folder where embeddings will be saved once computed
    dir_save_path = (
        pathlib.Path(cache_dir) / 'audio_embeddings' / model_name / checkpoint_name
    )

    # Loop on the songs
    for song_name in tqdm(song_list, desc=f"Computing {model_name} embeddings"):
        song_path = folder / song_name
        song_id = extract_song_id(song_name)

        # Name of the file where the embedding will be saved
        embedding_file = dir_save_path / f'audio_embedding_{song_id}.npz'

        # Check if the embedding has already been computed
        if embedding_file.exists():
            this_embedding = np.load(embedding_file, allow_pickle=True)
            assert this_embedding['song_id'] == song_id
            this_audio_embedding = this_embedding['audio_embedding']
            if verbose:
                print(f"  ✓ Loaded audio embedding for song {song_id} from cache")
        else:
            # Compute the embedding
            this_audio_embedding = embed_fn(str(song_path), model, processor)

            # Convert to numpy if it's a tensor
            if isinstance(this_audio_embedding, torch.Tensor):
                this_audio_embedding = this_audio_embedding.detach().cpu().numpy()
            
            if this_audio_embedding.ndim > 2:
                this_audio_embedding = this_audio_embedding.reshape(1,-1)
            elif this_audio_embedding.ndim == 1:
                this_audio_embedding = this_audio_embedding.reshape(1,-1)
            
            dir_save_path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                embedding_file,
                song_id=song_id,
                audio_embedding=this_audio_embedding,
            )

            if verbose:
                print(f"  ✓ Computed audio embeddings for song {song_id}")
        
        audio_embeddings[song_id] = this_audio_embedding

    song_ids_list = [extract_song_id(name) for name in song_list]
    return audio_embeddings, song_ids_list

if __name__ == "__main__":

    model, processor, checkpoint = load_model(checkpoint_name)

    song_audio_embeddings, song_list = compute_or_load_audio_embeddings(
                folder_path=audio_folder_path,
                model_name=model_name,
                model=model,
                processor=None,
                embed_fn=embed_this_audio,
                checkpoint_name=checkpoint,
            )
