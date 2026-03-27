"""
Compute the CLAP embeddings, for both text and audio data. 
The function embed_this_audio returns the embeddings of the audio data. 
The function embed_this_text returns the embeddings of the text data.
"""

# %% Imports
import torch
from transformers import ClapAudioModelWithProjection, ClapTextModelWithProjection, ClapProcessor, AutoTokenizer
import librosa

from groovestim.models.get_embeddings import device_general
import groovestim.utils.default_path as default_path
from groovestim.utils.io_utils import load_audio_file

# %% General model handling
def load_model(checkpoint_name="default", cache_dir_huggingface=default_path.cache_dir_huggingface, modality="audio"):
    """
    Load the CLAP HuggingFace model and processor.
    """
    #checkpoint_name # laion/clap-htsat-unfused # larger_clap_music # larger_clap_general # larger_clap_music_and_speech
    # Checkpoint for the model
    if checkpoint_name == "default":
        checkpoint_name = 'laion/larger_clap_general'

    match modality:
        case "audio":
            processor = ClapProcessor.from_pretrained(f"{cache_dir_huggingface}/{checkpoint_name}", cache_dir=cache_dir_huggingface)
            model = ClapAudioModelWithProjection.from_pretrained(f"{cache_dir_huggingface}/{checkpoint_name}", cache_dir=cache_dir_huggingface)
        case "text":
            processor = tokenizer = AutoTokenizer.from_pretrained(f"{cache_dir_huggingface}/{checkpoint_name}", cache_dir=cache_dir_huggingface)
            model = ClapTextModelWithProjection.from_pretrained(f"{cache_dir_huggingface}/{checkpoint_name}", cache_dir=cache_dir_huggingface)
        case _:
            raise ValueError(f"Modality {modality} not recognized. Please use 'audio' or 'text'.")

    model.to(device_general)
    model.eval()

    return model, processor, checkpoint_name

# %% Audio embedding
def embed_this_audio(audio_file_path, model, processor):
    """
    Embed an audio file using the CLAP model.
    """
    # Load and preprocess the audio (CLAP expects 48kHz)
    waveform, clap_sr = load_audio_file(audio_file_path, target_sr=48000)

    # Tokenize the audio for CLAP
    inputs_audio = processor(audio=waveform, sampling_rate=clap_sr, return_tensors="pt")

    # Cast the inputs for CPU or GPU inference        
    for key, value in inputs_audio.items():
        inputs_audio[key] = value.to(device_general)

    # Run the model 
    # with torch.inference_mode():
    outputs = model(**inputs_audio)
    audio_embedding = outputs.audio_embeds.detach().cpu().numpy()
    print(f"Shape: {audio_embedding.shape}")

    return audio_embedding
    
# %% Text embedding
def embed_this_text(text, model, processor):
    """
    Compute the text embedding using the CLAP model.
    """
    inputs_text = processor(text=text, return_tensors="pt", truncation=True, padding=True, truncation_side='right', model_max_length=512)

    # Cast the inputs for CPU or GPU inference
    for key, value in inputs_text.items():
        inputs_text[key] = value.to(device_general)

    # Run the model
    # with torch.inference_mode():
    outputs = model(**inputs_text)
    text_embedding = outputs.text_embeds.detach().cpu().numpy()
    print(f"Shape: {text_embedding.shape}")

    return text_embedding


# import zero_shot_tests.groove_sentences as groove_sentences
# def embed_groove_sentences(groove_keys, dictionary_name, checkpoint_name, verbose = False):
#     # Test for the zero shot
#     model_general, processor_general = load_model(checkpoint_name)

#     dictionary = groove_sentences.get_dictionary(dictionary_name)

#     text_embeddings = {}

#     for i in tqdm(range(len(groove_keys))):
#         groove_key = groove_keys[i]
#         sentences = dictionary[groove_key]
#         if verbose:
#             print(f'Processing {groove_key}, with the following sentences: {sentences}')
#         inputs_text = processor_general(text=sentences, return_tensors="pt", padding=True)

#         for key, value in inputs_text.items():
#             inputs_text[key] = value.to(device_general)

#         with torch.inference_mode():
#             outputs_text = model_general.get_text_features(**inputs_text)

#         text_embeddings[groove_key] = outputs_text.detach().cpu().numpy()

#     return text_embeddings