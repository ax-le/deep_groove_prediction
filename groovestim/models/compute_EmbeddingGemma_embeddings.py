"""
Compute the embeddings of the text using the EmbeddingGemma model.
The function embed_this_text returns the embeddings of the text.
"""

# %% Imports
import torch
from sentence_transformers import SentenceTransformer

from groovestim.models.get_embeddings import device_general
import groovestim.utils.default_path as default_path

# %% General model handling
def load_model(checkpoint_name="default", cache_dir_huggingface=default_path.cache_dir_huggingface):
    """
    Load the EmbeddingGemma HuggingFace model and tokenizer.
    """
    # Checkpoint for the model
    if checkpoint_name == "default":
        checkpoint_name = "google/embeddinggemma-300m"

    # Load the model and tokenizer
    model = SentenceTransformer(f"{cache_dir_huggingface}/{checkpoint_name}", tokenizer_kwargs={"model_max_length": 512})
    model.to(device_general)
    model.eval()

    return model, None, checkpoint_name

# %% Text embedding
def embed_this_text(text, model, tokenizer=None):
    """
    Embed the text using the EmbeddingGemma model.

    Parameters
    ----------
    text : str or list of str
        Input text(s) to embed.
    model : SentenceTransformer
        The EmbeddingGemma model.
    tokenizer : None
        The tokenizer is not used for EmbeddingGemma.
    """
    # Embed the text
    with torch.no_grad():
        text_embeddings = model.encode_document(text)

    return text_embeddings