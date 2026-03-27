"""
Compute the embeddings of the text using the RoBERTa model.
The function embed_this_text returns the embeddings of the text.
"""

# %% Imports
import torch
from transformers import AutoTokenizer, RobertaModel

from groovestim.models.get_embeddings import device_general
import groovestim.utils.default_path as default_path

# %% General model handling
def load_model(checkpoint_name="default", cache_dir_huggingface=default_path.cache_dir_huggingface):
    """
    Load the RoBERTa HuggingFace model and tokenizer.
    """
    # Checkpoint for the model
    if checkpoint_name == "default":
        checkpoint_name = "FacebookAI/roberta-base"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{cache_dir_huggingface}/{checkpoint_name}")
    model = RobertaModel.from_pretrained(f"{cache_dir_huggingface}/{checkpoint_name}", add_pooling_layer=False)

    model.to(device_general)
    model.eval()

    return model, tokenizer, checkpoint_name

# %% Text embedding
def embed_this_text(text, model, tokenizer, return_cls=True):
    """
    Embed the text using the RoBERTa model.

    Parameters
    ----------
    text : str or list of str
        Input text(s) to embed.
    model : RobertaModel
        The RoBERTa model.
    tokenizer : AutoTokenizer
        The tokenizer.
    return_cls : bool
        If True, return only the CLS token embedding. Otherwise return all token embeddings.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, max_length=512, truncation=True)

    # Cast the inputs for CPU or GPU inference
    for key, value in inputs.items():
        inputs[key] = value.to(device_general)

    # Compute the embeddings
    with torch.inference_mode():
        outputs = model(**inputs)

    # Return the CLS token only
    if return_cls:
        return outputs.last_hidden_state[:, 0, :]
    else:
        return outputs.last_hidden_state