import numpy as np
import torch
from typing import Any, List


# Max tokens for our embedding model.  This code is really designed for the gte-*
# series, e.g.: https://huggingface.co/thenlper/gte-small
# but could in theory be generated to work with other models I suspect.
MAX_LENGTH = 512


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def calculate_fragment_embeddings(model: Any, fragment: str) -> List[float]:
    """Calculate vector embeddings for a single input fragment, which is smaller than the
    max model length.
    """
    with torch.no_grad():
        return model.encode(fragment, normalize_embeddings=True)


def calculate_embeddings(input_text: str, model: Any, tokenizer: Any) -> List[float]:
    """Calculate the vector embeddings for the specified input text.

    1. split the text based on the model's max sequence length
    2. calculate the embeddings for each chunk
    3. calculate the average embedding across all chunks
    """

    # Tokenize the input, and convert tokens into chunks based on max model size.
    inputs = tokenizer(input_text, padding=False, truncation=False, return_tensors="pt")
    chunks = [
        torch.Tensor(inputs["input_ids"][0][i : i + MAX_LENGTH].tolist()).int()
        for i in range(0, len(inputs["input_ids"][0]), MAX_LENGTH)
    ]
    fragments = [tokenizer.decode(chunk) for chunk in chunks]

    # Now, calculate embeddings for each fragment.
    all_embeddings = []
    lengths = []
    for fragment in fragments:
        lengths.append(len(fragment))
        all_embeddings.append(calculate_fragment_embeddings(model, fragment))

    # Finally, calculate the average across all fragments.
    embeddings = np.average(all_embeddings, axis=0, weights=lengths)
    return embeddings / np.linalg.norm(embeddings)
