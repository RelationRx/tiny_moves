import hashlib
import re
import uuid

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def chunk_text_with_tokenizer(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 512,
    overlap: int = 50,
) -> list[str]:
    """
    Chunk text using a tokenizer.

    Uses the tokenizer to define the sequence length, with a specified overlap between chunks.

    Args:
        text (str): Full input string.
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer for the embedding model.
        overlap (int): Number of tokens to overlap between chunks.
        max_seq_length (int): Max token capacity of the model

    Returns:
        list[str]: List of decoded text chunks.

    """
    assert 0 <= overlap < max_seq_length

    encoding = tokenizer(
        text,
        return_overflowing_tokens=True,
        stride=overlap,
        max_length=max_seq_length,
        truncation=True,
    )

    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in encoding["input_ids"]]


def string_to_uuid(s: str) -> str:
    """
    Generate a UUID-like hash from any string.

    :param s: The input string.
    :return: A UUID object based on the SHA-256 hash of the string.
    """
    hash_bytes = hashlib.sha256(s.encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=hash_bytes))


# not in use for the moment but good to keep around
def simple_biomed_sentence_split(text: str) -> list[str]:
    """
    Sentence splitter using rule-based heuristics, better than naive period splitting.

    Args:
        text (str): Input text to split into sentences.

    Returns:
        list[str]: list of sentences.

    """
    # Common abbreviation and citation exceptions
    protected = [
        r"e\.g\.",
        r"i\.e\.",
        r"et al\.",
        r"Fig\.",
        r"Dr\.",
        r"vs\.",
        r"Inc\.",
        r"U\.S\.",
        r"[A-Z][a-z]{1,3}\.",  # journal abbrevs like "J. Biol. Chem."
    ]
    protected_pattern = "|".join(protected)

    # Placeholder replacement to prevent accidental splitting
    text = re.sub(f"({protected_pattern})", r"###\1###", text)

    # Primary split on punctuation followed by capital letter or number
    candidates = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", text)

    # Restore protected patterns
    sentences = [re.sub(r"###(.*?)###", r"\1", s).strip() for s in candidates if s.strip()]
    return sentences


# not in use for the moment but good to keep around
def chunk_text_by_sentences(
    text: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int = 512, truncate: bool = True
) -> list[str]:
    """
    Sentence-aware chunking using model tokenizer and token limit.

    Args:
        text (str): Full input string.
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer for the embedding model.
        max_tokens (int): Max tokens per chunk including special tokens.
        truncate (bool): If True, truncate long sentences to fit max_tokens.

    Returns:
        list[str]: List of decoded text chunks.

    """
    chunks: list[str] = []

    sentences = simple_biomed_sentence_split(text)

    for s in sentences:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            chunks.append(s)
        elif truncate:
            truncated = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
            chunks.append(truncated)
        else:
            chunks.append(s)

    return chunks
