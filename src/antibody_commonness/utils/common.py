from typing import Dict
import torch


def space_amino_acid_tokens(antibody_seq: str):
    """
    Spaces out the amino acid tokens in the given antibody sequence.

    Args:
        antibody_seq (str): The antibody sequence to process.

    Returns:
        str: The antibody sequence with amino acid tokens spaced out.
    """
    return " ".join(antibody_seq)


def tokenise_antibody_with_temp_mask(antibody: str, temp_mask: str, mask_token: str):
    """
    Tokenizes an antibody string by replacing a temporary mask with a mask token.

    Args:
        antibody (str): The antibody string to be tokenized.
        temp_mask (str): The temporary mask to be replaced.
        mask_token (str): The mask token to replace the temporary mask with.

    Returns:
        str: The tokenized antibody string.
    """
    spaced_tokens = space_amino_acid_tokens(antibody)
    tokenised = spaced_tokens.replace(temp_mask, mask_token)
    return tokenised


def encode_anitbody(antibody_seq: str, tokeniser) -> torch.Tensor:
    """
    Encodes the given antibody sequence using the provided tokeniser.

    Args:
        antibody_seq (str): The list of amino acids representing the antibody sequence.
        tokeniser: The tokeniser object used to encode the sequence.

    Returns:
        torch.Tensor: The encoded sequence as a tensor.

    """
    prepared_antibody_seq = " ".join(antibody_seq)
    return tokeniser.encode_plus(prepared_antibody_seq, return_tensors="pt")


def encode_antibody_list(antibody_list: list, tokeniser) -> Dict[str, torch.Tensor]:
    """
    Encodes a list of antibody sequences using the provided tokeniser.

    Args:
        antibody_seqs (list): A list of amino acid sequences representing the antibody sequences.
        tokeniser: The tokeniser object used to encode the sequences.

    Returns:
        torch.Tensor: The encoded sequences as a tensor.

    """
    spaced_anibody_list = [" ".join(antibody) for antibody in antibody_list]
    return tokeniser.batch_encode_plus(spaced_anibody_list, 
        add_special_tokens=True, 
        pad_to_max_length=True, 
        return_tensors="pt",
        return_special_tokens_mask=True
    ) 


def get_num_special_tokens_prefix(special_token_tensor: torch.Tensor) -> int:
    """
    Determines the number of special tokens added to the front of a sequence by the tokenizer.

    Returns:
        int: The number of special tokens added to the front of the sequence.
    """
    # Find the first non-one element in each row
    first_zero = (special_token_tensor != 1).long().argmax(dim=1)
    
    # If a row is all ones, argmax will return 0, so we need to correct for this
    all_ones = (special_token_tensor == 1).all(dim=1)
    first_zero[all_ones] = special_token_tensor.shape[1]
    
    return first_zero