from typing import Any, Dict, Generator, Iterable, List

import torch
from torch.utils.data import IterableDataset


class AntibodyPLLDataset(IterableDataset):
    """
    A dataset class for parallelized antibody data processing.

    Args:
        antibody_sequences (list[str]): List of antibody sequences.

    Attributes:
        MASK_TOKEN_PLACEHOLDER (str): Placeholder for masked tokens.
        antibody_sequences (Iterable): Iterable of antibody sequences.

    Yields:
        dict: A dictionary containing masked antibody sequences, masked amino acid, and antibody index.

    ```
    from torch.utils.data import IterableDataset, DataLoader

    dataset = AntibodyPLLDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_anitbody_pll_batch)
    ```
    """

    MASK_TOKEN_PLACEHOLDER: "str" = "?"

    def __init__(self, antibody_sequences: list[str]):
        self.antibody_sequences: Iterable = antibody_sequences

    def __iter__(self) -> Generator[Dict[str, Any], Any, Any]:
        for i, antibody_seq in enumerate(self.antibody_sequences):
            for mask_i in range(len(antibody_seq)):
                masked_antibody_seq = antibody_seq[:mask_i] + self.MASK_TOKEN_PLACEHOLDER + antibody_seq[mask_i+1:]
                masked_aa = antibody_seq[mask_i]
                yield {"masked_antibodies": masked_antibody_seq, "masked_aa": masked_aa, "mask_idx": mask_i, "antibody_idx": i}


def collate_anitbody_pll_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to batch the dataset items.
    """
    return {
        'masked_antibodies': [item['masked_antibodies'] for item in batch],
        'masked_aa': [item['masked_aa'] for item in batch],
        'antibody_idx': [item['antibody_idx'] for item in batch],
        'mask_idx': [item['mask_idx'] for item in batch]
    }


