from typing import Union
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoModelForMaskedLM, AutoTokenizer

from antibody_commonness.data.antibody_pll_dataset import (
    AntibodyPLLDataset,
    collate_anitbody_pll_batch,
)
from antibody_commonness.utils.common import get_num_special_tokens_prefix, tokenise_antibody_with_temp_mask
from antibody_commonness.utils.pll_aggregator import PseudoLogLikelihoodAggregator


def calculate_pll(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    dataset: AntibodyPLLDataset,
    batch_size: int,
    device: Union[str, torch.device],
    compile_model: bool = False
) -> PseudoLogLikelihoodAggregator:
    """
    Calculates the pseudo log-likelihood (PLL) for a given model, tokenizer, dataset, and batch size.

    Args:
        model (AutoModelForMaskedLM): The pre-trained language model for calculating PLL.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input data.
        dataset (AntibodyPLLDataset): The dataset containing the masked antibodies and their corresponding true amino acids.
        batch_size (int): The batch size for processing the data.
        device (torch.device or str): The device to run the calculations on.

    Returns:
        np.array: An array containing the average PLL values for each antibody in the dataset.
    """

    pll_storage = PseudoLogLikelihoodAggregator(10)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_anitbody_pll_batch,
        shuffle=False,
    )

    model.eval()
    model.to(device)
    
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Could not compile torch model, running uncompiled (compiled model runs faster)/n Error: {e}")
    
    # Iterate over the pll dataset
    for batch in tqdm(dataloader):
        tokens = [
            tokenise_antibody_with_temp_mask(
                masked_anti, dataset.MASK_TOKEN_PLACEHOLDER, tokenizer.mask_token
            )
            for masked_anti in batch["masked_antibodies"]
        ]
        true_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(batch["masked_aa"])
        ).int()

        inputs = tokenizer.batch_encode_plus(
            tokens,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True
        )

        # Some tokenisers like bert adds special tokens at the front and some (like t5) don't
        mask_ids = torch.tensor(batch["mask_idx"]).int() + get_num_special_tokens_prefix(inputs["special_tokens_mask"])

        # Not needed for model inference
        del inputs["special_tokens_mask"]

        inputs.to(device)

        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits.cpu()

        token_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
            torch.arange(logits.size(0)), mask_ids, true_ids
        ]
        pll_storage.update(token_log_probs.numpy(), batch["antibody_idx"])

    return pll_storage
