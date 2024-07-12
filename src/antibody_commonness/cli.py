import click
import numpy as np
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer

from antibody_commonness.pseudo_likelihood import calculate_pll as calculate_pll_func
from antibody_commonness.data import AntibodyPLLDataset


@click.command()
@click.option("--model-name", default="Exscientia/IgBert_unpaired", help="Name of the pre-trained model to use")
@click.option("--batch-size", default=128, help="Batch size for processing")
@click.option(
    "--input-csv", required=True, help="Path to the input file containing sequences"
)
@click.option(
    "--input-column",
    default="sequence_alignment_aa",
    help="Name of the column containing antibody amino acid sequence",
)
@click.option("--device", default="cpu", help="What device to use for computation")
@click.option(
    "--output-file", required=True, help="Path to the output file for PLL scores"
)
def calculate_pll(
    model_name, batch_size, input_csv, input_column, device, output_file
):
    """Calculate Pseudo Log Likelihood for antibody sequences."""

    # Load model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    antibody_seqs = pd.read_csv(input_csv)[input_column].to_list()
    apll_dataset = AntibodyPLLDataset(antibody_seqs)
    # Calculate PLL
    pll_scores = calculate_pll_func(
        model, tokenizer, apll_dataset, batch_size=batch_size, device=device
    )

    # Store PLL scores
    np.save(
        output_file,
        np.concatenate((pll_scores.get_pll(), pll_scores.get_length_normalised_pll())),
    )


if __name__ == "__main__":
    calculate_pll()
