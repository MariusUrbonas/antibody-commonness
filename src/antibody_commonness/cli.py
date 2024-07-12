from functools import reduce
import click
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForMaskedLM,
    BertForMaskedLM,
    T5EncoderModel,
    AutoTokenizer,
    T5Tokenizer,
)

from antibody_commonness.pseudo_likelihood import calculate_pll as calculate_pll_func
from antibody_commonness.data import AntibodyPLLDataset


@click.group()
def cli():
    """Antibody Commonness CLI application."""
    pass


@cli.group()
def calculate_pll_unpaired():
    """Calculate Pseudo Log Likelihood for unpaired antibody sequences."""
    pass


def common_options(func):
    """Common options for both IgBERT and IgT5 commands."""
    options = [
        click.option("--batch-size", default=128, help="Batch size for processing"),
        click.option(
            "--input-csv",
            required=True,
            help="Path to the input file containing sequences",
        ),
        click.option(
            "--input-column",
            default="sequence_alignment_aa",
            help="Name of the column containing antibody amino acid sequence",
        ),
        click.option(
            "--device", default="cpu", help="What device to use for computation"
        ),
        click.option(
            "--output-file",
            required=True,
            help="Path to the output file for PLL scores",
        ),
    ]
    return reduce(lambda x, opt: opt(x), options, func)


@calculate_pll_unpaired.command()
@common_options
def igbert(**kwargs):
    """Calculate PLL using IgBERT model."""
    model = BertForMaskedLM.from_pretrained("Exscientia/IgBert_unpaired")
    tokenizer = AutoTokenizer.from_pretrained("Exscientia/IgBert_unpaired")
    run_calculate_pll(model=model, tokenizer=tokenizer, **kwargs)


@calculate_pll_unpaired.command()
@common_options
def igt5(**kwargs):
    """Calculate PLL using IgT5 model."""
    model = T5EncoderModel.from_pretrained("Exscientia/IgT5_unpaired")
    tokenizer = T5Tokenizer.from_pretrained("Exscientia/IgT5_unpaired")
    run_calculate_pll(model=model, tokenizer=tokenizer, **kwargs)


@calculate_pll_unpaired.command()
@common_options
@click.option("--model-name", required=True, help="Model to load with HF autodmodel")
def automodel(model_name, **kwargs):
    """Calculate PLL using an AutoModel model."""
    click.echo(
        "You are using an AudoModel for ppl calculation instead of one of the defined models, behaviour is not guaranteed to be correct."
    )
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    run_calculate_pll(model=model, tokenizer=tokenizer, **kwargs)


def run_calculate_pll(
    model, tokenizer, batch_size, input_csv, input_column, device, output_file
):
    """Calculate Pseudo Log Likelihood for antibody sequences."""

    antibody_seqs = pd.read_csv(input_csv)[input_column].to_list()
    apll_dataset = AntibodyPLLDataset(antibody_seqs)
    # Calculate PLL
    pll_scores = calculate_pll_func(
        model, tokenizer, apll_dataset, batch_size=batch_size, device=device
    )

    pll = pll_scores.get_pll()
    pll_normalised = pll_scores.get_length_normalised_pll()
    # Store PLL scores
    np.save(
        output_file,
        np.concatenate(
            (np.expand_dims(pll, axis=1), np.expand_dims(pll_normalised, axis=1)),
            axis=1,
        ),
    )


if __name__ == "__main__":
    cli()
