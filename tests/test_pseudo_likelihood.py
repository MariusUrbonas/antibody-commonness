import pytest
from unittest.mock import MagicMock

import torch
import numpy as np
from transformers import AutoTokenizer, T5Tokenizer

from antibody_commonness.data.antibody_pll_dataset import AntibodyPLLDataset
from antibody_commonness.utils.pll_aggregator import PseudoLogLikelihoodAggregator
from antibody_commonness.pseudo_likelihood import calculate_pll 

class MockOutput:
    def __init__(self, logits):
        self.logits = logits

@pytest.fixture
def mock_model():
    def mock_forward(**kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        seq_length = kwargs['input_ids'].shape[1]
        vocab_size = 30
        logits = torch.ones((batch_size, seq_length, vocab_size))
        return MockOutput(logits)
    model = MagicMock()
    model.side_effect = mock_forward
    return model

@pytest.fixture
def tokenizer_igbert():
    return AutoTokenizer.from_pretrained("Exscientia/IgBert_unpaired", do_lower_case=False)

@pytest.fixture
def tokenizer_igT5():
    tokeniser = T5Tokenizer.from_pretrained("Exscientia/IgT5_unpaired", do_lower_case=False)
    tokeniser.mask_token = "<extra_id_0>"
    return tokeniser

@pytest.fixture
def sample_sequences():
    return [
        "ACDEF",
        "GHIKL",
        "MNPQR"
    ]

@pytest.fixture
def dataset(sample_sequences):
    return AntibodyPLLDataset(sample_sequences)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("tokenizer_fixture", ["tokenizer_igbert", "tokenizer_igT5"])
def test_calculate_pll(request, mock_model, tokenizer_fixture, dataset, device):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    batch_size = 2
    pll_storage = calculate_pll(mock_model, tokenizer, dataset, batch_size, device)

    assert isinstance(pll_storage, PseudoLogLikelihoodAggregator)
    assert pll_storage.num_items == 3  # Number of input sequences

    pll_values = pll_storage.get_pll()
    assert isinstance(pll_values, np.ndarray)
    assert len(pll_values) == 3  # One PLL value for each input sequence

    # Check that PLL values are negative (log probabilities)
    assert np.all(pll_values < 0)

@pytest.mark.parametrize("tokenizer_fixture", ["tokenizer_igbert", "tokenizer_igT5"])
def test_calculate_pll_empty_dataset(request, mock_model, tokenizer_fixture, device):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    empty_dataset = AntibodyPLLDataset([])
    batch_size = 2
    pll_storage = calculate_pll(mock_model, tokenizer, empty_dataset, batch_size, device)

    assert isinstance(pll_storage, PseudoLogLikelihoodAggregator)
    assert pll_storage.num_items == 0

    pll_values = pll_storage.get_pll()
    assert isinstance(pll_values, np.ndarray)
    assert len(pll_values) == 0

@pytest.mark.parametrize("tokenizer_fixture", ["tokenizer_igbert", "tokenizer_igT5"])
def test_calculate_pll_single_sequence(request, mock_model, tokenizer_fixture, device):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    single_seq_dataset = AntibodyPLLDataset(["ACDEF"])
    batch_size = 1
    pll_storage = calculate_pll(mock_model, tokenizer, single_seq_dataset, batch_size, device)

    assert isinstance(pll_storage, PseudoLogLikelihoodAggregator)
    assert pll_storage.num_items == 1

    pll_values = pll_storage.get_pll()
    assert isinstance(pll_values, np.ndarray)
    assert len(pll_values) == 1
    assert pll_values[0] < 0  # Log probability should be negative

@pytest.mark.parametrize("tokenizer_fixture", ["tokenizer_igbert", "tokenizer_igT5"])
def test_calculate_pll_different_batch_sizes(request, mock_model, tokenizer_fixture, dataset, device):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    batch_sizes = [1, 2, 5]
    pll_storages = [calculate_pll(mock_model, tokenizer, dataset, bs, device) for bs in batch_sizes]

    # Check that results are consistent across different batch sizes
    pll_values = [storage.get_pll() for storage in pll_storages]
    for pll in pll_values[1:]:
        np.testing.assert_allclose(pll, pll_values[0], rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_calculate_pll_device(mock_model, tokenizer_igbert, dataset, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device(device)
    batch_size = 2
    pll_storage = calculate_pll(mock_model, tokenizer_igbert, dataset, batch_size, device)

    assert isinstance(pll_storage, PseudoLogLikelihoodAggregator)
    assert pll_storage.num_items == 3  # Number of input sequences

    pll_values = pll_storage.get_pll()
    assert isinstance(pll_values, np.ndarray)
    assert len(pll_values) == 3  # One PLL value for each input sequence