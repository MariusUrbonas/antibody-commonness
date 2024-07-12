import pytest
import torch
from antibody_commonness.data import AntibodyPLLDataset, collate_anitbody_pll_batch

@pytest.fixture
def sample_sequences():
    return [
        "ACDEF",
        "GHIKL",
        "MNPQR"
    ]

@pytest.fixture
def sample_dataset(sample_sequences):
    return AntibodyPLLDataset(sample_sequences)

def test_antibody_pll_dataset_initialization(sample_dataset, sample_sequences):
    assert isinstance(sample_dataset, AntibodyPLLDataset)
    assert sample_dataset.antibody_sequences == sample_sequences
    assert sample_dataset.MASK_TOKEN_PLACEHOLDER == "?"

def test_antibody_pll_dataset_iter(sample_dataset):
    items = list(sample_dataset)
    assert len(items) == 15  # 3 sequences * 5 amino acids each
    
    first_item = items[0]
    assert isinstance(first_item, dict)
    assert "masked_antibodies" in first_item
    assert "masked_aa" in first_item
    assert "mask_idx" in first_item
    assert "antibody_idx" in first_item

    assert first_item["masked_antibodies"] == "?CDEF"
    assert first_item["masked_aa"] == "A"
    assert first_item["mask_idx"] == 0
    assert first_item["antibody_idx"] == 0

    last_item = items[-1]
    assert last_item["masked_antibodies"] == "MNPQ?"
    assert last_item["masked_aa"] == "R"
    assert last_item["mask_idx"] == 4
    assert last_item["antibody_idx"] == 2

def test_collate_anitbody_pll_batch():
    batch = [
        {"masked_antibodies": "?CDEF", "masked_aa": "A", "mask_idx": 0, "antibody_idx": 0},
        {"masked_antibodies": "A?DEF", "masked_aa": "C", "mask_idx": 1, "antibody_idx": 0},
        {"masked_antibodies": "?HIKL", "masked_aa": "G", "mask_idx": 0, "antibody_idx": 1}
    ]

    collated = collate_anitbody_pll_batch(batch)

    assert isinstance(collated, dict)
    assert "masked_antibodies" in collated
    assert "masked_aa" in collated
    assert "antibody_idx" in collated
    assert "mask_idx" in collated

    assert collated["masked_antibodies"] == ["?CDEF", "A?DEF", "?HIKL"]
    assert collated["masked_aa"] == ["A", "C", "G"]
    assert collated["antibody_idx"] == [0, 0, 1]
    assert collated["mask_idx"] == [0, 1, 0]

def test_antibody_pll_dataset_with_dataloader(sample_dataset):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(sample_dataset, batch_size=2, collate_fn=collate_anitbody_pll_batch)
    
    batches = list(dataloader)
    assert len(batches) == 8  # 15 items / 2 batch size, rounded up

    first_batch = batches[0]
    assert len(first_batch["masked_antibodies"]) == 2
    assert len(first_batch["masked_aa"]) == 2
    assert len(first_batch["antibody_idx"]) == 2
    assert len(first_batch["mask_idx"]) == 2

def test_antibody_pll_dataset_empty_input():
    empty_dataset = AntibodyPLLDataset([])
    assert list(empty_dataset) == []

def test_antibody_pll_dataset_single_character_sequence():
    single_char_dataset = AntibodyPLLDataset(["A"])
    items = list(single_char_dataset)
    assert len(items) == 1
    assert items[0]["masked_antibodies"] == "?"
    assert items[0]["masked_aa"] == "A"
    assert items[0]["mask_idx"] == 0
    assert items[0]["antibody_idx"] == 0