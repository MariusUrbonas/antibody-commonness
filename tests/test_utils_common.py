import pytest
import torch
from transformers import AutoTokenizer, T5Tokenizer
from antibody_commonness.utils.common import (
    space_amino_acid_tokens,
    tokenise_antibody_with_temp_mask,
    encode_anitbody,
    encode_antibody_list
)

@pytest.fixture
def tokenizer_igbert():
    return AutoTokenizer.from_pretrained("Exscientia/IgBert_unpaired", do_lower_case=False)

@pytest.fixture
def tokenizer_igT5():
    return T5Tokenizer.from_pretrained("Exscientia/IgT5_unpaired", do_lower_case=False)

def test_space_amino_acid_tokens():
    assert space_amino_acid_tokens("ACDEFG") == "A C D E F G"
    assert space_amino_acid_tokens("A") == "A"
    assert space_amino_acid_tokens("") == ""

def test_tokenise_antibody_with_temp_mask():
    assert tokenise_antibody_with_temp_mask("AC?FG", "?", "[MASK]") == "A C [MASK] F G"
    assert tokenise_antibody_with_temp_mask("ACDEFG", "?", "[MASK]") == "A C D E F G"
    assert tokenise_antibody_with_temp_mask("??", "?", "[MASK]") == "[MASK] [MASK]"


@pytest.mark.parametrize("tokenizer_fixture", ["tokenizer_igbert", "tokenizer_igT5"])
def test_encode_anitbody(request, tokenizer_fixture):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    result = encode_anitbody("ACDEFG", tokenizer)
    assert 'input_ids' in result
    assert isinstance(result['input_ids'], torch.Tensor)
    assert result['input_ids'].shape[0] == 1  # batch size of 1
    assert result['input_ids'].shape[1] > 6  # at least 6 tokens (plus special tokens)

@pytest.mark.parametrize("tokenizer_fixture", ["tokenizer_igbert", "tokenizer_igT5"])
def test_encode_antibody_list(request, tokenizer_fixture):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    antibody_list = ["ACDEFG", "HIJKLM", "NPQRST"]
    result = encode_antibody_list(antibody_list, tokenizer)
    
    assert 'input_ids' in result
    assert 'attention_mask' in result
    assert 'special_tokens_mask' in result
    
    assert isinstance(result['input_ids'], torch.Tensor)
    assert result['input_ids'].shape[0] == 3  # batch size of 3
    assert result['input_ids'].shape[1] > 6  # at least 6 tokens (plus special tokens)
    
    assert result['attention_mask'].shape == result['input_ids'].shape
    assert result['special_tokens_mask'].shape == result['input_ids'].shape

@pytest.mark.parametrize("tokenizer_fixture", ["tokenizer_igbert", "tokenizer_igT5"])
def test_encode_antibody_list_padding(request, tokenizer_fixture):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    antibody_list = ["AC", "HIJKLM", "NPQRST"]
    result = encode_antibody_list(antibody_list, tokenizer)
    
    # Check if all sequences are padded to the same length
    assert torch.all(result['input_ids'][0, 4:] == tokenizer.pad_token_id)
    assert result['input_ids'].shape[1] == result['input_ids'][1].ne(tokenizer.pad_token_id).sum()