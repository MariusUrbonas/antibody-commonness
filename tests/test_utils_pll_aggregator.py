import pytest
import numpy as np
from antibody_commonness.utils.pll_aggregator import PseudoLogLikelihoodAggregator

@pytest.fixture
def pll_aggregator():
    return PseudoLogLikelihoodAggregator(initial_size=5)

def test_initialization(pll_aggregator):
    assert len(pll_aggregator.counts) == 5
    assert len(pll_aggregator.pseudo_loglikelihood) == 5
    assert pll_aggregator.num_items == 0

def test_update_within_initial_capacity(pll_aggregator):
    new_values = [1.0, 2.0, 3.0]
    ids = [0, 1, 2]
    result = pll_aggregator.update(new_values, ids)
    assert np.array_equal(result, [1.0, 2.0, 3.0])
    assert pll_aggregator.num_items == 3

def test_update_beyond_initial_capacity():
    pll = PseudoLogLikelihoodAggregator(initial_size=3)
    new_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ids = [0, 1, 2, 3, 4]
    result = pll.update(new_values, ids)
    assert np.array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert pll.num_items == 5
    assert len(pll.counts) >= 5
    assert len(pll.pseudo_loglikelihood) >= 5

def test_update_with_repeated_ids(pll_aggregator):
    new_values = [1.0, 2.0, 3.0, 4.0]
    ids = [0, 1, 0, 1]
    result = pll_aggregator.update(new_values, ids)
    assert np.array_equal(result, [4.0, 6.0])
    assert np.array_equal(pll_aggregator.counts[:2], [2, 2])

def test_get_pll(pll_aggregator):
    pll_aggregator.update([1.0, 2.0, 3.0], [0, 1, 2])
    result = pll_aggregator.get_pll()
    assert np.array_equal(result, [1.0, 2.0, 3.0])

def test_get_length_normalised_pll(pll_aggregator):
    pll_aggregator.update([1.0, 2.0, 3.0], [0, 0, 1])
    result = pll_aggregator.get_length_normalised_pll()
    assert np.allclose(result, [1.5, 3.0])

def test_get_pll_at(pll_aggregator):
    pll_aggregator.update([1.0, 2.0, 3.0], [0, 1, 2])
    assert pll_aggregator.get_pll_at(1) == 2.0

def test_update_with_numpy_arrays():
    pll = PseudoLogLikelihoodAggregator()
    new_values = np.array([1.0, 2.0, 3.0])
    ids = np.array([0, 1, 2])
    result = pll.update(new_values, ids)
    assert np.array_equal(result, [1.0, 2.0, 3.0])

def test_update_with_mismatched_shapes():
    pll = PseudoLogLikelihoodAggregator()
    new_values = [1.0, 2.0, 3.0]
    ids = [0, 1]
    with pytest.raises(ValueError, match="new_values and ids must have the same shape"):
        pll.update(new_values, ids)

def test_update_with_negative_ids():
    pll = PseudoLogLikelihoodAggregator()
    new_values = [1.0, 2.0, 3.0]
    ids = [0, -1, 2]
    with pytest.raises(ValueError, match="ids must be non-negative"):
        pll.update(new_values, ids)

def test_get_pll_at_with_negative_id(pll_aggregator):
    with pytest.raises(ValueError, match="id must be non-negative"):
        pll_aggregator.get_pll_at(-1)

def test_large_id_expansion():
    pll = PseudoLogLikelihoodAggregator(initial_size=5)
    pll.update([1.0], [100])
    assert len(pll.counts) > 100
    assert len(pll.pseudo_loglikelihood) > 100
    assert pll.num_items == 101