from antibody_commonness import antibody_commonness
from antibody_commonness import antibody_commonness

def test_dummy_func():
    def test_dummy_func():
        # Test if the function returns 1
        assert antibody_commonness.dummy_func() == 1

        # Test if the function returns an integer
        assert isinstance(antibody_commonness.dummy_func(), int)

        # Test if the function does not return a string
        assert not isinstance(antibody_commonness.dummy_func(), str)

        # Test if the function does not return None
        assert antibody_commonness.dummy_func() is not None
