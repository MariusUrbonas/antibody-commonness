import numpy as np
from typing import List, Union


class PseudoLogLikelihoodAggregator:
    def __init__(self, initial_size: int = 10):
        """
        Initialize the PseudoLogLikelihoodAggregator object.

        Args:
            initial_size (int): The initial size of the arrays. Defaults to 10.
        """
        self.counts = np.zeros(initial_size, dtype=int)
        self.pseudo_loglikelihood = np.zeros(initial_size, dtype=float)
        self.num_items = 0

    def _ensure_capacity(self, max_id: int):
        """
        Ensure the arrays have enough capacity to store the given ID.

        Args:
            max_id (int): The maximum ID that needs to be stored.
        """
        if max_id >= len(self.counts):
            new_size = max(len(self.counts) * 2, max_id + 1)
            self.counts = np.pad(self.counts, (0, new_size - len(self.counts)))
            self.pseudo_loglikelihood = np.pad(self.pseudo_loglikelihood, (0, new_size - len(self.pseudo_loglikelihood)))

    def update(self, new_values: Union[List[float], np.ndarray], ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Update multiple running averages with new observations.

        Args:
            new_values (Union[List[float], np.ndarray]): The new values to incorporate into the averages.
            ids (Union[List[int], np.ndarray]): The IDs indicating which average each new value belongs to.

        Returns:
            np.ndarray: The updated averages for the affected IDs.
        """
        new_values = np.asarray(new_values)
        ids = np.asarray(ids)

        if new_values.shape != ids.shape:
            raise ValueError("new_values and ids must have the same shape")

        if np.min(ids) < 0:
            raise ValueError("ids must be non-negative")

        # Ensure capacity
        self._ensure_capacity(np.max(ids))

        self.num_items = max(np.max(ids)+1, self.num_items)

        # Increment counts for affected averages
        np.add.at(self.counts, ids, 1)

        # Update averages
        np.add.at(self.pseudo_loglikelihood, ids, new_values)

        return self.pseudo_loglikelihood[:self.num_items]

    def get_pll(self) -> np.ndarray:
        """
        Get all current averages.

        Returns:
            np.ndarray: The current averages.
        """
        return self.pseudo_loglikelihood[:self.num_items]
    
    def get_length_normalised_pll(self) -> np.ndarray:
        """
        Get all current averages.

        Returns:
            np.ndarray: The current averages.
        """
        return self.pseudo_loglikelihood[:self.num_items] / self.counts[:self.num_items]

    def get_pll_at(self, id: int) -> float:
        """
        Get a specific average.

        Args:
            id (int): The ID of the average to retrieve.

        Returns:
            float: The current average for the specified ID.
        """
        if id < 0:
            raise ValueError("id must be non-negative")
        self._ensure_capacity(id)
        return self.pseudo_loglikelihood[id]