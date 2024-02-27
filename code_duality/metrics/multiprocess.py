import multiprocessing as mp
import time
import numpy as np

from typing import Any, Optional, Sequence

__all__ = ("MultiProcess", "Expectation")


class MultiProcess:
    """MultiProcess utility base class.

    This class is a base class for all classes that need to compute a function in parallel.
    To use this class, derive a new class and implement the `func` method."""

    def __init__(self, inputs: Sequence[Any]):
        self.inputs = inputs

    def func(self, x: Any):
        """The function to compute in parallel. This method should be implemented in the derived class."""
        raise NotImplementedError()

    def compute(self, pool: Optional[mp.Pool] = None):
        """Compute the function in parallel using a given pool.

        Args:
            pool: the multiprocessing pool to use. If None, the function is computed sequentially. Default: None.

        Returns:
            The result of the function applied to the inputs.
        """
        if pool is None:
            return [self.func(x) for x in self.inputs]
        out = pool.map(self.func, self.inputs)
        return out

    def compute_async(self, pool: Optional[mp.Pool] = None):
        """Compute the function in parallel using a given pool asynchronously.

        Args:
            pool: the multiprocessing pool to use. If None, the function is computed sequentially. Default: None.

        Returns:
            The result of the function applied to the inputs.
        """
        if pool is None:
            return self.compute()
        return pool.map_async(self.func, self.inputs)


class Expectation(MultiProcess):
    """Expectation utility base class.

    This class is a base class for all classes that need to compute an expectation---i.e.,
    statistics over a set of samples---, in parallel. To use this class, derive a new class
    and implement the `func` method."""

    def __init__(self, seed=None, n_samples=1):
        self.seed = seed if seed is not None else int(time.time())
        super().__init__(self.seed + np.arange(n_samples))
