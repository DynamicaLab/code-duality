import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import pathlib
import time

from tempfile import mkdtemp
from typing import Dict, Optional, List

from code_duality.config import Config
from code_duality.metrics.callback import MetricsCallback
from code_duality.statistics import Statistics
from code_duality.utils import to_batch

from .multiprocess import Expectation

__all__ = ("Metrics", "ExpectationMetrics")


class Metrics:
    """Metrics base class.

    This class is a base class for all metrics classes. It provides a common interface for
    computing metrics and storing the results in pandas dataframes. This class supports
    both synchronous and asynchronous computation of metrics, if sequenced configurations
    are provided.

    The methods `eval` and `eval_async` are used to compute the metrics for a single configuration,
    they should be implemented in the derived classes.

    Use the `compute` method to compute the metrics and store the results in the `data` attribute:

    ```python
    metrics = MyMetrics()
    metrics.compute(configs)

    print(metrics.data) # shows the results in a pandas dataframe
    ```
    """

    shortname = "metrics"
    keys = []

    def __init__(self):
        self.data = None

    def eval(
        self,
        config: Config,
        pool: Optional[mp.Pool] = None,
    ) -> Dict[str, float]:
        """Compute the metrics for a single configuration."""
        raise NotImplementedError()

    def eval_async(
        self,
        config: Config,
        pool: Optional[mp.Pool] = None,
    ) -> Dict[str, float]:
        """Compute the metrics for a single configuration asynchronously."""
        raise NotImplementedError()

    def compute(
        self,
        configs: Config,
        resume: bool = True,
        n_workers: int = 1,
        n_async_jobs: int = 1,
        callbacks: Optional[List[MetricsCallback]] = None,
    ) -> None:
        """Compute the metrics for a sequence of configurations.

        Args:
            configs: the sequence of configurations.
            resume: whether to resume the computation from the last computed configuration.
            n_workers: the number of workers to use.
            n_async_jobs: the number of asynchronous jobs to use.
            callbacks: the callbacks to use.
        """
        self.configs = configs
        config_seq = list(
            filter(lambda c: not self.already_computed(c), configs.to_sequence()) if resume else configs.to_sequence()
        )

        if n_async_jobs > 1 and n_workers > 1:
            data = self.run_async(config_seq, n_async_jobs, n_workers, callbacks)
        else:
            data = self.run(config_seq, n_workers, callbacks)

        self.data = dict(data)

    def run(
        self,
        config_seq: List[Config],
        n_workers: int = 1,
        callbacks: Optional[List[MetricsCallback]] = None,
    ) -> pd.DataFrame:
        """
        Run the computation of the metrics for a sequence of configurations.

        Args:
            config_seq (list[Config]): the sequence of configurations.
            n_workers (int): the number of workers to use. Default: 1.
            callbacks: the callbacks to use. If None, no callbacks are used. Default: None.

        Returns:
            (pd.DataFrame) The computed data.

        """
        callbacks = [] if callbacks is None else callbacks
        for i, config in enumerate(config_seq):
            if n_workers > 1:
                with mp.get_context("spawn").Pool(n_workers) as p:
                    raw = pd.DataFrame(self.postprocess(self.eval(config, p)))
            else:
                raw = pd.DataFrame(self.postprocess(self.eval(config)))
            for k, v in self.configs.summarize_subconfig(config).items():
                raw[k] = v
            raw["experiment"] = config.name
            if self.data is None:
                self.data = raw.copy()
            else:
                self.data = pd.concat([self.data, raw], ignore_index=True)
            for c in callbacks:
                c.update()
        return self.data

    def run_async(
        self,
        config_seq: List[Config],
        n_async_jobs: int,
        n_workers: int,
        callbacks: Optional[List[MetricsCallback]] = None,
    ):
        """
        Run the computation of the metrics for a sequence of configurations asynchronously.

        Args:
            config_seq (list[Config]): the sequence of configurations.
            n_async_jobs (int): the number of asynchronous jobs to use.
            n_workers (int): the number of workers to use.
            callbacks: the callbacks to use. If None, no callbacks are used. Default: None.

        Returns:
            (pd.DataFrame) The computed data.

        """
        if n_workers == 1:
            raise ValueError("Cannot use async mode when n_workers == 1.")
        callbacks = [] if callbacks is None else callbacks
        for batch in to_batch(config_seq, n_async_jobs):
            with mp.get_context("spawn").Pool(n_workers) as p:
                async_jobs = []

                # assign jobs
                for config in batch:
                    async_jobs.append(self.eval_async(config, p))

                # waiting for jobs to finish
                for job in async_jobs:
                    job.wait()

            # gathering results
            for job, config in zip(async_jobs, batch):
                raw = pd.DataFrame(self.postprocess(job.get()))
                for k, v in self.configs.summarize_subconfig(config).items():
                    raw[k] = v
                raw["experiment"] = config.name

                if self.data is None:
                    self.data = raw.copy()
                else:
                    self.data = pd.concat([self.data, raw], ignore_index=True)

            # callbacks update
            for c in callbacks:
                c.update()

        return self.data

    def postprocess(self, raw):
        """Postprocess the raw data."""
        return raw

    def already_computed(self, config):
        """Check if the configuration has already been computed."""
        if config.name not in self.data:
            return False
        cond = pd.DataFrame()
        for k, v in self.configs.summarize_subconfig(config).items():
            cond[k] = self.data[config.name][k] == v
        return np.any(np.prod(cond.values, axis=-1))

    def to_pickle(self, path: Optional[str | pathlib.Path] = None, **kwargs) -> str:
        """Save the data to a pickle file.

        Args:
            path (str, Path): the path to save the data. If None, a temporary directory is created.
            kwargs: additional arguments to pass to `pd.to_pickle`.

        Returns:
            (str) The path to the saved file.
        """

        if path is None:
            path = os.path.join(mkdtemp(), f"{self.shortname}.pkl")
        elif os.path.isdir(path):
            path = os.path.join(path, f"{self.shortname}.pkl")
        pd.to_pickle(self.data, path, **kwargs)
        return str(path)

    def read_pickle(self, path: str | pathlib.Path, **kwargs):
        """Read the data from a pickle file.

        Args:
            path (str, Path): the path to read the data from.
            kwargs: additional arguments to pass to `pd.read_pickle`.
        """

        if os.path.isdir(path):
            path = os.path.join(path, f"{self.shortname}.pkl")
        if not os.path.exists(path):
            return
        self.data = pd.read_pickle(path, **kwargs)


class ExpectationMetrics(Metrics):
    """Expectation metrics base class.

    This class is a base class for metrics that involve sampling and computing statistics.
    To use this class, you need to specify the `expectation_factory` attribute with a class
    that inherits from `Expectation`.
    """

    expectation_factory: Expectation = None

    def eval(self, config: Config, pool: mp.Pool = None):
        Metrics.eval.__doc__
        expectation = self.expectation_factory(
            config=config,
            seed=config.get("seed", int(time.time())),
            n_samples=config.metrics.get("n_samples", 1),
        )
        return expectation.compute(pool)

    def eval_async(self, config: Config, pool: mp.Pool = None):
        Metrics.eval_async.__doc__
        expectation = self.expectation_factory(
            config=config,
            seed=config.get("seed", int(time.time())),
            n_samples=config.metrics.get("n_samples", 1),
        )
        return expectation.compute_async(pool)

    def reduce(self, samples: List[Dict[str, float]], reduction: str = "normal"):
        """Computes as set of statistics from samples."""
        return {k: Statistics.from_samples([s[k] for s in samples], reduction=reduction, name=k) for k in samples[0]}

    def format(self, stats: Dict[str, Statistics]):
        """Formats the statistics in a dictionary.

        Note:
            This method is used to format the statistics in a dictionary. The default implementation
            formats the statistics in a dictionary of lists. If the statistics are not in a list format,
            the method will format them in a list format.
        """
        out = dict()
        for k, s in stats.items():
            if "samples" in s:
                out[k] = s.samples.tolist()
                continue
            for sk, sv in s.__data__.items():
                out[k + "_" + sk] = [sv]
        return out

    def postprocess(self, samples: List[Dict[str, float]]) -> Dict[str, Statistics]:
        return self.format(self.reduce(samples, self.configs.metrics.get("reduction", "normal")))


if __name__ == "__main__":
    pass
