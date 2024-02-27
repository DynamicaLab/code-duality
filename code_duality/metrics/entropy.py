import time
from typing import Any, Dict, Tuple, List

import numpy as np
from math import ceil
from graphinf.utility import seed as gi_seed, EdgeCollector
from graphinf.data.util import mcmc_on_graph
from graphinf.graph import RandomGraphWrapper
from code_duality.config import Config
from code_duality.factories import DataModelFactory, GraphFactory
from basegraph import core as bg


from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = ("ReconstructionInformation", "ReconstructionInformationMetrics")


class CallbackEdgeCollector:
    """Callback for the edge collector. It updates the collector with the current graph
    at each iteration.

    Args:
        collector: the edge collector.
        freq: the frequency of updates.
    """

    def __init__(self, collector: EdgeCollector, freq: int = 1):
        self.iteration = 0
        self.collector = collector
        self.freq = freq

    def __call__(self, model):
        self.collector.update(model.graph_copy(), keep_graph=self.iteration % self.freq == 0)
        self.iteration += 1


class EntropyMeasures(Expectation):
    """Entropy measures: log likelihood, log prior, log posterior, log evidence, prior entropy,
    posterior entropy, exact posterior entropy, evidence entropy, and information gain.

    The parameters of the config must contain the following:
        - prior: the prior graph.
        - data_model: the data model.
        - target (optional): the target graph. If None, the prior is used as the target
        - metrics: additional parameters specific to the calculation of the metrics:
            . graph_mcmc: the MCMC parameters for the prior.
            . data_mcmc: the MCMC parameters for the data model.
            . to_bits: whether to convert the results to bits.
            . n_graph_samples: the number of samples to use for the graph.
            . resample_graph: whether to resample the graph at each iteration.
        - n_samples: the number of samples to use.
        - n_workers: the number of workers to use.
        - n_async_jobs: the number of asynchronous jobs to use.
        - callbacks: the callbacks to use.
    """

    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def setup(self, seed: int) -> Tuple[Config, Dict[str, Any]]:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)

        model.set_prior(prior)
        if config.target == "None":
            prior.sample()
            g0 = prior.state()
        else:
            target = GraphFactory.build(config.target)
            if isinstance(target, bg.UndirectedMultigraph):
                g0 = target
            else:
                assert issubclass(target.__class__, RandomGraphWrapper)
                g0 = target.state()

        model.set_graph(g0)
        if config.metrics.get("resample_graph", False):
            prior.sample()

        if "n_active" in config.data_model:
            n0 = config.data_model.get("n_active", -1)
            n0 = ceil(n0 * g0.get_size()) if 0 < n0 < 1 else n0
            x0 = model.random_state(n0)
            model.sample_state(x0)
        else:
            model.sample_state()
        return config, dict(model=model, prior=prior)

    def gather(self, model, config):
        graph_mcmc = config.metrics.get("graph_mcmc", Config("c")).dict
        data_mcmc = config.metrics.get("data_mcmc", Config("c")).dict
        graph_mcmc.pop("name", None)
        data_mcmc.pop("name", None)

        log_likelihood = model.log_likelihood()
        collector = EdgeCollector(
            graphs=[model.graph_copy()],
            epsilon=data_mcmc.get("epsilon", 1e-5),
        )
        freq = max(
            1,
            data_mcmc.get("n_sweeps", 1000) // config.metrics.get("n_graph_samples", 1),
        )

        callback = CallbackEdgeCollector(collector, freq=freq)

        mcmc_on_graph(
            model,
            callback=callback,
            n_sweeps=data_mcmc.get("n_sweeps", 1000),
            n_gibbs_sweeps=data_mcmc.get("n_gibbs_sweeps", 4),
            n_steps_per_vertex=data_mcmc.get("n_steps_per_vertex", 1),
            burn_sweeps=data_mcmc.get("burn_sweeps", 0),
            sample_prior=data_mcmc.get("sample_prior", True),
            sample_params=data_mcmc.get("sample_params", False),
            start_from_original=data_mcmc.get("start_from_original", False),
            reset_original=True,
            verbose=False,
        )
        log_posterior = collector.log_prob_estimate(model.graph())
        log_prior = model.prior.log_evidence(**graph_mcmc)
        log_evidence = log_prior + log_likelihood - log_posterior
        exact_posterior_entropy = collector.entropy()
        prior_entropy = []
        posterior_entropy = []
        evidence_entropy = []
        for _ in range(config.metrics.get("n_graph_samples", 1)):
            g = collector.sample_from_collection()
            posterior_entropy.append(-collector.log_prob_estimate(g))
            prior_entropy.append(-model.prior.log_evidence(graph=g, **graph_mcmc))
            model.set_graph(g)
            evidence_entropy.append(posterior_entropy[-1] - model.log_likelihood() - prior_entropy[-1])
        posterior_entropy = np.mean(posterior_entropy)
        prior_entropy = np.mean(prior_entropy)
        evidence_entropy = np.mean(evidence_entropy)
        return dict(
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_posterior=log_posterior,
            log_evidence=log_evidence,
            prior_entropy=prior_entropy,
            posterior_entropy=posterior_entropy,
            exact_posterior_entropy=exact_posterior_entropy,
            evidence_entropy=evidence_entropy,
            infogain=prior_entropy - exact_posterior_entropy,
        )

    def func(self, seed: int) -> float:
        config, model_dict = self.setup(seed)
        model, _ = model_dict["model"], model_dict["prior"]
        out = self.gather(model, config)
        if config.metrics.get("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out


class EntropyMeasuresMetrics(ExpectationMetrics):
    shortname = "entropy"
    keys = [
        "log_likelihood",
        "log_prior",
        "log_posterior",
        "log_evidence",
        "prior_entropy",
        "posterior_entropy",
        "exact_posterior_entropy",
        "evidence_entropy",
        "infogain",
    ]
    expectation_factory = EntropyMeasures

    def postprocess(self, samples: List[Dict[str, float]]) -> Dict[str, float]:
        stats = self.reduce(samples, self.configs.metrics.get("reduction", "normal"))
        out = self.format(stats)
        print(out)
        return out


if __name__ == "__main__":
    pass
