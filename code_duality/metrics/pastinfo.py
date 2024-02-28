import time
import numpy as np

from typing import Dict, List
from graphinf.utility import seed as gi_seed
from code_duality.factories import (
    GraphFactory,
    DataModelFactory,
)

from code_duality.config import Config
from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = [
    "PastDependentInformationMeasures",
    "PastDependentInformationMeasuresMetrics",
]


class PastDependentInformationMeasures(Expectation):
    """Past dependent information measures: prior, likelihood, posterior, evidence and mutual information.

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
            . past_length: the length of the past.
        - n_samples: the number of samples to use.
        - n_workers: the number of workers to use.
        - n_async_jobs: the number of asynchronous jobs to use.
        - callbacks: the callbacks to use.
    """

    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def func(self, seed: int) -> float:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.data_model.prior)
        data_model = DataModelFactory.build(config.data_model)

        data_model.set_prior(prior)
        if config.target == "None":
            prior.sample()
            g0 = prior.state()
        else:
            g0 = prior.state()
        x0 = data_model.random_state(config.data_model.get("n_active", -1))
        data_model.set_graph(g0)
        data_model.sample_state(x0)
        out = {}

        # computing full
        og = data_model.graph()

        data_model.set_graph(og)
        full = self.gather(data_model, config.metrics)
        out.update(full)
        out["mutualinfo"] = full["prior"] - full["posterior"]

        # computing past
        past_length = config.metrics.past_length
        if 0 < past_length < 1:
            past_length = int(past_length * config.data_model.length)
        elif past_length < 0:
            past_length = config.data_model.length + past_length
        data_model.set_length(int(past_length))
        past = self.gather(data_model, config.metrics)
        data_model.set_length(config.data_model.length)
        out["likelihood_past"] = past["likelihood"]
        out["evidence_past"] = past["evidence"]
        out["posterior_past"] = past["posterior"]
        out["mutualinfo_past"] = past["prior"] - past["posterior"]

        if config.metrics.get("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out

    def gather(self, data_model, metrics_cf):
        graph_mcmc = metrics_cf.graph_mcmc.dict if metrics_cf.graph_mcmc is not None else dict()
        graph_mcmc.pop("reset_to_original", None)
        prior = -data_model.prior.log_evidence(reset_to_original=True, **graph_mcmc)
        likelihood = -data_model.log_likelihood()
        posterior = data_model.log_posterior(**metrics_cf.data_mcmc)
        evidence = prior + likelihood - posterior
        return dict(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
        )


class PastDependentInformationMeasureMetrics(ExpectationMetrics):
    PastDependentInformationMeasures.__doc__
    shortname = "pastinfo"
    keys = [
        "prior",
        "likelihood",
        "posterior",
        "evidence",
        "mutualinfo",
        "prior_past",
        "likelihood_past",
        "posterior_past",
        "evidence_past",
        "mutualinfo_past",
    ]
    expectation_factory = PastDependentInformationMeasures

    def postprocess(self, samples: List[Dict[str, float]]) -> Dict[str, float]:
        stats = self.reduce(samples, self.configs.metrics.get("reduction", "normal"))
        # stats["recon"] = stats["mutualinfo"] / stats["prior"]
        # stats["pred"] = stats["mutualinfo"] / stats["evidence"]
        out = self.format(stats)
        print(out)
        return out


if __name__ == "__main__":
    pass
