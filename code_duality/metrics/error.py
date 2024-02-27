from copy import deepcopy
from typing import Dict

import numpy as np
from basegraph import core as bs
from graphinf.utility import seed as gi_seed
from graphinf.graph import RandomGraphWrapper
from code_duality.config import Config
from code_duality.factories import DataModelFactory, GraphFactory
from code_duality.statistics import Statistics

from .heuristics import (
    AverageProbabilityPredictor,
    BayesianReconstructor,
    PeixotoReconstructor,
    get_predictor,
    get_reconstructor,
    prepare_training_data,
)
from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = ["ErrorProbMetrics"]


class ReconstructionError(Expectation):
    """Reconstruction error.

    The parameters of the config must contain the following:
        - prior: the prior graph.
        - data_model: the data model.
        - target (optional): the target graph. If None, the prior is used as the target
        - metrics: additional parameters specific to the calculation of the metrics:
            . reconstructor: the reconstructor to use. Options are "bayesian", "peixoto", or "heuristic".
            . data_mcmc: the MCMC parameters for the data model. Only needed for the Bayesian reconstructor.
            . measures: the measures to compute.
        - n_samples: the number of samples to use.
        - n_workers: the number of workers to use.
        - n_async_jobs: the number of asynchronous jobs to use.
        - callbacks: the callbacks to use.
    """

    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def func(self, seed: int) -> Dict[str, float]:
        # Data generation
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)
        model.set_prior(prior)

        if config.target != "None":
            prior.sample()
            g0 = prior.state()
        else:
            target = GraphFactory.build(config.target)
            if isinstance(target, bs.UndirectedMultigraph):
                g0 = target
            else:
                assert issubclass(target.__class__, RandomGraphWrapper)
                g0 = target.state()
        prior.from_graph(g0)

        if "n_active" in config.data_model:
            x0 = model.random_state(config.data_model.get("n_active", -1))

            model.sample_state(x0)
            x = np.array(model.past_states())
        else:
            model.sample_state()
            x = np.array(model.state())

        # Reconstruction
        if config.metrics.get("reconstructor") == "bayesian":
            reconstructor = BayesianReconstructor(config)
            reconstructor.model.from_model(model)
            data_mcmc = config.metrics.get("data_mcmc", Config("c")).dict
            reconstructor.fit(g0=g0, **data_mcmc)
        elif config.metrics.get("reconstructor") == "peixoto":
            reconstructor = PeixotoReconstructor(config)
            reconstructor.model.from_model(model)
            data_mcmc = config.metrics.get("data_mcmc", Config("c")).dict
            reconstructor.fit(g0=g0, **data_mcmc)
        else:
            reconstructor = get_reconstructor(config.metrics)
            reconstructor.fit(x)

        # Evaluation
        out = reconstructor.compare(g0, measures=config.metrics.measures)
        out = {k: v for k, v in out.items() if isinstance(v, (float, int))}
        return out


class ReconstructionErrorMetrics(ExpectationMetrics):
    ReconstructionError.__doc__
    shortname = "recon_error"
    keys = "recon_error"
    expectation_factory = ReconstructionError

    def postprocess(self, samples: list[Dict[str, float]]) -> Dict[str, Statistics]:
        out = super().postprocess(samples)
        return out


class PredictionError(Expectation):
    """Prediction error.

    The parameters of the config must contain the following:
        - prior: the prior graph.
        - data_model: the data model.
        - metrics: additional parameters specific to the calculation of the metrics:
            . predictor: the predictor to use. Options are "average_probability" or "heuristic".
            . n_train_samples: the number of training samples to use.
            . measures: the measures to compute.
        - n_samples: the number of samples to use.
        - n_workers: the number of workers to use.
        - n_async_jobs: the number of asynchronous jobs to use.
        - callbacks: the callbacks to use.
    """

    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def func(self, seed: int) -> Dict[str, float]:
        # Data generation
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)
        model.set_prior(prior)
        g0 = model.graph()

        if "n_active" in config.data_model:
            x0 = model.random_state(config.data_model.get("n_active", -1))

            model.sample_state(x0)
            x = np.array(model.past_states()).T
        else:
            model.sample_state()
            x = np.array(model.state()).T

        # Prediction
        if config.metrics.get("predictor") == "average_probability":
            predictor = AverageProbabilityPredictor(config)
            predictor.fit(
                inputs=model.past_states(),
                targets=model.future_states(),
                n_train_samples=config.metrics.get("n_train_samples", 100),
            )
        else:
            predictor = get_predictor(config.metrics)
            x_train, y_train = prepare_training_data(
                config,
                n_train_samples=config.metrics.get("n_train_samples", 100),
            )
            predictor.fit(x_train, y_train, **config.metrics)

        # Evaluation
        targets = np.array(model.transition_matrix(out_state=1)).T
        preds = predictor.predict(inputs=x)
        out = predictor.eval(targets, preds, measures=config.metrics.measures)
        out = {k: v for k, v in out.items() if isinstance(v, (float, int))}
        return out


class PredictionErrorMetrics(ExpectationMetrics):
    PredictionError.__doc__
    shortname = "pred_error"
    keys = "pred_error"
    expectation_factory = PredictionError

    def postprocess(self, samples: list[Dict[str, float]]) -> Dict[str, Statistics]:
        out = super().postprocess(samples)
        return out
