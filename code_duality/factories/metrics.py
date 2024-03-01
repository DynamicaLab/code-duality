from __future__ import annotations

from typing import TypedDict, Union
from typing_extensions import Unpack

import code_duality.metrics
from code_duality.config import Config, static

from .factory import Factory, OptionError

__all__ = ("MetricsConfig", "MetricsCollectionConfig", "MetricsFactory")


class MCMCKwargs(TypedDict):
    n_sweeps: int
    n_gibbs_sweeps: int
    n_steps_per_vertex: int
    burn_sweeps: int
    start_from_original: bool
    reset_original: bool


@static
class MCMCDataConfig(Config):
    @classmethod
    def exact(cls, **kwargs):
        return cls(
            "data_mcmc",
            method="exact",
            reset_original=kwargs.pop("reset_original", True),
            **kwargs,
        )

    @classmethod
    def meanfield(cls, sample_prior: bool = True, sample_params: bool = False, **kwargs: Unpack[MCMCKwargs]):
        return cls(
            "data_mcmc",
            method="meanfield",
            sample_prior=sample_prior,
            sample_params=sample_params,
            n_sweeps=kwargs.pop("n_sweeps", 1000),
            n_gibbs_sweeps=kwargs.pop("n_gibbs_sweeps", 10),
            n_steps_per_vertex=kwargs.pop("n_steps_per_vertex", 1),
            burn_sweeps=kwargs.pop("burn_sweeps", 4),
            start_from_original=kwargs.pop("start_from_original", False),
            reset_original=kwargs.pop("reset_original", True),
            **kwargs,
        )

    @classmethod
    def annealed(
        cls,
        sample_prior: bool = True,
        sample_params: bool = False,
        n_betas: int = 10,
        exp_betas: float = 0.5,
        **kwargs: Unpack[MCMCKwargs],
    ):
        return cls(
            "annealed",
            method="annealed",
            sample_prior=sample_prior,
            sample_params=sample_params,
            n_betas=n_betas,
            exp_betas=exp_betas,
            n_sweeps=kwargs.pop("n_sweeps", 1000),
            n_gibbs_sweeps=kwargs.pop("n_gibbs_sweeps", 10),
            n_steps_per_vertex=kwargs.pop("n_steps_per_vertex", 1),
            burn_sweeps=kwargs.pop("burn_sweeps", 4),
            start_from_original=kwargs.pop("start_from_original", False),
            reset_original=kwargs.pop("reset_original", True),
            **kwargs,
        )


@static
class MCMCGraphConfig(Config):
    @classmethod
    def exact(cls, **kwargs):
        return cls("exact", method="exact", reset_original=True, **kwargs)

    @classmethod
    def meanfield(cls, equilibriate_mode_cluster: bool = False, **kwargs: Unpack[MCMCKwargs]):
        return cls(
            "meanfield",
            method="partition_meanfield",
            equilibriate_mode_cluster=equilibriate_mode_cluster,
            n_sweeps=kwargs.pop("n_sweeps", 1000),
            n_gibbs_sweeps=kwargs.pop("n_gibbs_sweeps", 5),
            n_steps_per_vertex=kwargs.pop("n_steps_per_vertex", 1),
            burn_sweeps=kwargs.pop("burn_sweeps", 4),
            start_from_original=kwargs.pop("start_from_original", False),
            reset_original=kwargs.pop("reset_original", True),
            **kwargs,
        )


@static
class MetricsConfig(Config):
    @classmethod
    def mcmc(
        cls,
        name: str,
        graph_mcmc: MCMCGraphConfig | str = "meanfield",
        data_mcmc: MCMCDataConfig | str = "meanfield",
        reduction="normal",
        n_samples=100,
        resample_graph=False,
        **kwargs,
    ):
        graph_mcmc = getattr(MCMCGraphConfig, graph_mcmc)() if isinstance(graph_mcmc, str) else graph_mcmc
        data_mcmc = getattr(MCMCDataConfig, data_mcmc)() if isinstance(data_mcmc, str) else data_mcmc
        return cls(
            name,
            graph_mcmc=graph_mcmc,
            data_mcmc=data_mcmc,
            reduction=reduction,
            n_samples=n_samples,
            resample_graph=resample_graph,
            **kwargs,
        )

    @classmethod
    def bayesian(cls, **kwargs):
        return cls.mcmc("bayesian", **kwargs)

    @classmethod
    def pastinfo(cls, **kwargs):
        return cls.mcmc("pastinfo", past_length=1.0, **kwargs)

    @classmethod
    def entropy(cls, **kwargs):
        return cls.mcmc(
            "entropy",
            n_graph_samples=kwargs.pop("n_graph_samples", 20),
            **kwargs,
        )

    @classmethod
    def susceptibility(cls):
        return cls(
            "susceptibility",
            n_samples=100,
            reduction="identity",
            resample_graph=False,
        )

    @classmethod
    def recon_error(cls, **kwargs):
        return cls.mcmc(
            "recon_error",
            reconstructor=kwargs.pop("reconstructor", "bayesian"),
            measures=kwargs.pop("measures", "roc, posterior_similarity, accuracy"),
            **kwargs,
        )

    @classmethod
    def pred_error(cls, **kwargs):
        return cls(
            "pred_error",
            predictor=kwargs.pop("predictor", "average_probability"),
            n_samples=kwargs.pop("n_samples", 100),
            reduction=kwargs.pop("reduction", "normal"),
            measures=kwargs.pop("measures", "absolute_error"),
            **kwargs,
        )


@static
class MetricsCollectionConfig(Config):
    @classmethod
    def auto(cls, configs: Union[str, list[str], list[MetricsConfig]]):
        if not isinstance(configs, list):
            configs = [configs]
        configs = [getattr(MetricsConfig, c)() if isinstance(c, str) else c for c in configs]
        configs = {c.name: c for c in configs}

        config = cls(
            "metrics",
            **configs,
        )
        config._state["metrics_names"] = list(configs.keys())
        config.__types__["metrics_names"] = str
        config.not_sequence("metrics_names")
        return config


class MetricsFactory(Factory):
    @classmethod
    def build(cls, config: Config) -> code_duality.metrics.Metrics:
        options = cls.options()
        if config.name in options:
            return getattr(cls, "build_" + config.name)()
        else:
            raise OptionError(actual=config.name, expected=list(options.keys()))

    @staticmethod
    def build_bayesian():
        return code_duality.metrics.BayesianInformationMeasuresMetrics()

    @staticmethod
    def build_pastinfo():
        return code_duality.metrics.PastDependentInformationMeasureMetrics()

    @staticmethod
    def build_susceptibility():
        return code_duality.metrics.SusceptibilityMetrics()

    @staticmethod
    def build_recon_error():
        return code_duality.metrics.ReconstructionErrorMetrics()

    @staticmethod
    def build_pred_error():
        return code_duality.metrics.PredictionErrorMetrics()

    @staticmethod
    def build_entropy():
        return code_duality.metrics.EntropyMeasuresMetrics()


if __name__ == "__main__":
    pass
