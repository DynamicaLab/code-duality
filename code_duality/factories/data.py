from __future__ import annotations
from typing import Any
from graphinf.data.dynamics import (
    CowanDynamics,
    DegreeDynamics,
    SISDynamics,
    GlauberDynamics,
)
from graphinf.data.uncertain import PoissonUncertainGraph
from code_duality.config import Config, static
from .factory import Factory, OptionError

__all__ = ("DataModelConfig", "DataModelFactory")


@static
class DataModelConfig(Config):
    @classmethod
    def glauber(
        cls,
        length: int = 100,
        coupling: float = 1.0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        n_active: int = -1,
    ):
        return cls(
            name="glauber",
            length=length,
            coupling=coupling,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            n_active=n_active,
        )

    @classmethod
    def sis(
        cls,
        length: int = 100,
        infection_prob: float = 0.1,
        recovery_prob: float = 0.1,
        auto_activation_prob=0.001,
        auto_deactivation_prob=0,
        n_active: int = 1,
    ):
        return cls(
            name="sis",
            length=length,
            infection_prob=infection_prob,
            recovery_prob=recovery_prob,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            n_active=n_active,
        )

    @classmethod
    def cowan(
        cls,
        length: int = 100,
        nu: float = 1.0,
        a: float = 8.0,
        mu: float = 1.0,
        eta: float = 0.1,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        n_active: int = 1,
    ):
        return cls(
            name="cowan",
            length=length,
            nu=nu,
            a=a,
            mu=mu,
            eta=eta,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            n_active=n_active,
        )

    @classmethod
    def cowan_forward(cls, **kwargs):
        cfg = cls.cowan(**kwargs)
        cfg.n_active = 1
        return cfg

    @classmethod
    def cowan_backward(cls, **kwargs):
        cfg = cls.cowan(**kwargs)
        cfg.n_active = -1
        return cfg

    @classmethod
    def degree(
        cls,
        length: int = 100,
        C: float = 1.0,
        auto_activation_prob=0,
        auto_deactivation_prob=0,
        n_active: int = 2**31 - 1,
    ):
        return cls(
            name="degree",
            length=length,
            C=C,
            auto_activation_prob=auto_activation_prob,
            auto_deactivation_prob=auto_deactivation_prob,
            n_active=n_active,
        )

    @classmethod
    def poisson_graph(
        cls,
        mu: float = 5,
        mu_no_edge: float = 0,
    ):
        return cls(
            name="poisson_graph",
            mu=mu,
            mu_no_edge=mu_no_edge,
        )


class DataModelFactory(Factory):
    __all_config__ = DataModelConfig

    @classmethod
    def build(cls, config: Config) -> Any:
        options = {k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"}
        name = config.name
        if name in options:
            return options[name](config)
        else:
            raise OptionError(actual=name, expected=list(options.keys()))

    @staticmethod
    def build_glauber(config: DataModelConfig):
        return GlauberDynamics(
            length=config.length,
            coupling=config.coupling,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
        )

    @staticmethod
    def build_sis(config: DataModelConfig):
        return SISDynamics(
            length=config.length,
            infection_prob=config.infection_prob,
            recovery_prob=config.recovery_prob,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
        )

    @staticmethod
    def build_cowan(config: DataModelConfig):
        return CowanDynamics(
            length=config.length,
            nu=config.nu,
            a=config.a,
            mu=config.mu,
            eta=config.eta,
            auto_activation_prob=config.auto_activation_prob,
            auto_deactivation_prob=config.auto_deactivation_prob,
        )

    @staticmethod
    def build_degree(config: DataModelConfig):
        return DegreeDynamics(length=config.length, C=config.C)

    @staticmethod
    def build_poisson_graph(config: DataModelConfig):
        return PoissonUncertainGraph(mu=config.mu, mu_no_edge=config.mu_no_edge)
