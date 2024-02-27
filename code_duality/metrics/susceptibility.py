import numpy as np

from typing import Dict, Any
from math import ceil
from basegraph import core as bg
from graphinf.graph import RandomGraphWrapper
from graphinf.utility import seed as gi_seed
from code_duality.factories import (
    GraphFactory,
    DataModelFactory,
)

from code_duality.config import Config
from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = ("Susceptibility", "SusceptibilityMetrics")


def spin_average(x: np.ndarray) -> float:
    """Compute the average magnetization of the states."""
    y = x.copy()
    y[x == 0] = -1
    y = y.mean(0)
    return np.mean(np.abs(y))


def spin_susceptibility(x: np.ndarray) -> float:
    """Compute the susceptibility of the states."""
    y = x.copy()
    y[x == 0] = -1
    m = np.abs(y.mean(0))
    return (np.mean(m**2) - np.mean(m) ** 2) / np.mean(m)


def spreading_susceptibility(x: np.ndarray) -> float:
    """Compute the susceptibility of the states."""
    n = x.sum(0)
    return (np.mean(n**2) - np.mean(n) ** 2) / np.mean(n)


def gap_susceptibility(x: np.ndarray, epsilon=0.05) -> float:
    """Compute the susceptibility of the states based on the gap between
    the absorbing and active states (1st order phase transition)."""
    y = x.copy()
    y = y.mean(0)
    if np.any(y < epsilon):
        y = y[y < epsilon]
    return np.std(y) / np.mean(y)


susceptibility_func = {
    "glauber": spin_susceptibility,
    "sis": spreading_susceptibility,
    "cowan": lambda x: gap_susceptibility(x) if x[:, 0].mean() == 1.0 else spreading_susceptibility(x),
}

average_func = {
    "glauber": spin_average,
    "sis": lambda x: x.mean(),
    "cowan": lambda x: x.mean(),
}


class Susceptibility(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def setup(self, seed: int) -> Any:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)

        model.set_graph_prior(prior)
        if config.target is None:
            g0 = prior.state()
        else:
            target = GraphFactory.build(config.target)
            if isinstance(target, bg.UndirectedMultigraph):
                g0 = target
            elif isinstance(target, RandomGraphWrapper):
                g0 = target.state()

        prior.set_state(g0)
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

    def func(self, seed: int) -> Dict[str, float]:
        config, model_dict = self.setup(seed)
        model = model_dict["model"]
        X = np.array(model.past_states())
        avg = average_func[config.data_model.name](X)
        susc = susceptibility_func[config.data_model.name](X)
        out = dict(average=avg, susceptibility=susc)
        print(out)
        return out


class SusceptibilityMetrics(ExpectationMetrics):
    shortname = "susceptibility"
    keys = [
        "average",
        "susceptibility",
    ]
    expectation_factory = Susceptibility


if __name__ == "__main__":
    pass
