import os
import numpy as np
import itertools
from code_duality import Config
from code_duality.factories import DataModelConfig, GraphConfig, MetricsConfig
from utils import format_sequence


class CouplingDualitySynthetic(Config):
    graph_model = GraphConfig.nbinom(size=1000, edge_count=2500, heterogeneity=1.0)
    data_models = {
        "glauber": DataModelConfig.glauber(
            prior=graph_model, length=2000, coupling=format_sequence((0.0, 0.5, 6), (0.5, 0.75, 22), (0.75, 1.5, 6))
        ),
        "sis": DataModelConfig.sis(
            prior=graph_model,
            length=2000,
            infection_prob=format_sequence((0.0, 0.125, 3), (0.125, 0.5, 30)),
            recovery_prob=0.5,
            auto_activation_prob=1e-4,
        ),
        "cowan_forward": DataModelConfig.cowan_forward(
            prior=graph_model,
            length=5000,
            nu=format_sequence(
                (2.0, 2.5, 23),
                (2.5, 4, 10),
            ),
        ),
        "cowan_backward": DataModelConfig.cowan_backward(
            prior=graph_model,
            length=5000,
            nu=format_sequence(
                (1.0, 1.25, 4),
                (1.25, 1.5, 20),
                (1.5, 3, 10),
            ),
        ),
    }

    def __init__(
        self,
        path_to_data: str,
        data_model: str,
        num_workers=1,
        seed=None,
    ):
        super().__init__(
            name=f"coupling-duality-synthetic-" + data_model,
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        self.path += "/" + self.name
        self.data_model = self.data_models[data_model]
        self.metrics = [
            MetricsConfig.bayesian(graph_mcmc="meanfield", data_mcmc="meanfield", n_samples=25, reduction="normal")
        ]
        self.lock()


class CouplingDualityRealNetworks(Config):
    graph_model = GraphConfig.degree_constrained_configuration()
    target_graphs = {
        "glauber": GraphConfig.littlerock(),
        "sis": GraphConfig.euairlines(),
        "cowan_forward": GraphConfig.celegans(),
        "cowan_backward": GraphConfig.celegans(),
    }
    data_models = {
        "glauber": DataModelConfig.glauber(
            prior=graph_model, length=2000, coupling=format_sequence((0.0, 0.5, 6), (0.5, 0.75, 22), (0.75, 1.5, 6))
        ),
        "sis": DataModelConfig.sis(
            prior=graph_model,
            length=2000,
            infection_prob=format_sequence((0.0, 0.125, 3), (0.125, 0.5, 30)),
            recovery_prob=0.5,
            auto_activation_prob=1e-4,
        ),
        "cowan_forward": DataModelConfig.cowan_forward(
            prior=graph_model,
            length=5000,
            nu=format_sequence(
                (2.0, 2.5, 23),
                (2.5, 4, 10),
            ),
        ),
        "cowan_backward": DataModelConfig.cowan_backward(
            prior=graph_model,
            length=5000,
            nu=format_sequence(
                (1.0, 1.25, 4),
                (1.25, 1.5, 20),
                (1.5, 3, 10),
            ),
        ),
    }

    def __init__(
        self,
        path_to_data: str,
        data_model: str,
        num_workers=1,
        seed=None,
    ):
        super().__init__(
            name=f"coupling-duality-synthetic-" + data_model,
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        self.path += "/" + self.name
        self.data_model = self.data_models[data_model]
        self.target = self.target_graphs[data_model]
        self.data_model.prior.size = self.target.size

        self.metrics = [
            MetricsConfig.bayesian(graph_mcmc="meanfield", data_mcmc="meanfield", n_samples=25, reduction="normal")
        ]
        self.lock()


if __name__ == "__main__":
    os.makedirs("fig7", exist_ok=True)
    for d in CouplingDualitySynthetic.data_models.keys():
        config = CouplingDualitySynthetic(data_model=d, path_to_data="data", num_workers=4, seed=None)
        config.save("fig7/coupling-duality-synthetic-" + d + ".json")
        config = CouplingDualityRealNetworks(data_model=d, path_to_data="data", num_workers=4, seed=None)
        config.save("fig7/coupling-duality-real-networks-" + d + ".json")
