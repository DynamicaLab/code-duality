import os
import numpy as np
import itertools
from code_duality import Config
from code_duality.factories import DataModelConfig, GraphConfig, MetricsConfig
from utils import format_sequence


class TimeStepDuality(Config):
    num_steps = np.unique(np.logspace(1, 4, 100).astype("int")).tolist()
    graph_model = GraphConfig.erdosrenyi(size=5, edge_count=5, multigraph=False, loopy=False)
    data_models = {
        "glauber": DataModelConfig.glauber(prior=graph_model, length=num_steps, coupling=[0.25, 0.5, 1.0]),
        "sis": DataModelConfig.sis(
            prior=graph_model,
            length=num_steps,
            infection_prob=[0.025, 0.05, 0.1],
            recovery_prob=0.1,
            auto_activation_prob=1e-4,
        ),
        "cowan": DataModelConfig.cowan(prior=graph_model, length=num_steps, nu=[0.5, 1.0, 2.0], eta=0.1),
    }

    def __init__(
        self,
        suffix: str,
        path_to_data: str,
        data_model: str,
        past_length: int = 1,
        num_workers=1,
        seed=None,
    ):
        super().__init__(
            name=f"timestep-duality-" + suffix,
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        self.path += "/" + self.name
        self.data_model = self.data_models[data_model]
        if past_length != 1:
            self.metrics = [
                MetricsConfig.pastinfo(
                    past_length=past_length, graph_mcmc="exact", data_mcmc="exact", n_samples=1000, reduction="normal"
                )
            ]
        else:
            self.metrics = MetricsConfig.bayesian(
                graph_mcmc="exact", data_mcmc="exact", num_samples=1000, reduction="normal", resample_graph=False
            )

        self.lock()


if __name__ == "__main__":
    os.makedirs("fig5", exist_ok=True)
    for t, d in itertools.product([1, 0.5], TimeStepDuality.data_models.keys()):
        suffix = f"{d}-t{t}"
        config = TimeStepDuality(suffix=suffix, data_model=d, path_to_data="data", num_workers=4, seed=None)
        config.save("fig5/timestep-duality-" + suffix + ".json")
