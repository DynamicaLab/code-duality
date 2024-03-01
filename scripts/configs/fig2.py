import os
from code_duality import Config
from code_duality.factories import DataModelConfig, GraphConfig, MetricsConfig
from utils import format_sequence


class MIvsHeuristics(Config):
    def __init__(
        self,
        path_to_data: str,
        num_workers=1,
        seed=None,
    ):
        super().__init__(
            name=f"mi-vs-heuristics",
            path=path_to_data,
            num_workers=num_workers,
            seed=seed,
        )
        self.path += "/" + self.name
        prior = GraphConfig.erdosrenyi(size=1000, edge_count=2500, multigraph=False, loopy=False)
        self.data_model = DataModelConfig.glauber(
            length=1000, coupling=format_sequence((0, 0.2, 20), (0.2, 0.8, 10)), prior=prior
        )
        self.metrics = [
            MetricsConfig.bayesian(
                graph_mcmc="meanfield", data_mcmc="meanfield", num_samples=25, reduction="normal", resample_graph=False
            ),
            MetricsConfig.recon_error(
                reconstructor=["correlation", "granger_causality", "transfer_entropy"],
                measures="roc",
                num_samples=25,
                reduction="normal",
            ),
            MetricsConfig.pred_error(
                predictor=["mle", "logistic", "mlp"],
                measures="absolute_error",
                num_samples=25,
                reduction="normal",
            ),
        ]
        self.metrics[1].pop("graph_mcmc")
        self.metrics[1].pop("data_mcmc")
        self.metrics[1].pop("resample_graph")
        self.metrics[2].pop("graph_mcmc")
        self.metrics[2].pop("data_mcmc")
        self.lock()


if __name__ == "__main__":
    config = MIvsHeuristics(path_to_data="data", num_workers=4, seed=None)
    os.makedirs("fig2", exist_ok=True)
    config.save("fig2/mi-vs-heuristics.json")
