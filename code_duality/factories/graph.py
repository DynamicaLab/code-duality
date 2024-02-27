from typing import List, Optional

from basegraph import core
from graphinf.graph import (
    ConfigurationModel,
    ConfigurationModelFamily,
    ErdosRenyiModel,
    NegativeBinomialGraph,
    PlantedPartitionGraph,
    PoissonGraph,
    StochasticBlockModelFamily,
)
from code_duality.config import Config, static

from .factory import Factory

__all__ = ("GraphConfig", "GraphFactory")


@static
class GraphConfig(Config):
    def from_target(self, target: Config):
        for prop in ["size", "edge_count", "loopy", "multigraph"]:
            if prop in self and prop in target:
                setattr(self, prop, target.get(prop))

    @classmethod
    def karate(cls, path=None):
        return cls(
            "karate",
            size=34,
            edge_count=78,
            gt_id="karate/78",
            path=path,
            loopy=False,
            multigraph=False,
        )

    @classmethod
    def littlerock(cls, path=None):
        return cls(
            "littlerock",
            size=183,
            edge_count=2_494,
            gt_id="foodweb_little_rock",
            path=path,
            loopy=True,
            multigraph=True,
        )

    @classmethod
    def football(cls, path=None):
        return cls(
            "football",
            size=115,
            edge_count=615,
            gt_id="football",
            path=path,
            loopy=True,
            multigraph=True,
        )

    @classmethod
    def polbooks(cls, path=None):
        return cls(
            "polbooks",
            size=105,
            edge_count=441,
            gt_id="polbooks",
            path=path,
            loopy=True,
            multigraph=True,
        )

    @classmethod
    def us_congress(cls, path=None):
        return cls(
            "us_congress",
            size=446,
            edge_count=18_083,
            gt_id="us_congress/H93",
            path=path,
            loopy=True,
            multigraph=True,
        )

    @classmethod
    def openflights(cls, path=None):
        return cls(
            "openflights",
            size=3_214,
            edge_count=66_771,
            gt_id="openflights",
            path=path,
            loopy=False,
            multigraph=False,
        )

    @classmethod
    def euairlines(cls, path=None):
        return cls(
            "euairlines",
            size=450,
            edge_count=3_588,
            gt_id="eu_airlines",
            path=path,
            loopy=False,
            multigraph=True,
        )

    @classmethod
    def celegans(cls, path=None):
        return cls(
            "celegans",
            size=514,
            edge_count=2_363,
            gt_id="celegans_2019/male_gap_junction_synapse",
            path=path,
            loopy=True,
            multigraph=True,
            weights="synapses",
        )

    @classmethod
    def polblogs(cls, path=None):
        return cls(
            "polblogs",
            size=1490,
            edge_count=19090,
            gt_id="polblogs",
            path=path,
            loopy=True,
            multigraph=True,
        )

    @classmethod
    def erdosrenyi(
        cls,
        size: int = 100,
        edge_count: float = 250,
        likelihood_type: str = "uniform",
        canonical: bool = False,
        loopy: bool = True,
        multigraph: bool = True,
        edge_proposer_type: str = "uniform",
    ):
        return cls(
            "erdosrenyi",
            size=size,
            likelihood_type=likelihood_type,
            edge_count=edge_count,
            canonical=canonical,
            loopy=loopy,
            multigraph=multigraph,
            edge_proposer_type=edge_proposer_type,
        )

    @classmethod
    def configuration(
        cls,
        size: int = 100,
        edge_count: float = 250,
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        edge_proposer_type: str = "uniform",
    ):
        return cls(
            "configuration",
            size=size,
            edge_count=edge_count,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            edge_proposer_type=edge_proposer_type,
        )

    @classmethod
    def degree_constrained_configuration(cls, size: int = 100, degree_seq: Optional[List[int]] = None):
        return cls(
            "degree_constrained_configuration",
            size=size,
            degree_seq=degree_seq,
        )

    @classmethod
    def poisson(cls, size: int = 100, edge_count: int = 250):
        return cls(
            "poisson",
            size=size,
            edge_count=edge_count,
        )

    @classmethod
    def nbinom(
        cls,
        size: int = 100,
        edge_count: int = 250,
        heterogeneity: float = 0,
    ):
        return cls(
            "nbinom",
            size=size,
            edge_count=edge_count,
            heterogeneity=heterogeneity,
        )

    @classmethod
    def stochastic_block_model(
        cls,
        size: int = 100,
        edge_count: float = 250,
        block_count: Optional[int] = None,
        likelihood_type: str = "uniform",
        label_graph_prior_type: str = "uniform",
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        edge_proposer_type: str = "uniform",
        block_proposer_type: str = "multiflip",
        sample_label_count_prob: float = 0.1,
        shift: float = 1,
    ):
        return cls(
            "stochastic_block_model",
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            likelihood_type=likelihood_type,
            label_graph_prior_type=label_graph_prior_type,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            edge_proposer_type=edge_proposer_type,
            block_proposer_type=block_proposer_type,
            sample_label_count_prob=sample_label_count_prob,
            shift=shift,
        )

    @classmethod
    def degree_corrected_stochastic_block_model(cls, **kwargs):
        kwargs["likelihood_type"] = "degree_corrected"
        obj = cls.stochastic_block_model(**kwargs)
        obj.name = obj._name = "degree_corrected_stochastic_block_model"

        return obj

    @classmethod
    def planted_partition(
        cls,
        size: int = 100,
        edge_count: int = 250,
        block_count: int = 3,
        assortativity: float = 0.5,
        stub_labeled: bool = False,
        loopy: bool = True,
        multigraph: bool = True,
    ):
        return cls(
            "planted_partition",
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            assortativity=assortativity,
            stub_labeled=stub_labeled,
            loopy=loopy,
            multigraph=multigraph,
        )


class GraphFactory(Factory):
    __all_configs__ = GraphConfig

    @staticmethod
    def load_gtgraph(name: str, weight_name: str = None) -> core.UndirectedMultigraph:
        from graph_tool import collection
        from midynet.utility.convert import convert_graphtool_to_basegraph

        gt_graph = collection.ns[name]
        return convert_graphtool_to_basegraph(
            gt_graph,
            weights=gt_graph.ep[weight_name] if weight_name is not None else None,
        )

    @staticmethod
    def load_graph(config: GraphConfig) -> core.UndirectedMultigraph:
        if config.path is not None:
            from graphinf.utility import load_graph

            return load_graph(config.path)
        try:
            # print("Fetching graph from Network Repo...")
            return GraphFactory.load_gtgraph(config.gt_id, weight_name=config.weights)
        except KeyError:
            from graphinf.utility import load_graph

            # print("Loading graph locally...")
            if config.path is None:
                raise ValueError(f"Fetching is forbidden, and did not find path to `{config.name}`.")

            return load_graph(config.path)

    @staticmethod
    def build_karate(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_littlerock(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_euairlines(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_openflights(config: GraphConfig) -> core.UndirectedMultigraph:
        g = GraphFactory.load_graph(config)
        g.remove_selfloops()
        return g

    @staticmethod
    def build_celegans(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_polblogs(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_polbooks(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_football(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_erdosrenyi(config: GraphConfig) -> ErdosRenyiModel:
        return ErdosRenyiModel(
            config.size,
            config.edge_count,
            canonical=config.canonical,
            loopy=config.loopy,
            multigraph=config.multigraph,
            edge_proposer_type=config.edge_proposer_type,
        )

    @staticmethod
    def build_configuration(config: GraphConfig) -> ConfigurationModelFamily:
        return ConfigurationModelFamily(
            config.size,
            config.edge_count,
            degree_prior_type=config.degree_prior_type,
            canonical=config.canonical,
            edge_proposer_type=config.edge_proposer_type,
        )

    @staticmethod
    def build_degree_constrained_configuration(
        config: GraphConfig,
    ) -> ConfigurationModel:
        degrees = [0] * config.size if config.degree_seq is None else config.degree_seq

        return ConfigurationModel(degrees)

    @staticmethod
    def build_poisson(config: GraphConfig) -> PoissonGraph:
        return PoissonGraph(config.size, config.edge_count)

    @staticmethod
    def build_nbinom(config: GraphConfig) -> NegativeBinomialGraph:
        return NegativeBinomialGraph(config.size, config.edge_count, config.heterogeneity)

    @staticmethod
    def build_stochastic_block_model(
        config: GraphConfig,
    ) -> StochasticBlockModelFamily:
        return StochasticBlockModelFamily(
            size=config.size,
            edge_count=config.edge_count,
            block_count=config.block_count,
            likelihood_type=config.likelihood_type,
            label_graph_prior_type=config.label_graph_prior_type,
            degree_prior_type=config.degree_prior_type,
            canonical=config.canonical,
            edge_proposer_type=config.edge_proposer_type,
            block_proposer_type=config.block_proposer_type,
            sample_label_count_prob=config.sample_label_count_prob,
            shift=config.shift,
        )

    @staticmethod
    def build_degree_corrected_stochastic_block_model(
        config: GraphConfig,
    ) -> StochasticBlockModelFamily:
        return GraphFactory.build_stochastic_block_model(config)

    @staticmethod
    def build_planted_partition(config: GraphConfig):
        return PlantedPartitionGraph(
            size=config.size,
            edge_count=config.edge_count,
            block_count=config.block_count,
            assortativity=config.assortativity,
            stub_labeled=config.stub_labeled,
            loopy=config.loopy,
            multigraph=config.multigraph,
        )


if __name__ == "__main__":
    pass
