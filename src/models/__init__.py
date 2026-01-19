"""Drug repurposing models."""

from .link_prediction import DrugDiseasePredictor
from .explainer import PredictionExplainer
from .embeddings import (
    TransE,
    RotatE,
    ComplEx,
    DistMult,
    ConvE,
    create_embedding_model,
    EmbeddingConfig,
)
from .gnn import (
    GAT,
    GraphSAGE,
    GIN,
    RGCN,
    create_gnn_model,
)
from .ensemble import (
    EnsembleAggregator,
    EnsemblePredictor,
    AggregationMethod,
)
from .rare_disease import (
    RareDiseaseRepurposer,
    DiseaseSimilarityCalculator,
    MultiHopReasoner,
)
from .llm_extractor import (
    LLMRelationshipExtractor,
    LiteratureKnowledgeEnricher,
)

__all__ = [
    # Core
    "DrugDiseasePredictor",
    "PredictionExplainer",
    # Embeddings
    "TransE",
    "RotatE",
    "ComplEx",
    "DistMult",
    "ConvE",
    "create_embedding_model",
    "EmbeddingConfig",
    # GNNs
    "GAT",
    "GraphSAGE",
    "GIN",
    "RGCN",
    "create_gnn_model",
    # Ensemble
    "EnsembleAggregator",
    "EnsemblePredictor",
    "AggregationMethod",
    # Rare disease
    "RareDiseaseRepurposer",
    "DiseaseSimilarityCalculator",
    "MultiHopReasoner",
    # LLM extraction
    "LLMRelationshipExtractor",
    "LiteratureKnowledgeEnricher",
]
