from pathlib import Path

from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    data_dir: Path = _PROJECT_ROOT / "data"
    model_dir: Path = _PROJECT_ROOT / "models"
    knowledge_store_path: Path = (
        _PROJECT_ROOT / "data" / "external_kb" / "merchant_knowledge.sqlite3"
    )
    knowledge_base_path: Path = (
        _PROJECT_ROOT / "data" / "external_kb" / "merchant_knowledge_base.json"
    )
    knowledge_chroma_dir: Path = _PROJECT_ROOT / "data" / "external_kb" / "chroma"
    db_path: Path = _PROJECT_ROOT / "data" / "corrections.db"

    rules_confidence: float = 0.98
    sgd_confidence_threshold: float = 0.70
    knowledge_min_similarity: float = 0.58
    knowledge_metadata_similarity: float = 0.68
    knowledge_direct_similarity: float = 0.90
    knowledge_direct_category_confidence: float = 0.82
    knowledge_collection_name: str = "merchant_knowledge"
    knowledge_dense_model_name: str = "BAAI/bge-small-en-v1.5"
    knowledge_reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    knowledge_dense_candidates: int = 15
    knowledge_lexical_candidates: int = 15
    knowledge_rerank_candidates: int = 20
    knowledge_rrf_k: int = 20
    knowledge_dense_weight: float = 1.0
    knowledge_lexical_weight: float = 0.8

    primary_model: str = "finetune"
    zeroshot_confidence_threshold: float = 0.60
    use_zeroshot: bool = False

    # keep these old knobs so older .env files still work
    bert_confidence_threshold: float = 0.60
    use_bert: bool = False

    token_analysis_enabled: bool = True

    batch_size: int = 256

    model_config = {"env_prefix": "TC_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
