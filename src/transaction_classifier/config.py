from pathlib import Path

from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    data_dir: Path = _PROJECT_ROOT / "data"
    model_dir: Path = _PROJECT_ROOT / "models"
    knowledge_base_path: Path = (
        _PROJECT_ROOT / "data" / "external_kb" / "merchant_knowledge_base.json"
    )
    db_path: Path = _PROJECT_ROOT / "data" / "corrections.db"

    rules_confidence: float = 0.98
    sgd_confidence_threshold: float = 0.70
    knowledge_min_similarity: float = 0.58
    knowledge_metadata_similarity: float = 0.68
    knowledge_direct_similarity: float = 0.90
    knowledge_direct_category_confidence: float = 0.82

    primary_model: str = "finetune"
    zeroshot_confidence_threshold: float = 0.60
    use_zeroshot: bool = False

    # legacy knobs kept for backward compatibility with older env files.
    bert_confidence_threshold: float = 0.60
    use_bert: bool = False

    batch_size: int = 256

    model_config = {"env_prefix": "TC_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
