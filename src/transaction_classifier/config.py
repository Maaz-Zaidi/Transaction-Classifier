from pathlib import Path

from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    data_dir: Path = _PROJECT_ROOT / "data"
    model_dir: Path = _PROJECT_ROOT / "models"
    db_path: Path = _PROJECT_ROOT / "data" / "corrections.db"

    rules_confidence: float = 0.98
    sgd_confidence_threshold: float = 0.70
    bert_confidence_threshold: float = 0.60
    use_bert: bool = False

    batch_size: int = 256

    model_config = {"env_prefix": "TC_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
