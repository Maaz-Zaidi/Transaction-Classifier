"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from transaction_classifier.api.routes import router, set_ensemble
from transaction_classifier.config import settings
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.models.sgd_model import SGDModel
from transaction_classifier.rules.engine import RulesEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    rules_engine = RulesEngine()

    sgd_model = SGDModel()
    sgd_path = settings.model_dir / "sgd"
    if (sgd_path / "sgd_pipeline.joblib").exists():
        print(f"Loading SGD model from {sgd_path}...")
        sgd_model.load(sgd_path)
        print("SGD model loaded.")
    else:
        print(f"WARNING: No SGD model found at {sgd_path}. Run train_sgd.py first.")

    ensemble = Ensemble(
        rules_engine=rules_engine,
        sgd_model=sgd_model,
    )
    set_ensemble(ensemble)
    print("Ensemble ready.")

    yield  # App runs

    print("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Transaction Classifier",
        description="Classify bank transactions into budget categories",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
