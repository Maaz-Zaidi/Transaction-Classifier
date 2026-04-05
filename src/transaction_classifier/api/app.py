"""fastapi app setup."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from transaction_classifier.api.routes import router, set_ensemble
from transaction_classifier.config import settings
from transaction_classifier.knowledge.merchant_kb import MerchantKnowledgeBase
from transaction_classifier.models.ensemble import Ensemble
from transaction_classifier.rules.engine import RulesEngine


def _load_sgd_model():
    from transaction_classifier.models.sgd_model import SGDModel

    sgd_model = SGDModel()
    sgd_path = settings.model_dir / "sgd"
    if (sgd_path / "sgd_pipeline.joblib").exists():
        print(f"Loading SGD model from {sgd_path}...")
        sgd_model.load(sgd_path)
        print("SGD model loaded.")
        return sgd_model

    print(f"WARNING: No SGD model found at {sgd_path}.")
    return None


def _load_primary_model():
    primary_name = settings.primary_model.lower()

    if primary_name == "finetune":
        try:
            from transaction_classifier.models.finetune_model import FineTuneModel
        except ImportError as exc:
            print(f"WARNING: FineTune dependencies unavailable: {exc}")
            return None

        finetune_path = settings.model_dir / "finetune"
        if (finetune_path / "model").exists():
            print(f"Loading fine-tune model from {finetune_path}...")
            model = FineTuneModel()
            model.load(finetune_path)
            print("Fine-tune model loaded.")
            return model

        print(f"WARNING: No fine-tune model found at {finetune_path}.")
        return None

    if primary_name == "canine":
        try:
            from transaction_classifier.models.canine_model import CanineModel
        except ImportError as exc:
            print(f"WARNING: CANINE dependencies unavailable: {exc}")
            return None

        canine_path = settings.model_dir / "canine"
        if (canine_path / "model").exists():
            print(f"Loading CANINE model from {canine_path}...")
            model = CanineModel()
            model.load(canine_path)
            print("CANINE model loaded.")
            return model

        print(f"WARNING: No CANINE model found at {canine_path}.")
        return None

    print(f"WARNING: Unsupported primary model '{settings.primary_model}', using SGD fallback.")
    return None


def _load_zeroshot_model():
    if not (settings.use_zeroshot or settings.use_bert):
        return None

    try:
        from transaction_classifier.models.zeroshot_model import ZeroShotModel
    except ImportError as exc:
        print(f"WARNING: Zero-shot dependencies unavailable: {exc}")
        return None

    try:
        print("Loading zero-shot fallback model...")
        model = ZeroShotModel()
        model.load()
        print("Zero-shot model loaded.")
        return model
    except Exception as exc:
        print(f"WARNING: Zero-shot model failed to load: {exc}")
        return None


def _load_knowledge_base():
    store_path = None
    if settings.knowledge_store_path.exists():
        store_path = settings.knowledge_store_path
    elif settings.knowledge_base_path.exists():
        store_path = settings.knowledge_base_path

    if store_path is None:
        print(
            "Knowledge base not found at "
            f"{settings.knowledge_store_path} or {settings.knowledge_base_path}."
        )
        return None

    kb = MerchantKnowledgeBase()
    kb.load(store_path)
    print(
        f"Knowledge base loaded with {kb.size} entries "
        f"(chroma_ready={kb.chroma_ready})."
    )
    return kb


def _derive_version(ensemble: Ensemble) -> str:
    primary = ensemble.primary_source
    model = None
    if primary == "finetune":
        model = ensemble.finetune_model
    elif primary == "canine":
        model = ensemble.canine_model

    if model is None:
        return "0.1.0"

    metadata = getattr(model, "metadata", {}) or {}
    trained_at = metadata.get("trained_at")
    if trained_at:
        return f"{primary}@{trained_at}"
    return primary


@asynccontextmanager
async def lifespan(app: FastAPI):
    """load models on startup and clean up on shutdown."""
    rules_engine = RulesEngine()
    sgd_model = _load_sgd_model()
    primary_model = _load_primary_model()
    knowledge_base = _load_knowledge_base()
    zeroshot_model = _load_zeroshot_model()

    ensemble = Ensemble(
        rules_engine=rules_engine,
        finetune_model=primary_model if settings.primary_model.lower() == "finetune" else None,
        canine_model=primary_model if settings.primary_model.lower() == "canine" else None,
        knowledge_base=knowledge_base,
        zeroshot_model=zeroshot_model,
        primary_model_name=settings.primary_model,
        sgd_model=sgd_model,
    )
    set_ensemble(ensemble, version=_derive_version(ensemble))
    print("Ensemble ready.")

    yield  # app runs

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
