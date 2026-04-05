"""API request/response models."""

from pydantic import BaseModel


class TransactionInput(BaseModel):
    description: str
    amount: float | None = None  # not used yet


class ClassifyRequest(BaseModel):
    transactions: list[TransactionInput]


class ClassificationOutput(BaseModel):
    description: str
    category: str
    confidence: float
    source: str
    flagged_for_review: bool


class ClassifyResponse(BaseModel):
    results: list[ClassificationOutput]
    processing_time_ms: float
    model_version: str


class CorrectionRequest(BaseModel):
    description: str
    predicted_category: str
    correct_category: str


class CorrectionResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    sgd_loaded: bool
    bert_loaded: bool
    knowledge_base_loaded: bool = False
    zeroshot_loaded: bool = False
    primary_model: str = "none"
    rules_count: int
