"""Churn Prediction API.

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from typing import Literal
from litestar import Litestar, get, post
from pydantic import BaseModel

# Import the prediction logic function from your model_utils file
from app.model_utils import predict_churn
from app.logger_setup import setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    num_of_products: int
    has_cr_card: int
    is_active_member: int
    estimated_salary: float
    geography: Literal["Germany", "Spain", "France"]
    gender: Literal["Male", "Female"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@get("/")
async def index() -> str:
    return "Welcome There!"


@get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@post("/predict")
async def predict(data: ChurnRequest) -> dict[str, int]:
    # Extract raw data features exactly in the order the preprocessor pipeline expects:
    # [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender]
    raw_features = [
        data.credit_score,
        data.age,
        data.tenure,
        data.balance,
        data.num_of_products,
        data.has_cr_card,
        data.is_active_member,
        data.estimated_salary,
        data.geography,
        data.gender,
    ]

    # Log incoming payload attributes
    logger.info(f"Incoming prediction features: {raw_features}")

    # Generate inference prediction using your loaded pipeline (returns 0 or 1)
    prediction = predict_churn(raw_features)

    # Log resulting prediction value
    logger.info(f"Prediction output result: {prediction}")

    return {"churn_prediction": prediction}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Litestar(
    route_handlers=[index, health, predict],
)
