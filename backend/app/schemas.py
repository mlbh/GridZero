from pydantic import BaseModel, Field
from typing import List, Dict
from datetime import date

class PredictionRequest(BaseModel):
    # Field allows us to add a description and an example for Swagger (/docs)
    target_date: date = Field(
        ...,
        description="The date for prediction in YYYY-MM-DD format",
        example="2026-03-20"
    )

class PredictionResponse(BaseModel):
    date: date
    generation_prediction: List[float]
    carbon_intensity: List[float]
    unit_gen: str = "MW"
    unit_carbon: str = "gCO2/kWh"

    class Config:
        # This helps if you return numpy arrays; Pydantic will try to serialize them
        from_attributes = True
