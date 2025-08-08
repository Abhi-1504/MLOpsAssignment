from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class HouseFeatures(BaseModel):
    """
    Data model for housing features used in prediction.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        }
    )
    
    longitude: float = Field(..., description="Longitude coordinate")
    latitude: float = Field(..., description="Latitude coordinate") 
    housing_median_age: float = Field(..., description="Median age of housing in the area")
    total_rooms: float = Field(..., description="Total number of rooms")
    total_bedrooms: float = Field(..., description="Total number of bedrooms")
    population: float = Field(..., description="Population in the area")
    households: float = Field(..., description="Number of households")
    median_income: float = Field(..., description="Median income in the area")
    ocean_proximity: Literal["NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "INLAND", "ISLAND"] = Field(
        ..., 
        description="Proximity to ocean"
    )