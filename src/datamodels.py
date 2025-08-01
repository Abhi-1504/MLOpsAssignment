from pydantic import BaseModel, Field


# datamodel for predict endpoint
class HouseFeatures(BaseModel):
    longitude: float = Field(
        ...,
        description="Longitude of the house location (West is negative).",
        example=-122.23,
    )
    latitude: float = Field(
        ...,
        description="Latitude of the house location (North is positive).",
        example=37.88,
    )
    housing_median_age: float = Field(
        ..., description="Median age of a house within a block.", example=41.0
    )
    total_rooms: float = Field(
        ..., description="Total number of rooms within a block.", example=880.0
    )
    total_bedrooms: float = Field(
        ..., description="Total number of bedrooms within a block.", example=129.0
    )
    population: float = Field(
        ...,
        description="Total number of people residing within a block.",
        example=322.0,
    )
    households: float = Field(
        ...,
        description="Total number of households, a group of people residing within a home unit, for a block.",
        example=126.0,
    )
    median_income: float = Field(
        ...,
        description="Median income for households within a block (in tens of thousands of US Dollars).",
        example=8.3252,
    )
    ocean_proximity: str = Field(
        ..., description="Location of the house w.r.t ocean/sea.", example="NEAR BAY"
    )
