"""
Pydantic Schemas for Personalized Recommendation API
=====================================================
Defines interaction schemas and request/response models for Milestone 5
Multi-Signal Personalized Recommendation Intelligence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class InteractionItem(BaseModel):
    """
    Represents a single user interaction event.

    Supported event types:
        - view (weight: 1.0)
        - repeat_view (weight: 1.5)
        - wishlist (weight: 2.0)
        - cart (weight: 3.0)
        - purchase (weight: 5.0)
    """

    product_id: str = Field(..., description="Canonical product ID (e.g. B014I8SSD0)")
    event_type: str = Field(
        "view",
        description="Type of interaction: view, repeat_view, wishlist, cart, purchase",
    )
    timestamp: Optional[Union[str, float, int]] = Field(
        None, description="ISO timestamp or Unix epoch timestamp"
    )
    quantity: Optional[int] = Field(
        1, ge=1, le=100, description="Quantity associated with cart or purchase events"
    )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        valid_events = {"view", "repeat_view", "wishlist", "cart", "purchase"}
        cleaned = v.strip().lower()
        if cleaned not in valid_events:
            # Normalize unknown event types to view rather than crashing
            return "view"
        return cleaned

    @field_validator("product_id")
    @classmethod
    def validate_product_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("product_id cannot be empty")
        return s


class PersonalizedRecommendationRequest(BaseModel):
    """
    POST Request payload for multi-signal personalized recommendations.
    """

    interactions: List[InteractionItem] = Field(
        default_factory=list,
        description="Structured interaction history items, newest-first.",
    )
    history: Optional[List[str]] = Field(
        default=None,
        description="Optional list of simple product_id strings for backward compatibility.",
    )
    k: int = Field(8, ge=1, le=20, description="Number of recommendations to return")
    session_id: str = Field("anonymous", description="Session ID for tracking and logging")


class PersonalizedRecommendationResponse(BaseModel):
    """
    POST Response model for multi-signal personalized recommendations.
    """

    status: str = Field("success", description="Response status ('success' or 'error')")
    mode: str = Field(
        ...,
        description="Recommendation mode: 'personalized', 'cold_start', or 'fallback'",
    )
    history_used: List[str] = Field(
        ..., description="List of validated product IDs used to build the user profile"
    )
    dominant_signal: Optional[str] = Field(
        None, description="Dominant interaction event type influencing recommendations"
    )
    count: int = Field(..., description="Number of items returned in data")
    data: List[Dict[str, Any]] = Field(..., description="List of enriched product dicts")
