"""Health / liveness response models."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    db: str
    version: str
    #: Which embedder the server actually built, and whether it is the one that
    #: was asked for. A server serving keyless vectors against a real index
    #: answers every semantic query with nothing while looking healthy, so the
    #: degradation has to be visible to whatever is watching the container.
    embedder: str | None = None
    embedder_degraded: bool = False
    embedder_reason: str | None = None


class CoordinatorHealthResponse(BaseModel):
    sql_pages: int | None
    sql_decisions: int | None
    vector_count: int | None  # total page + decision vectors
    vector_page_count: int | None
    vector_decision_count: int | None
    graph_nodes: int | None
    drift_pct: float | None  # alias of page_drift_pct (backwards compat)
    page_drift_pct: float | None  # wiki_pages <-> page vectors
    decision_drift_pct: float | None  # decision_records <-> decision vectors
    status: str  # "ok" | "warning" | "critical"
    detail: str | None = None  # human-readable explanation of the status
