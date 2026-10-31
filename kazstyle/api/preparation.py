"""Portable corpus-preparation API; shares the real cleaner, never loads models."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from kazstyle.demo import prepare
from kazstyle.data.quality import CLEANING_VERSION

app = FastAPI(title="KazStyle Corpus Preparation", version="1.0.0",
              description="Text cleaning and exact deduplication. No style prediction or training.")


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id: str = Field(min_length=1, max_length=100, pattern=r"\S")
    text: str = Field(max_length=10000)


class Batch(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    records: list[Record] = Field(min_length=1, max_length=20)


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ready", "component": "corpus-preparation",
            "cleaning_version": CLEANING_VERSION, "model_loaded": False}


@app.post("/api/prepare")
def prepare_batch(batch: Batch):
    try:
        records, summary = prepare([record.model_dump() for record in batch.records])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # prepare() also serves a synthetic CLI demo; this endpoint accepts user data.
    summary["purpose"] = "Text cleaning and exact deduplication; not a style benchmark"
    return {"records": records, "summary": summary}
