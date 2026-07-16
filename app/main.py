"""FastAPI entrypoint for the Smart Day Planner."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

from .models import Schedule
from .monitoring import LOGGER_NAME, configure_logging
from .pipeline import run

load_dotenv()


class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=10_000,
        examples=[
            "Wake at 07:00, standup at 10:00 for 30 minutes, "
            "deep work for 3 hours, gym at 18:00, sleep at 23:00"
        ],
    )


class ScheduleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    model: str
    latency_ms: int
    attempts: int
    token_usage: dict[str, int]
    schedule: Schedule
    plan: dict[str, Any]


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logging.getLogger(LOGGER_NAME).info("Smart Scheduler API started")
    yield


app = FastAPI(
    title="Text-to-Schedule Smart Day Planner",
    description="Turn natural-language plans into a validated daily schedule.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "name": "Text-to-Schedule Smart Day Planner",
        "health": "/health",
        "interactive_docs": "/docs",
        "schedule_endpoint": "POST /schedule",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/schedule", response_model=ScheduleResponse, tags=["scheduler"])
async def create_schedule(request: ScheduleRequest) -> ScheduleResponse:
    result = await run_in_threadpool(run, request.text)
    if not result.success or result.parse.schedule is None or result.plan is None:
        errors = result.parse.errors
        if any(error.startswith("Configuration error") for error in errors):
            status_code = 503
        elif any(error.startswith("Provider error") for error in errors):
            status_code = 502
        else:
            status_code = 422
        raise HTTPException(
            status_code=status_code,
            detail={"run_id": result.run_id, "errors": errors},
        )

    return ScheduleResponse(
        run_id=result.run_id,
        model=result.parse.model,
        latency_ms=result.parse.latency_ms,
        attempts=result.parse.attempts,
        token_usage={
            "prompt": result.parse.prompt_tokens,
            "completion": result.parse.completion_tokens,
            "total": result.parse.total_tokens,
        },
        schedule=result.parse.schedule,
        plan=asdict(result.plan),
    )
