from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import AnalysisResponse, RegistrationRequest, RegistrationResponse
from app.services.analysis import analyzer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REGISTRATION_FILE = DATA_DIR / "registrations.jsonl"
ASSET_FILES = [BASE_DIR / "static" / "css" / "styles.css", BASE_DIR / "static" / "js" / "app.js"]
ASSET_VERSION = str(max(int(path.stat().st_mtime) for path in ASSET_FILES if path.exists()))

app = FastAPI(
    title="Fetal Head Biometrics App",
    description="Upload an ultrasound image and extract ellipse-based fetal head biometrics.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "asset_version": ASSET_VERSION})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/register", response_model=RegistrationResponse)
async def register_interest(payload: RegistrationRequest) -> RegistrationResponse:
    name = payload.name.strip()
    email = payload.email.strip()

    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRATION_FILE.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": name,
                    "email": email,
                    "college": (payload.college or "").strip(),
                    "role": (payload.role or "").strip(),
                    "message": (payload.message or "").strip(),
                }
            )
        )
        handle.write("\n")

    return RegistrationResponse(success=True, message="Registration saved successfully.")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    pixel_spacing_mm: Optional[float] = Form(None),
    gestational_age_weeks: Optional[int] = Form(None),
) -> AnalysisResponse:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Please choose an image file.")

    file_bytes = await image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")

    try:
        result = analyzer.analyze(
            file_bytes=file_bytes,
            filename=image.filename,
            pixel_spacing_mm=pixel_spacing_mm,
            gestational_age_weeks=gestational_age_weeks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Unexpected analysis error: {exc}") from exc

    return AnalysisResponse(**result)
