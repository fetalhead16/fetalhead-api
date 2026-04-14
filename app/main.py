from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import AnalysisResponse
from app.services.analysis import analyzer

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Fetal Head Biometrics App",
    description="Upload an ultrasound image and extract ellipse-based fetal head biometrics.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
