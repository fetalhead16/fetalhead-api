from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


@dataclass
class LoadedImage:
    rgb: np.ndarray
    grayscale: np.ndarray
    pixel_spacing_mm: Optional[float]
    pixel_spacing_source: str
    notes: list[str]


class UltrasoundAnalyzer:
    IMAGE_ABNORMAL_THRESHOLD = 0.70
    MIN_BIOMETRY_CONFIDENCE = 0.62
    MIN_PLAUSIBLE_CI = 60.0
    MAX_PLAUSIBLE_CI = 95.0

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.model_dir = project_root / "models"
        self.classifier = None
        self.scaler = None
        self.image_classifier = None
        self.image_scaler = None
        self._load_classifier()

    def _load_classifier(self) -> None:
        if joblib is None:
            return

        model_path = self.model_dir / "random_forest.joblib"
        scaler_path = self.model_dir / "feature_scaler.joblib"
        image_model_path = self.model_dir / "image_random_forest.joblib"
        image_scaler_path = self.model_dir / "image_feature_scaler.joblib"

        if model_path.exists():
            self.classifier = joblib.load(model_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        if image_model_path.exists():
            self.image_classifier = joblib.load(image_model_path)
        if image_scaler_path.exists():
            self.image_scaler = joblib.load(image_scaler_path)

    def analyze(
        self,
        file_bytes: bytes,
        filename: str,
        pixel_spacing_mm: Optional[float] = None,
        gestational_age_weeks: Optional[int] = None,
    ) -> dict[str, Any]:
        loaded = self._load_image(file_bytes, filename, pixel_spacing_mm)
        preprocessed = self._preprocess(loaded.grayscale)
        contour, ellipse, quality = self._extract_head_contour(preprocessed)

        if contour is None or ellipse is None:
            raise ValueError(
                "The app could not isolate a fetal head outline from this image. Try a clearer frame or upload a DICOM scan."
            )

        if quality["confidence"] < 0.48:
            raise ValueError(
                "This frame does not resemble a reliable fetal head measurement plane. Please upload a clearer fetal head cross-section."
            )

        measurements = self._calculate_measurements(ellipse, loaded.pixel_spacing_mm)

        if quality["confidence"] < self.MIN_BIOMETRY_CONFIDENCE:
            raise ValueError(
                "This upload may be useful for screening, but it is not reliable enough for medical fetal head biometry. Please upload a clearer standard head plane."
            )

        if not self.MIN_PLAUSIBLE_CI <= measurements["ci"]["value"] <= self.MAX_PLAUSIBLE_CI:
            raise ValueError(
                "The detected contour is not clinically plausible for a standard fetal head biometry plane. Please upload a calibrated DICOM or a clearer fetal head cross-section."
            )

        if loaded.pixel_spacing_mm is None:
            raise ValueError(
                "True medical biometry in mm requires calibration metadata. Please upload a DICOM scan with PixelSpacing so the app can report HC, BPD, OFD, and HA in millimeters."
            )

        assessment = self._build_assessment(
            measurements=measurements,
            quality=quality,
            gestational_age_weeks=gestational_age_weeks,
            absolute_measurements=loaded.pixel_spacing_mm is not None,
            source_frame=loaded.grayscale,
        )
        previews = self._build_previews(loaded.rgb, preprocessed, contour, ellipse)

        notes = list(loaded.notes)
        if loaded.pixel_spacing_mm is None:
            notes.append(
                "Absolute millimeter values need image calibration. Enter pixel spacing manually or upload a DICOM file with PixelSpacing metadata."
            )
        if gestational_age_weeks is None:
            notes.append(
                "Gestational age was not provided, so the app did not attempt age-normalized abnormality screening."
            )
        notes = self._unique_notes(notes)

        height, width = loaded.grayscale.shape
        return {
            "filename": filename,
            "image_size": [width, height],
            "calibration": {
                "pixel_spacing_mm": loaded.pixel_spacing_mm,
                "source": loaded.pixel_spacing_source,
                "absolute_measurements": loaded.pixel_spacing_mm is not None,
            },
            "quality": quality,
            "measurements": measurements,
            "assessment": assessment,
            "notes": notes,
            "previews": previews,
        }

    def _load_image(
        self,
        file_bytes: bytes,
        filename: str,
        manual_pixel_spacing_mm: Optional[float],
    ) -> LoadedImage:
        suffix = Path(filename).suffix.lower()
        if suffix in {".dcm", ".dicom"}:
            return self._load_dicom_image(file_bytes, manual_pixel_spacing_mm)

        rgb = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
        grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return LoadedImage(
            rgb=rgb,
            grayscale=grayscale,
            pixel_spacing_mm=manual_pixel_spacing_mm,
            pixel_spacing_source="manual" if manual_pixel_spacing_mm is not None else "not_provided",
            notes=[],
        )

    def _load_dicom_image(
        self,
        file_bytes: bytes,
        manual_pixel_spacing_mm: Optional[float],
    ) -> LoadedImage:
        if pydicom is None:
            raise ValueError("DICOM support is unavailable because pydicom is not installed.")

        dataset = pydicom.dcmread(io.BytesIO(file_bytes))
        pixels = dataset.pixel_array.astype(np.float32)

        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixels = (pixels * slope) + intercept

        if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            pixels = pixels.max() - pixels

        pixels = self._normalize_to_uint8(pixels)
        rgb = np.stack([pixels, pixels, pixels], axis=-1)

        dicom_spacing = None
        if hasattr(dataset, "PixelSpacing") and len(dataset.PixelSpacing) >= 1:
            try:
                dicom_spacing = float(dataset.PixelSpacing[0])
            except (TypeError, ValueError):
                dicom_spacing = None

        final_spacing = manual_pixel_spacing_mm if manual_pixel_spacing_mm is not None else dicom_spacing
        spacing_source = "manual" if manual_pixel_spacing_mm is not None else "dicom" if dicom_spacing else "not_provided"
        notes = []
        if dicom_spacing is None and manual_pixel_spacing_mm is None:
            notes.append("This DICOM file did not expose PixelSpacing metadata.")

        return LoadedImage(
            rgb=rgb,
            grayscale=pixels,
            pixel_spacing_mm=final_spacing,
            pixel_spacing_source=spacing_source,
            notes=notes,
        )

    def _preprocess(self, grayscale: np.ndarray) -> np.ndarray:
        normalized = self._normalize_to_uint8(grayscale)
        median = cv2.medianBlur(normalized, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(median)
        return cv2.GaussianBlur(clahe, (5, 5), 0)

    def _extract_head_contour(
        self,
        preprocessed: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[tuple[Any, ...]], dict[str, Any]]:
        best_score = -1.0
        best_contour = None
        best_ellipse = None
        best_quality = {
            "confidence": 0.0,
            "contour_points": 0,
            "fit_score": 0.0,
            "center_offset_px": 0.0,
        }

        for candidate in self._candidate_masks(preprocessed):
            contours, _ = cv2.findContours(candidate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                score, ellipse, quality = self._score_contour(contour, preprocessed.shape)
                if score > best_score:
                    best_score = score
                    best_contour = contour
                    best_ellipse = ellipse
                    best_quality = quality

        return best_contour, best_ellipse, best_quality

    def _candidate_masks(self, preprocessed: np.ndarray) -> list[np.ndarray]:
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        _, otsu = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            preprocessed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            2,
        )
        edges = cv2.Canny(preprocessed, 50, 150)
        edges = cv2.dilate(edges, kernel_small, iterations=1)

        masks = []
        for raw_mask in [otsu, cv2.bitwise_not(otsu), adaptive, cv2.bitwise_not(adaptive), edges]:
            refined = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_small, iterations=1)
            masks.append(refined)
        return masks

    def _score_contour(
        self,
        contour: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[float, Optional[tuple[Any, ...]], dict[str, Any]]:
        fallback = {
            "confidence": 0.0,
            "contour_points": int(len(contour)) if contour is not None else 0,
            "fit_score": 0.0,
            "center_offset_px": 0.0,
        }

        if contour is None or len(contour) < 5:
            return -1.0, None, fallback

        image_height, image_width = shape
        image_area = float(image_height * image_width)
        contour_area = float(cv2.contourArea(contour))
        relative_area = contour_area / image_area
        if relative_area < 0.025 or relative_area > 0.50:
            return -1.0, None, fallback

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis_a, axis_b), _ = ellipse
        major_axis = max(float(axis_a), float(axis_b))
        minor_axis = min(float(axis_a), float(axis_b))
        if major_axis < 40 or minor_axis < 20:
            return -1.0, None, fallback

        axis_ratio = minor_axis / major_axis
        if axis_ratio < 0.35 or axis_ratio > 1.0:
            return -1.0, None, fallback

        min_dimension = float(min(image_width, image_height))
        if major_axis > min_dimension * 0.92:
            return -1.0, None, fallback

        ellipse_area = math.pi * (major_axis / 2.0) * (minor_axis / 2.0)
        ellipse_coverage = ellipse_area / image_area
        if ellipse_coverage > 0.42:
            return -1.0, None, fallback

        border_contact_ratio = self._border_contact_ratio(contour, image_width, image_height)
        if border_contact_ratio > 0.08:
            return -1.0, None, fallback

        fit_score = min(contour_area, ellipse_area) / max(contour_area, ellipse_area)
        center_offset_px = float(np.hypot(cx - (image_width / 2.0), cy - (image_height / 2.0)))
        max_center_offset = float(np.hypot(image_width / 2.0, image_height / 2.0))
        center_score = 1.0 - min(center_offset_px / max_center_offset, 1.0)
        size_score = 1.0 if 0.07 <= relative_area <= 0.34 else 0.6
        edge_margin = min(
            cx - (major_axis / 2.0),
            cy - (minor_axis / 2.0),
            image_width - (cx + (major_axis / 2.0)),
            image_height - (cy + (minor_axis / 2.0)),
        )
        if edge_margin < min_dimension * 0.018:
            return -1.0, None, fallback
        margin_score = 1.0 if edge_margin > min_dimension * 0.05 else 0.55

        confidence = float(
            np.clip(
                (0.45 * fit_score) + (0.25 * center_score) + (0.18 * axis_ratio) + (0.12 * size_score * margin_score),
                0.0,
                1.0,
            )
        )
        return confidence, ellipse, {
            "confidence": round(confidence, 3),
            "contour_points": int(len(contour)),
            "fit_score": round(float(fit_score), 3),
            "center_offset_px": round(center_offset_px, 2),
        }

    def _calculate_measurements(
        self,
        ellipse: tuple[Any, ...],
        pixel_spacing_mm: Optional[float],
    ) -> dict[str, dict[str, Any]]:
        (_, _), (axis_1, axis_2), _ = ellipse
        major_axis = max(float(axis_1), float(axis_2))
        minor_axis = min(float(axis_1), float(axis_2))
        semi_major = major_axis / 2.0
        semi_minor = minor_axis / 2.0

        scale = pixel_spacing_mm if pixel_spacing_mm is not None else 1.0
        length_unit = "mm" if pixel_spacing_mm is not None else "px"
        area_unit = "mm2" if pixel_spacing_mm is not None else "px2"

        ofd = 2.0 * semi_major * scale
        bpd = 2.0 * semi_minor * scale
        hc = math.pi * (3.0 * (semi_major + semi_minor) - math.sqrt((3.0 * semi_major + semi_minor) * (semi_major + 3.0 * semi_minor))) * scale
        ci = (bpd / ofd) * 100.0 if ofd else 0.0
        ha = math.pi * semi_major * semi_minor * (scale**2)

        return {
            "hc": {"label": "Head Circumference", "value": round(hc, 2), "unit": length_unit},
            "bpd": {"label": "Biparietal Diameter", "value": round(bpd, 2), "unit": length_unit},
            "ofd": {"label": "Occipitofrontal Diameter", "value": round(ofd, 2), "unit": length_unit},
            "ci": {"label": "Cephalic Index", "value": round(ci, 2), "unit": "%"},
            "ha": {"label": "Head Area", "value": round(ha, 2), "unit": area_unit},
        }

    def _build_assessment(
        self,
        measurements: dict[str, dict[str, Any]],
        quality: dict[str, Any],
        gestational_age_weeks: Optional[int],
        absolute_measurements: bool,
        source_frame: Optional[np.ndarray],
    ) -> dict[str, Any]:
        if self.image_classifier is not None:
            image_features = self._extract_image_features_from_frame(source_frame)
            if image_features is not None:
                features = image_features.reshape(1, -1)
                features = self.image_scaler.transform(features) if self.image_scaler is not None else features
                abnormal_probability = float(self.image_classifier.predict_proba(features)[0][1])
                prediction = int(abnormal_probability >= self.IMAGE_ABNORMAL_THRESHOLD)
                return {
                    "classifier_mode": "image_random_forest",
                    "status": "abnormal" if prediction == 1 else "normal",
                    "summary": "Image classifier suggests an abnormal pattern."
                    if prediction == 1
                    else "Image classifier suggests a normal pattern.",
                    "notes": [
                        "Trained image Random Forest weights were loaded from the local models directory.",
                        f"Abnormality probability: {abnormal_probability:.2f}.",
                        "Use this output as a screening aid, not a clinical diagnosis.",
                    ],
                }

        features = np.array(
            [
                measurements["hc"]["value"],
                measurements["bpd"]["value"],
                measurements["ofd"]["value"],
                measurements["ci"]["value"],
                measurements["ha"]["value"],
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        if self.classifier is not None:
            features = self.scaler.transform(features) if self.scaler is not None else features
            prediction = int(self.classifier.predict(features)[0])
            return {
                "classifier_mode": "trained_random_forest",
                "status": "abnormal" if prediction == 1 else "normal",
                "summary": "Random Forest suggests an abnormal pattern." if prediction == 1 else "Random Forest suggests a normal pattern.",
                "notes": [
                    "Trained Random Forest weights were loaded from the local models directory.",
                    "Use this output as a screening aid, not a clinical diagnosis.",
                ],
            }

        ci_value = measurements["ci"]["value"]
        notes = [
            "No trained Random Forest weights were found, so the live deployment used a geometric fallback instead of the paper's classifier."
        ]
        if quality["confidence"] < 0.55:
            notes.append("Segmentation confidence is low. Review the overlay before trusting the measurements.")

        if quality["confidence"] < 0.58:
            status = "invalid_plane"
            summary = "This frame does not resemble a standard fetal head plane strongly enough for reliable biometric use."
        elif ci_value < 74.0:
            status = "review_recommended"
            summary = "The cephalic index is lower than the demo range, so a review is recommended."
        elif ci_value > 85.0:
            status = "review_recommended"
            summary = "The cephalic index is higher than the demo range, so a review is recommended."
        else:
            status = "no_shape_flag"
            summary = "No obvious shape anomaly was flagged by the demo heuristic."

        return {
            "classifier_mode": "heuristic_demo",
            "status": status,
            "summary": summary,
            "notes": self._unique_notes(notes),
        }

    def _extract_image_features_from_frame(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        grayscale = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale, (256, 256), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)

        hist = cv2.calcHist([blurred], [0], None, [16], [0, 256]).flatten().astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()

        edges = cv2.Canny(blurred, 50, 150)
        edge_density = float(edges.mean() / 255.0)
        lap_var = float(cv2.Laplacian(blurred, cv2.CV_64F).var())

        mean = float(blurred.mean())
        std = float(blurred.std())
        p10 = float(np.percentile(blurred, 10))
        p90 = float(np.percentile(blurred, 90))
        contrast = p90 - p10

        features = np.concatenate(
            [
                np.array([mean, std, p10, p90, contrast, edge_density, lap_var], dtype=np.float32),
                hist,
            ]
        )
        return features

    def _build_previews(
        self,
        original_rgb: np.ndarray,
        preprocessed: np.ndarray,
        contour: np.ndarray,
        ellipse: tuple[Any, ...],
    ) -> dict[str, str]:
        mask = np.zeros(preprocessed.shape, dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, thickness=-1)

        overlay_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, [contour], -1, (57, 211, 83), 2)
        cv2.ellipse(overlay_bgr, ellipse, (255, 168, 0), 3)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        return {
            "original": self._to_data_url(original_rgb),
            "preprocessed": self._to_data_url(preprocessed),
            "mask": self._to_data_url(mask),
            "overlay": self._to_data_url(overlay_rgb),
        }

    def _to_data_url(self, image: np.ndarray, max_preview_size: int = 900) -> str:
        pil_image = Image.fromarray(image.astype(np.uint8))
        if max(pil_image.size) > max_preview_size:
            pil_image.thumbnail((max_preview_size, max_preview_size))

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _normalize_to_uint8(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float32)
        min_value = float(pixels.min())
        max_value = float(pixels.max())
        if math.isclose(max_value, min_value):
            return np.zeros_like(pixels, dtype=np.uint8)
        normalized = (pixels - min_value) / (max_value - min_value)
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

    def _border_contact_ratio(self, contour: np.ndarray, image_width: int, image_height: int) -> float:
        margin = max(8, int(min(image_width, image_height) * 0.025))
        points = contour.reshape(-1, 2)
        contact = (
            (points[:, 0] <= margin)
            | (points[:, 1] <= margin)
            | (points[:, 0] >= image_width - margin)
            | (points[:, 1] >= image_height - margin)
        )
        return float(np.mean(contact))

    def _unique_notes(self, notes: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for note in notes:
            if note and note not in seen:
                seen.add(note)
                ordered.append(note)
        return ordered


analyzer = UltrasoundAnalyzer()
