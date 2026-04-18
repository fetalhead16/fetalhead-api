from __future__ import annotations

import unittest
from unittest import mock

import cv2
import numpy as np

from app.services.analysis import LoadedImage, UltrasoundAnalyzer


class UltrasoundAnalyzerTests(unittest.TestCase):
    def test_extracts_metrics_from_synthetic_ellipse(self) -> None:
        image = np.zeros((520, 680, 3), dtype=np.uint8)
        center = (340, 260)
        axes = (155, 112)

        cv2.ellipse(image, center, axes, 10, 0, 360, (238, 238, 238), thickness=14)

        rng = np.random.default_rng(21)
        noise = rng.normal(0, 12, size=image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        success, encoded = cv2.imencode(".png", noisy)
        self.assertTrue(success)

        analyzer = UltrasoundAnalyzer()
        result = analyzer.analyze(
            file_bytes=encoded.tobytes(),
            filename="synthetic.png",
            pixel_spacing_mm=0.2,
            gestational_age_weeks=24,
        )

        self.assertGreater(result["quality"]["confidence"], 0.45)
        self.assertAlmostEqual(result["measurements"]["ofd"]["value"], 62.0, delta=10.0)
        self.assertAlmostEqual(result["measurements"]["bpd"]["value"], 44.8, delta=10.0)
        self.assertAlmostEqual(result["measurements"]["ci"]["value"], 72.26, delta=10.0)

    def test_uses_demo_spacing_for_uncalibrated_uploads(self) -> None:
        image = np.zeros((520, 680, 3), dtype=np.uint8)
        center = (340, 260)
        axes = (155, 112)

        cv2.ellipse(image, center, axes, 10, 0, 360, (238, 238, 238), thickness=14)
        rng = np.random.default_rng(21)
        noise = rng.normal(0, 12, size=image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        success, encoded = cv2.imencode(".png", noisy)
        self.assertTrue(success)

        analyzer = UltrasoundAnalyzer()
        result = analyzer.analyze(
            file_bytes=encoded.tobytes(),
            filename="synthetic.png",
            pixel_spacing_mm=None,
            gestational_age_weeks=24,
        )

        self.assertEqual(result["measurements"]["bpd"]["unit"], "mm")
        self.assertEqual(result["measurements"]["ha"]["unit"], "mm2")
        self.assertEqual(result["calibration"]["source"], "demo_default")

    def test_returns_warning_note_for_borderline_biometry_confidence(self) -> None:
        analyzer = UltrasoundAnalyzer()
        grayscale = np.zeros((120, 120), dtype=np.uint8)
        rgb = np.zeros((120, 120, 3), dtype=np.uint8)
        contour = np.array([[[10, 10]], [[10, 60]], [[60, 60]], [[60, 10]], [[35, 5]]], dtype=np.int32)
        ellipse = ((60.0, 60.0), (80.0, 60.0), 0.0)
        quality = {
            "confidence": 0.55,
            "contour_points": 5,
            "fit_score": 0.82,
            "center_offset_px": 3.5,
        }

        with (
            mock.patch.object(
                analyzer,
                "_load_image",
                return_value=LoadedImage(
                    rgb=rgb,
                    grayscale=grayscale,
                    pixel_spacing_mm=0.2,
                    pixel_spacing_source="manual",
                    notes=[],
                ),
            ),
            mock.patch.object(analyzer, "_preprocess", return_value=grayscale),
            mock.patch.object(analyzer, "_extract_head_contour", return_value=(contour, ellipse, quality)),
            mock.patch.object(
                analyzer,
                "_build_assessment",
                return_value={
                    "classifier_mode": "heuristic_demo",
                    "status": "review_recommended",
                    "summary": "Review recommended.",
                    "notes": [],
                },
            ),
            mock.patch.object(
                analyzer,
                "_build_previews",
                return_value={"original": "o", "preprocessed": "p", "mask": "m", "overlay": "ov"},
            ),
        ):
            result = analyzer.analyze(
                file_bytes=b"ignored",
                filename="synthetic.png",
                pixel_spacing_mm=0.2,
                gestational_age_weeks=24,
            )

        self.assertIn("measurements", result)
        self.assertTrue(
            any("not reliable enough for medical fetal head biometry" in note for note in result["notes"]),
            msg=result["notes"],
        )


if __name__ == "__main__":
    unittest.main()
