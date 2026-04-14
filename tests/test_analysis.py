from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.services.analysis import UltrasoundAnalyzer


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


if __name__ == "__main__":
    unittest.main()
