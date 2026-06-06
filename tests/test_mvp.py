from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

from altcredit.config import MODEL_TYPES
from altcredit.comparison import evaluate_uploaded_testing_data
from altcredit.data import generate_all_datasets
from altcredit.features import get_model_spec
from altcredit.json_io import loads_json_bytes
from altcredit.models import train_one_model
from altcredit.scoring import default_feature_payload, load_model_bundle, score_features, score_trace
from altcredit.templates import score_json_template, testing_json_template


class AltCreditMvpTests(unittest.TestCase):
    def test_generated_data_has_targets_and_excludes_protected_features(self) -> None:
        datasets = generate_all_datasets(rows_per_model=320, seed=123)
        for model_type in MODEL_TYPES:
            spec = get_model_spec(model_type)
            df = datasets[model_type]
            self.assertIn(spec.target_column, df.columns)
            self.assertGreater(df[spec.target_column].nunique(), 1)
            for protected in spec.protected_columns:
                self.assertIn(protected, df.columns)
                self.assertNotIn(protected, spec.feature_columns)

    def test_training_and_scoring_one_model(self) -> None:
        datasets = generate_all_datasets(rows_per_model=260, seed=321)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_one_model(
                datasets["retail_active"],
                "retail_active",
                models_dir=tmp_path / "models",
                reports_dir=tmp_path / "reports",
                fast=True,
            )
            bundle = load_model_bundle("retail_active", tmp_path / "models")
            result = score_features(bundle, default_feature_payload(bundle))
            self.assertIn(result["decision"], {"APPROVED", "MANUAL_REVIEW", "DECLINED"})
            self.assertGreaterEqual(result["score"], 300)
            self.assertLessEqual(result["score"], 900)

            trace = score_trace(bundle, default_feature_payload(bundle))
            self.assertEqual(len(trace["stages"]), 8)
            self.assertIn("cibil_like_benchmark", trace["result"])

            score_template = score_json_template(bundle)
            self.assertEqual(score_template["model_type"], "retail_active")
            self.assertIn("features", score_template)

            testing_template = testing_json_template(bundle)
            evaluation = evaluate_uploaded_testing_data(bundle, testing_template["records"], persist=False)
            self.assertIn("comparison_table", evaluation)
            self.assertGreater(len(evaluation["comparison_table"]), 10)

    def test_json_upload_decoder_accepts_utf16_windows_files(self) -> None:
        payload = '{"model_type": "retail_active", "records": []}'
        decoded = loads_json_bytes(payload.encode("utf-16"))
        self.assertEqual(decoded["model_type"], "retail_active")


if __name__ == "__main__":
    unittest.main()
