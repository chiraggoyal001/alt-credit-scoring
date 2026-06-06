from __future__ import annotations

import argparse
import json
from pathlib import Path

from altcredit.config import DEFAULT_ROWS_PER_MODEL, MODEL_TYPES, RANDOM_SEED
from altcredit.comparison import evaluate_uploaded_testing_data
from altcredit.data import generate_all_datasets, load_generated_datasets
from altcredit.json_io import load_json_file
from altcredit.models import train_all_models
from altcredit.scoring import default_feature_payload, load_model_bundle, score_features, score_trace
from altcredit.templates import normalize_testing_json, score_json_template, testing_json_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternative credit scoring MVP simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate", help="Generate unbiased synthetic datasets")
    gen.add_argument("--rows", type=int, default=DEFAULT_ROWS_PER_MODEL)
    gen.add_argument("--seed", type=int, default=RANDOM_SEED)
    gen.add_argument("--out", default="data/synthetic")

    train = subparsers.add_parser("train", help="Train all four alt-credit models")
    train.add_argument("--data", default="data/synthetic")
    train.add_argument("--models-dir", default="artifacts/models")
    train.add_argument("--reports-dir", default="artifacts/reports")
    train.add_argument("--fast", action="store_true", help="Use smaller estimators for smoke tests")

    run_all = subparsers.add_parser("run-all", help="Generate data, train models, and write reports")
    run_all.add_argument("--rows", type=int, default=DEFAULT_ROWS_PER_MODEL)
    run_all.add_argument("--seed", type=int, default=RANDOM_SEED)
    run_all.add_argument("--data", default="data/synthetic")
    run_all.add_argument("--models-dir", default="artifacts/models")
    run_all.add_argument("--reports-dir", default="artifacts/reports")
    run_all.add_argument("--fast", action="store_true", help="Use smaller estimators for smoke tests")

    score = subparsers.add_parser("score", help="Score a default median synthetic applicant")
    score.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    score.add_argument("--models-dir", default="artifacts/models")
    score.add_argument("--trace", action="store_true", help="Show stage-by-stage score trace")

    template = subparsers.add_parser("template", help="Print JSON templates")
    template.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    template.add_argument("--kind", choices=("score", "testing-data"), default="score")
    template.add_argument("--models-dir", default="artifacts/models")

    evaluate = subparsers.add_parser("evaluate-json", help="Evaluate uploaded/replacement testing JSON")
    evaluate.add_argument("--file", required=True)
    evaluate.add_argument("--model-type", choices=MODEL_TYPES)
    evaluate.add_argument("--models-dir", default="artifacts/models")
    evaluate.add_argument("--reports-dir", default="artifacts/reports")

    args = parser.parse_args()

    if args.command == "generate":
        generate_all_datasets(rows_per_model=args.rows, seed=args.seed, out_dir=args.out)
        print(f"Generated {args.rows} rows per model in {args.out}")
    elif args.command == "train":
        datasets = load_generated_datasets(args.data)
        results = train_all_models(datasets, args.models_dir, args.reports_dir, fast=args.fast)
        print(json.dumps(_compact_results(results), indent=2))
    elif args.command == "run-all":
        datasets = generate_all_datasets(rows_per_model=args.rows, seed=args.seed, out_dir=args.data)
        results = train_all_models(datasets, args.models_dir, args.reports_dir, fast=args.fast)
        print(json.dumps(_compact_results(results), indent=2))
    elif args.command == "score":
        bundle = load_model_bundle(args.model_type, args.models_dir)
        payload = default_feature_payload(bundle)
        output = score_trace(bundle, payload) if args.trace else score_features(bundle, payload)
        print(json.dumps(output, indent=2))
    elif args.command == "template":
        bundle = load_model_bundle(args.model_type, args.models_dir)
        payload = score_json_template(bundle) if args.kind == "score" else testing_json_template(bundle)
        print(json.dumps(payload, indent=2))
    elif args.command == "evaluate-json":
        payload = load_json_file(args.file)
        model_type, records = normalize_testing_json(payload, selected_model_type=args.model_type)
        bundle = load_model_bundle(model_type, args.models_dir)
        output = evaluate_uploaded_testing_data(bundle, records, reports_dir=args.reports_dir)
        print(json.dumps(output["comparison_table"], indent=2))


def _compact_results(results: dict) -> dict:
    compact = {}
    for model_type, result in results.items():
        metrics = result["metrics"]
        compact[model_type] = {
            "roc_auc": round(metrics["roc_auc"], 4),
            "gini": round(metrics["gini"], 4),
            "approval_rate": round(metrics["approval_rate"], 4),
            "approved_default_rate": round(metrics["approved_default_rate"], 4),
            "cibil_like_approval_rate": round(metrics["cibil_like_approval_rate"], 4),
            "cibil_like_approved_default_rate": round(metrics["cibil_like_approved_default_rate"], 4),
            "score_threshold": round(metrics["score_threshold"], 1),
            "model_path": str(Path(result["model_path"])),
        }
    return compact


if __name__ == "__main__":
    main()
