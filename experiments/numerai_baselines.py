"""Memory-bounded Ridge/LightGBM comparison on local Numerai data."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import time
from pathlib import Path

# Leave CPU capacity for the desktop, and avoid BLAS oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from numerai_tools.scoring import numerai_corr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_ender_60"


def iter_eras(path: Path, columns: list[str]):
    """Yield complete eras even when they straddle Arrow batch boundaries.

    The source must be sorted chronologically. Fail rather than accidentally
    score fragments of an era as if they were separate eras.
    """
    parquet = pq.ParquetFile(path)
    pending = []
    current = None
    for batch in parquet.iter_batches(batch_size=8192, columns=columns,
                                      use_threads=False):
        # Numerai stores `id` as a physical column with pandas-index metadata.
        # Keep it as an ordinary column while assembling complete eras.
        frame = batch.to_pandas(ignore_metadata=True)
        era_numbers = frame["era"].astype(int).to_numpy()
        if np.any(np.diff(era_numbers) < 0):
            raise ValueError(f"Unsorted eras in {path}")
        for era, group in frame.groupby("era", sort=False):
            number = int(era)
            if current is not None and number < current:
                raise ValueError(f"Unsorted eras across batches in {path}")
            if current is not None and number != current:
                yield current, pd.concat(pending, ignore_index=True)
                pending = []
            current = number
            pending.append(group.copy())
    if pending:
        yield current, pd.concat(pending, ignore_index=True)


def summarize(scores: pd.DataFrame, model_names=None) -> pd.DataFrame:
    records = []
    for name in (["Ridge", "LightGBM"] if model_names is None else model_names):
        values = scores[name].to_numpy(dtype=float)
        std = float(values.std(ddof=0))
        cumulative = np.r_[0.0, np.cumsum(values)]
        drawdown = np.maximum.accumulate(cumulative) - cumulative
        records.append({
            "model": name, "eras": len(values),
            "mean_corr": float(values.mean()), "std_corr": std,
            "corr_sharpe": float(values.mean() / std) if std else None,
            "positive_era_fraction": float((values > 0).mean()),
            "worst_era_corr": float(values.min()),
            "max_drawdown_cumulative_corr": float(drawdown.max()),
        })
    return pd.DataFrame(records).set_index("model")


def run(data_dir: Path, output_dir: Path, rows_per_era: int = 200,
        seed: int = 42, gap_eras: int = 13, trees: int = 500):
    """Run a fixed first baseline, without tuning on validation results."""
    if rows_per_era < 1 or gap_eras < 13 or trees < 1:
        raise ValueError("Require rows_per_era >= 1, gap_eras >= 13, trees >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = ROOT / "downloads" / "numerai" / "baseline_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data_dir / "features.json").read_text())
    features = metadata["feature_sets"]["small"]
    columns = ["id", "era", "data_type", TARGET] + features
    for filename in ["train.parquet", "validation.parquet"]:
        missing = set(columns) - set(pq.ParquetFile(data_dir / filename).schema_arrow.names)
        if missing:
            raise ValueError(f"Missing columns in {filename}: {missing}")

    rng = np.random.default_rng(seed)
    selected = []
    train_era_counts = []
    started = time.monotonic()
    print("Reading a chronological, era-balanced training sample", flush=True)
    for era, frame in iter_eras(data_dir / "train.parquet", columns):
        if not frame["data_type"].eq("train").all():
            raise ValueError("Unexpected data_type in train")
        eligible = frame.dropna(subset=[TARGET])
        if eligible.empty:
            continue
        count = min(rows_per_era, len(eligible))
        indices = rng.choice(len(eligible), size=count, replace=False)
        selected.append(eligible.iloc[indices].copy())
        train_era_counts.append({"era": era, "eligible_rows": len(eligible),
                                 "sampled_rows": count})
        if len(train_era_counts) % 100 == 0:
            print("Sampled", len(train_era_counts), "training eras", flush=True)
    training = pd.concat(selected, ignore_index=True)
    del selected, frame, eligible
    gc.collect()
    if training["id"].duplicated().any():
        raise ValueError("Duplicate training IDs")
    train_max_era = int(training["era"].astype(int).max())
    pd.DataFrame(train_era_counts).to_csv(output_dir / "training_eras.csv", index=False)
    X = training[features].astype(np.float32)
    y = training[TARGET].astype(np.float32)
    print("Training rows:", len(training), "features:", len(features),
          "last era:", train_max_era, flush=True)
    # Both models see identical rows. All learned preprocessing uses train only.
    models = {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"),
                               StandardScaler(), Ridge(alpha=100.0, solver="cholesky")),
        "LightGBM": make_pipeline(SimpleImputer(strategy="median").set_output(transform="pandas"), LGBMRegressor(
            n_estimators=trees, learning_rate=0.03, max_depth=5, num_leaves=31,
            colsample_bytree=0.8, min_child_samples=200, reg_lambda=1.0,
            max_bin=15, random_state=seed, n_jobs=2, verbosity=-1,
            deterministic=True, force_col_wise=True)),
    }
    fit_seconds = {}
    with threadpool_limits(limits=2):
        for name, model in models.items():
            tick = time.monotonic()
            model.fit(X, y)
            fit_seconds[name] = time.monotonic() - tick
            print("Fitted", name, "seconds:", round(fit_seconds[name], 1), flush=True)
            joblib.dump({"model": model, "features": features, "target": TARGET},
                        artifact_dir / f"{name.lower()}.joblib")
    training_info = {
        "rows": len(training), "era_count": len(train_era_counts),
        "first_era": min(item["era"] for item in train_era_counts),
        "last_era": train_max_era,
        "sample_feature_nulls": int(X.isna().sum().sum()),
    }
    del training, X, y
    gc.collect()

    records = []
    skipped = {"gap_rows": 0, "non_validation_rows": 0, "null_target_rows": 0}
    prediction_path = artifact_dir / "validation_predictions.parquet"
    writer = None
    try:
        with threadpool_limits(limits=2):
            for era, frame in iter_eras(data_dir / "validation.parquet", columns):
                if era <= train_max_era + gap_eras:
                    skipped["gap_rows"] += len(frame)
                    continue
                mask = frame["data_type"].eq("validation")
                skipped["non_validation_rows"] += int((~mask).sum())
                frame = frame.loc[mask]
                skipped["null_target_rows"] += int(frame[TARGET].isna().sum())
                frame = frame.dropna(subset=[TARGET]).set_index("id")
                if len(frame) < 2:
                    continue
                if frame.index.has_duplicates:
                    raise ValueError(f"Duplicate validation IDs in era {era}")
                X_era = frame[features].astype(np.float32)
                predictions = pd.DataFrame({name: model.predict(X_era)
                                            for name, model in models.items()},
                                           index=frame.index)
                if not np.isfinite(predictions.to_numpy()).all():
                    raise ValueError(f"Invalid predictions in era {era}")
                if predictions.nunique().min() < 2:
                    raise ValueError(f"Constant predictions in era {era}")
                scores = numerai_corr(predictions, frame[TARGET],
                                      max_filtered_index_ratio=0.0)
                if not np.isfinite(scores.to_numpy()).all():
                    raise ValueError(f"Invalid CORR in era {era}")
                records.append({"era": era, "rows": len(frame), **scores.to_dict()})
                out = predictions.assign(era=era, **{TARGET: frame[TARGET]}).reset_index()
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(prediction_path, table.schema,
                                              compression="zstd")
                writer.write_table(table)
                if len(records) % 100 == 0:
                    print("Scored", len(records), "validation eras; latest:", era,
                          flush=True)
    finally:
        if writer is not None:
            writer.close()
    if not records:
        raise ValueError("No eligible validation eras")
    scores = pd.DataFrame(records).set_index("era").sort_index()
    assert scores.index.min() > train_max_era + gap_eras
    scores.to_csv(output_dir / "per_era_corr.csv")
    summary = summarize(scores)
    summary.to_csv(output_dir / "summary.csv")
    midpoint = len(scores) // 2
    period_summaries = pd.concat([
        summarize(scores.iloc[:midpoint]).assign(period="earlier_half"),
        summarize(scores.iloc[midpoint:]).assign(period="later_half"),
    ])
    period_summaries.to_csv(output_dir / "period_summary.csv")
    config = {
        "dataset_version": "v5.3", "target": TARGET, "feature_set": "small",
        "feature_count": len(features), "features": features, "seed": seed,
        "rows_per_training_era": rows_per_era, "excluded_validation_gap_eras": gap_eras,
        "training": training_info,
        "validation": {"rows": int(scores["rows"].sum()), "era_count": len(scores),
                       "first_era": int(scores.index.min()),
                       "last_era": int(scores.index.max()), "skipped": skipped},
        "ridge": {"alpha": 100.0, "solver": "cholesky",
                  "preprocessing": "train-only median imputation and StandardScaler"},
        "lightgbm": models["LightGBM"][-1].get_params(),
        "metric": "numerai_tools.scoring.numerai_corr (target_pow15=True)",
        "corr_sharpe_definition": "mean per-era CORR / population std; not annualized",
        "drawdown_definition": "max drop in cumulative CORR; not portfolio returns",
        "validation_usage": "fixed baseline, no tuning or early stopping on validation",
        "fit_seconds": fit_seconds, "total_seconds": time.monotonic() - started,
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "numerai-tools"]},
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n",
                                                encoding="utf-8")
    print(summary.to_string(), flush=True)
    print("Results:", output_dir, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "downloads/numerai/v5.3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results/numerai_baselines")
    parser.add_argument("--rows-per-era", type=int, default=200)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, rows_per_era=args.rows_per_era, trees=args.trees)
