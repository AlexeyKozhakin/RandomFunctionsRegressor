"""Compare a fixed LHS RFR configuration with the saved Numerai baselines."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

from numerai_baselines import ROOT, iter_eras, summarize

sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numerai_tools.scoring import numerai_corr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from RandomFunctionsRegressor import RandomFunctionsRegressor

NAMES = ["Ridge", "LightGBM", "RFR_LHS"]


def run(n_support=128, train_only=False):
    if isinstance(n_support, bool) or not isinstance(n_support, int) or n_support < 1:
        raise ValueError("n_support must be a positive integer")
    data_dir = ROOT / "downloads/numerai/v5.3"
    baseline_dir = ROOT / "experiments/results/numerai_baselines"
    suffix = "" if n_support == 128 else f"_{n_support}"
    model_name = "RFR_LHS" if n_support == 128 else f"RFR_LHS_{n_support}"
    output_dir = ROOT / f"experiments/results/numerai_rfr{suffix}"
    artifact_dir = ROOT / f"downloads/numerai/rfr_artifacts{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((baseline_dir / "run_config.json").read_text())
    features = config["features"]
    target = config["target"]
    seed = config["seed"]
    columns = ["id", "era", "data_type", target] + features
    rng = np.random.default_rng(seed)
    selected, counts = [], []
    started = time.monotonic()
    cached_sample = ROOT / "downloads/numerai/rfr_artifacts/training_sample.parquet"
    prior_config_path = ROOT / "experiments/results/numerai_rfr/run_config.json"
    if cached_sample.exists() and prior_config_path.exists():
        print("Loading the verified training sample from the 128-support run", flush=True)
        prior_config = json.loads(prior_config_path.read_text())
        # Runtime timings and dependency versions can differ on a new machine.
        # The sample protocol and data identity must match; the Ridge refit
        # below also checks it against the current saved baseline model.
        sample_keys = ("dataset_version", "target", "features", "seed",
                       "rows_per_training_era", "excluded_validation_gap_eras", "training")
        for key in sample_keys:
            assert prior_config["baseline_config"][key] == config[key], key
        training = pd.read_parquet(cached_sample)
        sample_hash = hashlib.sha256("\n".join(training["id"]).encode()).hexdigest()
        assert sample_hash == prior_config["training_ids_sha256"]
    else:
        print("Reconstructing the exact baseline training sample", flush=True)
        for era, frame in iter_eras(data_dir / "train.parquet", columns):
            assert frame["data_type"].eq("train").all()
            eligible = frame.dropna(subset=[target])
            if eligible.empty:
                continue
            count = min(config["rows_per_training_era"], len(eligible))
            indices = rng.choice(len(eligible), size=count, replace=False)
            selected.append(eligible.iloc[indices].copy())
            counts.append({"era": era, "eligible_rows": len(eligible), "sampled_rows": count})
            if len(counts) % 100 == 0:
                print("Sampled", len(counts), "training eras", flush=True)
        pd.testing.assert_frame_equal(pd.DataFrame(counts),
                                      pd.read_csv(baseline_dir / "training_eras.csv"))
        training = pd.concat(selected, ignore_index=True)
        del frame, eligible
    del selected
    gc.collect()
    assert len(training) == config["training"]["rows"]
    assert not training["id"].duplicated().any()
    training_hash = hashlib.sha256("\n".join(training["id"]).encode()).hexdigest()
    training.to_parquet(artifact_dir / "training_sample.parquet", index=False)
    X = training[features].astype(np.float32)
    y = training[target].astype(np.float32)
    loading_seconds = time.monotonic() - started
    parameters = dict(n_support=n_support, random_state=42, method_reg="ridge",
                      alpha=0.1, batch_size=1024, fit_intercept=True,
                      C=1, n=1e5, t=1e5, sig=0.0)
    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomFunctionsRegressor(**parameters))
    with threadpool_limits(limits=2):
        # Refit the inexpensive Ridge to check that reconstructing train
        # reproduces the original fitted coefficients, not merely row counts.
        saved_ridge = joblib.load(ROOT / "downloads/numerai/baseline_artifacts/ridge.joblib")["model"]
        check_ridge = clone(saved_ridge).fit(X, y)
        np.testing.assert_allclose(check_ridge[-1].coef_, saved_ridge[-1].coef_,
                                   rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(check_ridge[-1].intercept_, saved_ridge[-1].intercept_,
                                   rtol=1e-6, atol=1e-8)
        del check_ridge, saved_ridge
        tick = time.monotonic()
        print("Fitting RFR: rows", len(X), "features", len(features), "supports", n_support,
              flush=True)
        model.fit(X, y)
        fit_seconds = time.monotonic() - tick
        regressor = model[-1]
        assert regressor.q.shape == (n_support,)
        train_prediction = model.predict(X)
        train_corr = numerai_corr(
            pd.DataFrame({"RFR_LHS": train_prediction}, index=training["id"]),
            pd.Series(y.to_numpy(), index=training["id"], name=target),
            max_filtered_index_ratio=0.0,
        ).iloc[0]
        # This pooled in-sample score is diagnostic only, not comparable with
        # the out-of-sample per-era average below.
        print("Fitted RFR in", round(fit_seconds, 2), "seconds", flush=True)
        joblib.dump({"model": model, "features": features, "target": target},
                    artifact_dir / "rfr_lhs.joblib")
    del training, X, y, train_prediction
    gc.collect()

    if train_only:
        report = {
            "status": "trained_only", "n_support": n_support,
            "fit_seconds": fit_seconds, "training_ids_sha256": training_hash,
            "rfr_parameters": parameters, "fitted_model_saved": True,
            "validation_performed_in_this_run": False,
            "note": "Any existing validation tables belong to earlier runs.",
        }
        (output_dir / "run_status.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Training only; saved:", artifact_dir / "rfr_lhs.joblib", flush=True)
        return None

    baseline = pd.read_csv(baseline_dir / "per_era_corr.csv").set_index("era")
    if n_support != 128 and prior_config_path.exists():
        prior_scores = pd.read_csv(prior_config_path.parent / "per_era_corr.csv").set_index("era")
        pd.testing.assert_frame_equal(prior_scores[baseline.columns], baseline)
        baseline = baseline.join(prior_scores[["RFR_LHS"]].rename(columns={"RFR_LHS": "RFR_LHS_128"}),
                                 validate="one_to_one")
    names = [name for name in baseline.columns if name != "rows"] + [model_name]
    records = []
    writer = None
    prediction_path = artifact_dir / "validation_predictions.parquet"
    timings = {f"validation_{stage}_seconds": 0.0 for stage in
               ["read", "prepare", "predict", "score", "write"]}
    validation_started = time.monotonic()
    try:
        with threadpool_limits(limits=2):
            eras = iter(iter_eras(data_dir / "validation.parquet", columns))
            while True:
                tick = time.monotonic()
                try:
                    era, frame = next(eras)
                except StopIteration:
                    timings["validation_read_seconds"] += time.monotonic() - tick
                    break
                timings["validation_read_seconds"] += time.monotonic() - tick
                if era <= config["training"]["last_era"] + config["excluded_validation_gap_eras"]:
                    continue
                tick = time.monotonic()
                frame = frame.loc[frame["data_type"].eq("validation")].dropna(subset=[target])
                if len(frame) < 2:
                    timings["validation_prepare_seconds"] += time.monotonic() - tick
                    continue
                frame = frame.set_index("id")
                assert frame.index.is_unique
                assert era in baseline.index and len(frame) == baseline.loc[era, "rows"]
                X_era = frame[features].astype(np.float32)
                timings["validation_prepare_seconds"] += time.monotonic() - tick
                tick = time.monotonic()
                predicted = model.predict(X_era)
                timings["validation_predict_seconds"] += time.monotonic() - tick
                tick = time.monotonic()
                predictions = pd.DataFrame({model_name: predicted},
                                           index=frame.index)
                assert np.isfinite(predictions.to_numpy()).all()
                assert predictions[model_name].nunique() > 1
                score = numerai_corr(predictions, frame[target],
                                     max_filtered_index_ratio=0.0).iloc[0]
                assert np.isfinite(score)
                records.append({"era": era, "rows": len(frame), model_name: float(score)})
                timings["validation_score_seconds"] += time.monotonic() - tick
                tick = time.monotonic()
                out = predictions.assign(era=era, **{target: frame[target]}).reset_index()
                table = pa.Table.from_pandas(out, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(prediction_path, table.schema, compression="zstd")
                writer.write_table(table)
                timings["validation_write_seconds"] += time.monotonic() - tick
                if len(records) % 25 == 0:
                    print("Scored", len(records), "/", len(baseline), "eras; latest", era,
                          "predict seconds", round(timings["validation_predict_seconds"], 1),
                          flush=True)
    finally:
        if writer is not None:
            writer.close()
    rfr_scores = pd.DataFrame(records).set_index("era").sort_index()
    pd.testing.assert_index_equal(rfr_scores.index, baseline.index)
    np.testing.assert_array_equal(rfr_scores["rows"], baseline["rows"])
    scores = baseline.join(rfr_scores[[model_name]], validate="one_to_one")
    scores.to_csv(output_dir / "per_era_corr.csv")
    summary = summarize(scores, names)
    summary.to_csv(output_dir / "summary.csv")
    midpoint = len(scores) // 2
    pd.concat([
        summarize(scores.iloc[:midpoint], names).assign(period="earlier_half"),
        summarize(scores.iloc[midpoint:], names).assign(period="later_half"),
    ]).to_csv(output_dir / "period_summary.csv")
    report = {
        "baseline_config": config, "rfr_parameters": parameters,
        "input_preprocessing": "train-only median imputation; original feature scale",
        "kernel": "original Kf, unchanged", "training_ids_sha256": training_hash,
        "training_match_check": "reconstructed sample reproduces saved Ridge coefficients",
        "fit_seconds": fit_seconds, "total_seconds": time.monotonic() - started,
        "data_loading_seconds": loading_seconds,
        "validation_seconds": time.monotonic() - validation_started,
        "stage_timings": timings,
        "support_shape": list(regressor.support_points_.shape),
        "coefficient_count": len(regressor.q),
        "pooled_training_corr_diagnostic_only": float(train_corr),
        "validation_usage": "fixed configuration; support count compared on previously viewed validation",
    }
    (output_dir / "run_config.json").write_text(json.dumps(report, indent=2) + "\n",
                                                encoding="utf-8")
    (output_dir / "run_status.json").write_text(json.dumps({
        "status": "complete", "n_support": n_support,
        "completed_validation_eras_reported": len(scores),
        "complete_validation_summary_available": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(), flush=True)
    print("Results:", output_dir, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-support", type=int, default=128)
    parser.add_argument("--train-only", action="store_true",
                        help="Save fitted model and training sample, then skip validation")
    args = parser.parse_args()
    run(n_support=args.n_support, train_only=args.train_only)
