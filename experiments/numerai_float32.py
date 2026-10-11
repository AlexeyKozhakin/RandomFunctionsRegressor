"""Paired CPU precision/timing experiment and portable inputs for Colab."""
import argparse
import hashlib
import json
import sys
import time

from numerai_baselines import ROOT, iter_eras, summarize

sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from rfr_precision import kernel_matrix, predict


def run(era_limit=64):
    output = ROOT / "experiments/results/numerai_float32"
    artifacts = ROOT / "downloads/numerai/float32_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    saved = joblib.load(ROOT / "downloads/numerai/rfr_artifacts_256/rfr_lhs.joblib")
    model = saved["model"]
    reg = model[-1]
    features, target = saved["features"], saved["target"]
    cfg = json.loads((ROOT / "experiments/results/numerai_baselines/run_config.json").read_text())
    train = pd.read_parquet(ROOT / "downloads/numerai/rfr_artifacts/training_sample.parquet")
    X_train = model[0].transform(train[features].astype(np.float32)).astype(np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    fits, refits = {}, {}
    with threadpool_limits(limits=2):
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            tick = time.perf_counter()
            x = X_train.astype(dtype)
            points = reg.support_points_.astype(dtype)
            design = np.empty((len(x), reg.n_support_), dtype=dtype)
            for start in range(0, len(x), reg.batch_size):
                stop = min(start + reg.batch_size, len(x))
                design[start:stop] = kernel_matrix(x[start:stop], points, reg.C, reg.n, reg.t, True)
            ridge = Ridge(alpha=reg.alpha + reg.sig ** 2,
                          fit_intercept=reg.fit_intercept, solver="svd")
            ridge.fit(design, y_train.astype(dtype))
            fits[name] = time.perf_counter() - tick
            assert ridge.coef_.dtype == np.dtype(dtype)
            # predict() expects the original unshifted intercept.
            bias = float(ridge.intercept_) - reg.C * reg.n * ridge.coef_.astype(np.float64).sum()
            refits[name] = (ridge.coef_.copy(), bias)
            del design, x
            print("Centered", name, "fit:", round(fits[name], 3), "seconds", flush=True)

        methods = ["original_float64", "matrix_float64", "raw_float32",
                   "centered_float32", "refit_float64", "refit_float32"]
        times = {name: [] for name in methods}
        error_sum = {name: 0.0 for name in methods[1:]}
        error_max = {name: 0.0 for name in methods[1:]}
        metrics, timing_records, arrays, ids = [], [], [], []
        for era, frame in iter_eras(ROOT / "downloads/numerai/v5.3/validation.parquet",
                                    ["id", "era", "data_type", target] + features):
            if era <= cfg["training"]["last_era"] + cfg["excluded_validation_gap_eras"]:
                continue
            frame = frame.loc[frame.data_type.eq("validation")].dropna(subset=[target])
            if len(frame) < 2:
                continue
            X = model[0].transform(frame[features].astype(np.float32)).astype(np.float32)
            functions = {
                "original_float64": lambda: reg.predict(X),
                "matrix_float64": lambda: predict(X, reg.support_points_, reg.q, reg.intercept_,
                                                    reg.C, reg.n, reg.t, np.float64),
                "raw_float32": lambda: predict(X, reg.support_points_, reg.q, reg.intercept_,
                                                reg.C, reg.n, reg.t, centered=False),
                "centered_float32": lambda: predict(X, reg.support_points_, reg.q, reg.intercept_,
                                                     reg.C, reg.n, reg.t),
                "refit_float64": lambda: predict(X, reg.support_points_, *refits["float64"],
                                                 reg.C, reg.n, reg.t, np.float64),
                "refit_float32": lambda: predict(X, reg.support_points_, *refits["float32"],
                                                 reg.C, reg.n, reg.t),
            }
            # Warm every implementation, then rotate order to reduce order bias.
            for function in functions.values():
                function()
            predictions = {}
            for repeat in range(3):
                order = methods[repeat:] + methods[:repeat]
                for name in order:
                    tick = time.perf_counter()
                    predictions[name] = functions[name]()
                    elapsed = time.perf_counter() - tick
                    timing_records.append({"era": era, "repeat": repeat, "model": name,
                                           "seconds": elapsed, "rows": len(frame)})
            for name in methods:
                samples = [r["seconds"] for r in timing_records[-18:] if r["model"] == name]
                times[name].append(float(np.median(samples)))
            reference = predictions["original_float64"]
            np.testing.assert_allclose(predictions["matrix_float64"], reference,
                                       rtol=1e-7, atol=1e-8)
            for name in methods[1:]:
                delta = predictions[name].astype(np.float64) - reference
                error_sum[name] += float(delta @ delta)
                error_max[name] = max(error_max[name], float(np.abs(delta).max()))
            scores = numerai_corr(pd.DataFrame(predictions, index=frame.id),
                                  pd.Series(frame[target].to_numpy(), index=frame.id, name=target),
                                  max_filtered_index_ratio=0.0)
            metrics.append({"era": era, "rows": len(frame), **scores.to_dict()})
            arrays.append((X, frame[target].to_numpy(dtype=np.float32),
                           np.full(len(frame), era, dtype=np.int32), reference))
            ids.extend(frame.id.tolist())
            if len(metrics) % 8 == 0:
                print("Scored", len(metrics), "/", era_limit, "eras", flush=True)
            if len(metrics) >= era_limit:
                break

    scores = pd.DataFrame(metrics).set_index("era")
    summary = summarize(scores, methods)
    rows = int(scores.rows.sum())
    for name in methods:
        summary.loc[name, "predict_seconds"] = sum(times[name])
        summary.loc[name, "speedup_vs_original"] = sum(times[methods[0]]) / sum(times[name])
        summary.loc[name, "rmse_vs_original"] = np.sqrt(error_sum.get(name, 0) / rows)
        summary.loc[name, "max_abs_error"] = error_max.get(name, 0)
    summary.to_csv(output / "summary.csv")
    scores.to_csv(output / "per_era_corr.csv")
    pd.DataFrame(timing_records).to_csv(output / "timings.csv", index=False)
    report = dict(device="CPU", threads=2, supports=reg.n_support_, features=len(features),
                  rows=rows, eras=len(scores), first_era=int(scores.index.min()),
                  last_era=int(scores.index.max()), fit_seconds=fits,
                  timing="3 repeats after warmup; sum of per-era medians; excludes I/O and scoring",
                  training="same saved sample and same 256 support coordinates; centered Ridge SVD",
                  validation="first eligible eras only; preliminary precision and speed check",
                  ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest())
    (output / "run_config.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(artifacts / "colab_inputs.npz", X_train=X_train, y_train=y_train,
                        X_validation=np.concatenate([a[0] for a in arrays]),
                        y_validation=np.concatenate([a[1] for a in arrays]),
                        eras=np.concatenate([a[2] for a in arrays]),
                        reference=np.concatenate([a[3] for a in arrays]),
                        supports=reg.support_points_, coefficients=reg.q,
                        intercept=reg.intercept_, C=reg.C, n=reg.n, t=reg.t, alpha=reg.alpha)
    print(summary.to_string(), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", type=int, default=64)
    args = parser.parse_args()
    if args.eras < 1:
        parser.error("--eras must be positive")
    run(args.eras)
