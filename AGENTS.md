# Instructions for agents working on this branch

Read `experiments/HANDOFF.md` before continuing RFR or Numerai work. It records
the user's intended model, completed experiments, reproduction commands, and
the untested Colab workflow. Most experiment documentation is in Russian.

- Preserve the original `Kf` unless the user explicitly agrees to change it.
- `n_support=None` keeps the original training-point mode; LHS is opt-in.
- The NumPy float32 backend and PyTorch GPU implementation are experiments,
  not the public estimator. Validate predictions and per-era CORR before
  promoting either backend into the library.
- Keep Numerai inputs, models, prediction files, `.venv`, credentials, and
  browser profiles out of Git. `downloads/` and `.venv/` are ignored.
- Checked-in tables are historical results. Do not describe the interrupted
  256-point validation or the GPU notebook as completed experiments.
- Do not download the full dataset or retrain just to review the branch.
  Download and train when required by the user's next task.
- Use train-only preprocessing and support-point bounds. Evaluation uses
  complete eras and a 13-era gap. Report changes in evaluation coverage.
- Run `python -m unittest discover -s tests -v` for estimator/backend or
  downloader changes; no live network or Numerai data are needed by the tests.
