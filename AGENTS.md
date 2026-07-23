# AGENT Handoff

## Project
AFAC OCR pipeline for Tianchi 2026.
Canonical plan: [`docs/Plan.md`](/Users/hasono/Code/ML-Practice/docs/Plan.md).

## Goal
Improve A榜 score from `submission_v033.csv` (`49.5535`) without overfitting to A榜 and while keeping B榜 generalization stable.

## Current Status
- `v033` is the safety baseline and must stay untouched.
- Main implementation work has already shifted to:
  - coverage-first table routing,
  - HTML-preserving table reconstruction,
  - long-document block-level dedupe,
  - profile-based route selection,
  - train-set experiment scaffolding.

## What Has Been Implemented
- `src/afac_pipeline/images.py`
  - `DocumentProfile`
  - `profile_image(...)`
  - `make_adaptive_content_grid_slices(...)`
- `src/afac_pipeline/baseline.py`
  - `table_mode={coverage,anchor,hybrid}`
  - coverage-first table route with fallback to anchor
  - structural scoring for table candidates
  - profile-based long/table routing
  - adaptive acceptance for coverage tiles
- `src/afac_pipeline/tables.py`
  - HTML table reconstruction
  - `table_to_html(...)`
- `src/afac_pipeline/pipeline.py`
  - block-level merge / dedupe for long-doc slices
  - table preamble/postamble preservation
- `src/afac_pipeline/experiment.py`
  - train experiment runner
  - artifacts: `predictions.csv`, `metrics.csv`, `manifest.json`, `errors.csv`
  - deterministic train splits: dev / validation / rest / all
- `src/afac_pipeline/cli.py`
  - `baseline-submit` extended with route knobs
  - `experiment-train` extended with `--kind`, `--offset`, `--limit`

## Verification So Far
- `uv run python -m compileall src` passes.
- `uv run python -m unittest discover -s tests -v` passes.
- `uv run afac experiment-train --help` passes.
- `uv run afac inspect-data` reports:
  - train images: 200
  - train markdowns: 200
  - A images: 100
  - mock rows: 100

## Dataset Split Facts
- `build_train_splits(data/raw)` currently returns:
  - dev: 40
  - validation: 40
  - rest: 120
  - all: 200
- dev / validation each contain 20 long + 20 table samples.

## Probe Results
1. Coverage probe on one dev table with `table-target-tile-width=2800`
   - produced `1x2` tiles
   - both tiles were truncated by FinixDoc (`unclosed code fence`)
   - fallback anchor result was only `7724/229533` chars
   - local proxy score: `20.4361`
2. Coverage probe with `table-target-tile-width=700`
   - produced `1x7` tiles
   - hit RPM rate limit (`429`)
   - run was interrupted after observing the limit

## Important Caveats
- Multi-tile coverage needs throttling; no-sleep runs hit RPM quickly.
- Large table tiles can still trigger truncation, so table tile size probably needs another pass.
- The current code already avoids overfitting to UUIDs, filenames, or A榜-only directory strings.
- Existing unrelated worktree changes are present; do not revert them.

## Existing Outputs / Cache
- `outputs/submission_v033.csv` still exists.
- Existing baseline caches are present under:
  - `outputs/cache/a.baseline/baseline_v1_...`
  - `outputs/cache/a.baseline/baseline_long_v1_...`
- New probe outputs were written to:
  - `outputs/experiments/v034-probe-table1/`
  - `outputs/cache/experiments/v034-probe-table1/`

## Recommended Next Steps
1. Tune table coverage tile size downward from `2800x4200`.
2. Add/adjust sleep or retry policy for multi-tile coverage runs.
3. Re-run a small dev-table probe, then a small dev-long probe.
4. If scores improve and no truncation spikes appear, expand to full dev and then validation.

## Safe Working Rules
- Keep `v033` as the rollback point.
- Do not add UUID/file-specific rules.
- Do not optimize only for the visible A榜 subset.
- Prefer `apply_patch` for edits.
- Ignore unrelated dirty files unless they block the task.
