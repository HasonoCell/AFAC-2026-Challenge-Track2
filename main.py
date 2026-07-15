"""Backward-compatible entry point. Prefer `uv run afac`."""

from afac_pipeline.cli import main


if __name__ == "__main__":
    main()
