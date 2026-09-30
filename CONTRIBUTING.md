# Contributing

Create a branch for each change and connect it to a GitHub issue:

```bash
git switch -c fix/issue-1-segmentation
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest -m 'not deep'
git add src tests docs
git commit -m "fix: describe the change"
git push -u origin fix/issue-1-segmentation
```

Open a pull request against `main` and wait for CI to pass. Install `.[dev,deep]`
to also run LSTM and TCN integration tests. Use small synthetic fixtures for tests;
do not attach participant spreadsheets to issues or commit them to Git.

Reports should include the command, config, Python/library versions, and commit ID.
Scientific results should state the split unit, class counts, and whether the data
are real or synthetic. Synthetic demo accuracy is an execution check only.

To publish a version, first update `project.version` in `pyproject.toml` and commit
the change. Once CI passes, create a matching version tag:

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

The tag starts the same tests and build. If all three prerequisite jobs pass,
the release job publishes the wheel and source archive on GitHub Releases.
Versions must use new tags; do not move a published tag.
