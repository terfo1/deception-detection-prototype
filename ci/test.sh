#!/bin/sh
# Same gates in Jenkins, Actions and a local test container. Stop on any failure.
set -eu
mkdir -p test-results
python -m pip freeze > test-results/environment.txt
python -m pip check
python -m ruff check .
python -m pytest -m 'not deep' -p no:cacheprovider \
    --cov=src --cov-report=xml:test-results/coverage.xml \
    --junitxml=test-results/junit.xml
python -m build --no-isolation --outdir test-results/dist
