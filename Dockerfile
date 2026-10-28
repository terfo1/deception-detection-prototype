# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:62ed8e07e544e8555c5d93748d5275015136b7f5c52b5e8679b4de339c3779d3
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements/runtime.txt requirements/ci.txt /constraints/
RUN python -m pip install -r /constraints/runtime.txt
COPY pyproject.toml MANIFEST.in README.md LICENSE CONTRIBUTING.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
RUN python -m pip install --no-deps --no-build-isolation . \
    && useradd --uid 1000 --create-home researcher \
    && mkdir -p /app/outputs /app/test-results \
    && chown -R researcher:researcher /app
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Eye-tracking research prototype" \
      org.opencontainers.image.source="https://github.com/terfo1/deception-detection-prototype" \
      org.opencontainers.image.revision=${VCS_REF}

FROM base AS test
USER root
RUN python -m pip install -r /constraints/ci.txt
COPY --chown=researcher:researcher tests/ ./tests/
COPY --chown=researcher:researcher ci/test.sh ./ci/test.sh
USER researcher
ENTRYPOINT ["sh", "ci/test.sh"]

FROM base AS runtime
USER researcher
ENTRYPOINT ["python", "examples/eye_tracking_analysis_demo.py"]
