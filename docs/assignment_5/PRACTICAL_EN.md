# Practical Report: CI/CD Pipeline for the Eye-Tracking Research Prototype

Assignment 5 - practical part | 28 October 2026

## Project and repository

The pipeline serves the existing Python research prototype for offline eye-tracking analysis (NumPy, pandas, scikit-learn, Matplotlib). Repository: https://github.com/terfo1/deception-detection-prototype (branch codex/assignment-5, files Dockerfile, Jenkinsfile, ci/, k8s/). The demonstration input is explicitly synthetic; no participant recordings enter any image or build.

## Steps

1. Dockerfile. A multi-stage build on a digest-pinned python:3.12-slim image with pinned requirements: target "test" adds pytest, ruff, coverage and build; target "runtime" runs the synthetic analysis demo as a non-root user. An allow-list .dockerignore keeps data, archives and credentials out of the build context.

2. Jenkins server. ci/jenkins/compose.yaml starts a local lab: Jenkins LTS (JDK 21, plugins preinstalled, admin account and pipeline job created by init.groovy), a TLS Docker-in-Docker daemon, a private registry and a single-node k3s Kubernetes cluster. Jenkins is bound to 127.0.0.1:8085; the admin password is generated into an ignored .env file. ci/jenkins/manage.py starts the lab, triggers builds and collects their artifacts.

3. CI pipeline (Jenkinsfile). Jenkins polls Git every two minutes, so each new commit triggers a build: checkout, build test image, unit tests and quality gates (pip check, ruff, pytest with JUnit and coverage, package build), build runtime image, container smoke test, provenance record (revision, image and Docker versions).

4. CD. The runtime image is tagged with the Git revision and pushed to the lab registry; Jenkins then applies k8s/namespace.yaml and k8s/analysis-job.yaml to k3s, waits for the Job to complete and archives its log. An optional stage pushes the image to Docker Hub when DOCKERHUB_REPOSITORY and a Jenkins credential "dockerhub" are configured. GitHub Actions runs the same container checks on every push as a hosted second CI.

## Results

__RESULTS__

## Difficulties

__DIFFICULTIES__

## Conclusions

The project now has a repeatable chain from a Git revision to tested images, archived evidence and a Kubernetes test deployment, reproducible on any machine with Docker using one command. The pipeline verifies software and deployability only; the synthetic demonstration does not measure deception-detection accuracy, which still requires real labelled recordings and participant-grouped evaluation.

<!-- page -->

## Screenshots

__SCREENSHOTS__
