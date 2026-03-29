# Patient Readmission Prediction System

> **Live Demo:** [Streamlit UI](#) *(link added after deployment)*
> ![CI](https://github.com/hari-etta/hari-etta-portfolio/actions/workflows/p1_ci.yml/badge.svg)

---

## Problem Statement

<!-- What business problem, who loses money, how much, why prediction helps -->

## Live Demo

<!-- Screenshot of Streamlit UI embedded, curl command for the API -->

## Architecture

<!-- Data flow: GCS → sklearn Pipeline → MLflow → FastAPI → Cloud Run → Streamlit → monitor.py → GCS logs -->

## Tech Stack

<!-- Tool | Purpose | Why chosen over alternatives | Free tier confirmation -->

## Dataset

<!-- Source, size, key features, class imbalance ratio, preprocessing decisions and why -->

## Approach

<!-- Why these 3 models, what didn't work, documented tradeoffs -->

## Results

<!-- Business framing first, then full metrics table (AUC, precision, recall, F1, cost matrix) -->

## SHAP Explainability

<!-- Global importance bar chart, individual patient waterfall screenshot -->

## Fairness Audit

<!-- Demographic parity analysis findings, what mitigation would look like -->

## Model Monitoring

<!-- How predictions are logged, drift detection, how to trigger retraining -->

## API Documentation

<!-- /predict and /health endpoint specs, request/response schemas, working curl commands -->

## Latency Benchmarks

<!-- p50 and p95 in a table, how measured, acceptable range for production -->

## Why No Databricks

Dataset fits in memory — Spark would be over-engineering. I use Databricks in Project 2 where the data volume justifies distributed compute.

## Limitations and Future Work

<!-- What this model cannot do, when it should not be trusted, 3 things that would improve it -->

## Insurance Applicability

<!-- One paragraph repositioning methodology for actuarial risk scoring -->

## How to Run Locally

<!-- Step by step using .env.example, one Makefile command per action -->

## Commit History

This project was built incrementally following a documented commit strategy. Each commit represents one meaningful unit of work, making the build process auditable and reproducible.