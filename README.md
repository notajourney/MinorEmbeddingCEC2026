# MinorEmbeddingCEC2026

Code and reference material for our study of **minor embedding** as a combinatorial bottleneck in quantum annealing workflows.

## Focus

This repository accompanies a research effort centered on the empirical analysis of minor embedding performance, with emphasis on:

- benchmarking embedding behavior on structured, non-random graph instances
- studying problem representations related to **Max-Cut** and **QUBO**
- examining how graph structure and graph-derived metrics relate to embedding difficulty and outcomes
- organizing the supporting code, plots, and reference material used in the project

## Main idea

Rather than treating embedding as a hidden preprocessing step, this project studies it as an object of analysis in its own right.

The broader goal is to better understand which structural properties of input graphs are associated with harder or easier embeddings, and how these properties connect to measurable embedding outcomes.

## Repository layout

- `.src/` — core source code used in the project
- `plots/` — figures, visual outputs, and supporting plots used for analysis and write-up

## Current source files

- `graph_io.py` — graph loading, parsing, dataset access, and graph-level metadata/metrics support
- `graph_metrics.sh` — helper script for extracting graph metrics through MQLib-related tooling
- `minor_embedding_benchmark.py` — main benchmark driver for running embedding experiments across target architectures and source graph collections

## Status

This repository is currently being used as a compact public home for code and supporting materials connected to the paper and its references.
