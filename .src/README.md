# MinorEmbeddingCEC2026

Core code and supporting scripts for our Minor-Embedding systematic benchmark.

## Repository structure

### `.src/graph_io.py`
Infrastructure module for graph-related operations:
- reading graph instances from the filesystem
- parsing graphs in MQLib / GSet-like text format
- sampling and ordering dataset instances
- attaching MQLib-derived graph metrics to `networkx.Graph` objects

### `.src/graph_metrics.sh`
Bash helper script for metric extraction:
- receives the local path to the compiled MQLib executable
- receives a graph instance file
- runs MQLib metric extraction
- writes metrics to a temporary CSV file for downstream Python usage

### `.src/minor_embedding_benchmark.py`
Main benchmark driver:
- defines the command-line interface
- constructs target hardware-topology graphs (Chimera / Pegasus / Zephyr)
- resolves source-graph input mode
- resolves embedding presets for `find_embedding`
- runs the benchmark and reports run configuration

## Notes
- The current workflow assumes a local Unix-like environment.
- MQLib must be cloned and compiled locally for the metric-extraction flow.
- Some paths may still be hard-coded and may require adaptation before reuse.
