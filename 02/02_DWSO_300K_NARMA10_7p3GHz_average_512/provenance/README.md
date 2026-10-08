# Provenance and quality-assurance records

This directory preserves the evidence used to assemble and validate the complete R = 512 dataset.

- `summary_qa_R512.json` records the full source aggregation audit, including the per-seed source state and metadata checksums.
- `drive_thermal_seed_coverage_R512.csv` lists all 512 unique, contiguous drive-stage thermal seeds from 32345 through 32856 and their QA evidence.
- `analysis_summary_locked_R512.json` records the locked fitting protocol, alpha sweep, state checksum, prediction checksum, and test metrics.
- `artifact_sha256_source_R512.csv` is the source-directory hash manifest generated with the original R = 512 characterization.
- The Python files are copies of the original protocol, fitting, and validation code used to create and audit the source result.

Absolute paths in these records refer to the original Linux or Windows analysis locations and are retained verbatim for provenance. They are not required by the package-level `analyze_and_plot.py` script.
