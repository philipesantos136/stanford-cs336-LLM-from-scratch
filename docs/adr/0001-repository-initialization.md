# ADR 0001: Repository Initialization and Base Structure

* **Status:** Accepted
* **Date:** 2026-07-31

## Context

Need to structure the `Stanford-CS336` project to support building language models from scratch, adhering to software engineering best practices, automated testing, and technical documentation.

## Decision

1. **Directory Structure:**
   - `src/`: Contains core Python modules.
   - `tests/`: Contains automated tests.
   - `docs/`: Holds documentation and ADRs (`docs/adr/`).
   - `tmp/`: Temporary directory (ignored by Git via `.gitignore`).

2. **Version Control Management:**
   - Git as primary VCS with main branch `main`.
   - Inclusion of comprehensive `.gitignore` for Python environment and temporary files.

## Consequences

- Streamlined tracking of architecture and infrastructure decisions.
- Ensured isolation of temporary files and local environment dependencies.
