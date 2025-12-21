# Repository Guidelines

## Project Structure & Module Organization
- `trellis2/` contains the core Python package (pipelines, models, renderers, utils, trainers, datasets).
- `o-voxel/` hosts the O-Voxel implementation and related utilities.
- `assets/` stores demo inputs such as HDRI maps and example images.
- Entry points: `app.py` (web demo) and `example.py` (minimal inference script).
- There is no dedicated `tests/` directory at the moment.

## Build, Test, and Development Commands
- `. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm`  
  Creates a conda environment and installs required dependencies. Run `. ./setup.sh --help` for flags.
- `python app.py`  
  Launches the local web demo for image-to-3D generation.
- `python example.py`  
  Runs the minimal inference example and writes `sample.mp4` and `sample.glb`.

## Coding Style & Naming Conventions
- Python style follows existing code: 4-space indentation, PEP 8-ish layout, and docstrings where helpful.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes (e.g., `Trellis2ImageTo3DPipeline`), lowercase module names.
- No formatter or linter is enforced in-repo; match surrounding style and keep imports grouped.

## Testing Guidelines
- There is no automated test suite currently. Validate changes by running `python example.py` or `python app.py`.
- If adding tests, place them under `tests/` and use `test_*.py` names to enable pytest discovery.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative summaries (e.g., `Fix website url`, `Add demo link`).
- PRs should include a concise description, linked issues if applicable, and sample outputs or screenshots for UI/visual changes.
- Note GPU, CUDA version, and key environment variables (`CUDA_HOME`, `ATTN_BACKEND`) when they affect results.

## Security & Configuration Tips
- Follow `SECURITY.md` for reporting vulnerabilities; do not open public issues for security reports.
- The project assumes Linux + NVIDIA GPU; set `OPENCV_IO_ENABLE_OPENEXR=1` when using EXR inputs.
