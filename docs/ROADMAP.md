# InfraGuard AI — Project Roadmap

## Project Goal

Build an end-to-end AI Engineering (kỹ thuật hệ thống AI) project for UAV-based building surface defect detection.

The project should demonstrate:

* data engineering
* Computer Vision (thị giác máy tính)
* model training
* experimental design
* model evaluation
* error analysis
* cross-dataset generalization
* inference optimization
* API serving
* containerization
* automated testing
* CI/CD (tích hợp và triển khai liên tục)
* technical documentation

---

# Current Release Target

**Target:** `v1.0.0`

**Primary dataset:** MBDD2025

**External validation dataset:** CUBIT-Det

**Initial baseline:** YOLO11n

---

# M0 — Engineering Foundation

## Goal

Establish a clean, reproducible repository and development workflow.

## Tasks

* [ ] Create GitHub repository
* [ ] Add project README
* [ ] Add LICENSE
* [ ] Configure `.gitignore`
* [ ] Configure Python 3.11 project
* [ ] Add `pyproject.toml`
* [ ] Create package under `src/infraguard`
* [ ] Add test structure
* [ ] Configure pytest
* [ ] Configure Ruff
* [ ] Add `AGENTS.md`
* [ ] Add PR template
* [ ] Add Issue templates
* [ ] Add project documentation structure
* [ ] Establish branch naming convention
* [ ] Establish Issue → branch → PR workflow
* [ ] Configure basic CI checks

## Definition of Done

* repository installs successfully
* `pytest` runs successfully
* `ruff check .` passes
* AI agent rules are documented
* feature branches are used instead of direct pushes to `main`

## Release

`v0.0.1-foundation`

---

# M1 — MBDD2025 Data Foundation

## Goal

Understand and validate MBDD2025 before any serious model training.

## Dataset Registration

* [ ] Download MBDD2025
* [ ] Record dataset source
* [ ] Record dataset version
* [ ] Record dataset license
* [ ] Record paper citation
* [ ] Confirm directory structure
* [ ] Confirm annotation format
* [ ] Confirm class IDs and class names

## Dataset Loader

* [ ] Implement MBDD2025 configuration
* [ ] Implement image discovery
* [ ] Implement YOLO label parsing
* [ ] Match images with annotations
* [ ] Define internal image record schema
* [ ] Define bounding-box schema
* [ ] Handle empty labels
* [ ] Handle missing labels
* [ ] Add loader unit tests

## Dataset Validation

* [x] Detect missing images
* [x] Detect missing annotations
* [x] Detect corrupted images
* [x] Detect unreadable files
* [x] Detect malformed annotation rows
* [x] Detect invalid class IDs
* [x] Detect invalid normalized coordinates
* [x] Detect zero-area bounding boxes
* [x] Detect bounding boxes outside image boundaries
* [x] Detect duplicate bounding boxes
* [x] Generate machine-readable validation report

## Dataset Statistics

* [x] Count total images
* [x] Count total instances
* [x] Calculate image count per class
* [x] Calculate instance count per class
* [x] Calculate objects per image
* [x] Analyze bounding-box width
* [x] Analyze bounding-box height
* [x] Analyze normalized bounding-box area
* [x] Analyze bounding-box aspect ratio
* [x] Analyze object-center distributions
* [x] Analyze image resolutions
* [x] Analyze brightness distribution
* [x] Analyze image contrast

## Data Quality Audit

* [ ] Detect exact image duplicates
* [ ] Investigate near-duplicate images
* [ ] Investigate augmented `bulge` samples
* [ ] Check potential duplicate leakage across train/validation/test splits
* [ ] Document center bias
* [ ] Document class imbalance
* [ ] Document known dataset limitations

## Documentation

* [ ] Complete `docs/DATASETS.md`
* [ ] Generate Week 1 EDA figures
* [ ] Write Week 1 data-quality report
* [ ] Update technical decisions

## Definition of Done

* MBDD2025 can be loaded without notebook-only logic
* full dataset validation can run through CLI
* dataset statistics can be reproduced through CLI
* major data-quality risks are documented
* duplicate/leakage audit is complete
* all tests pass
* main branch is clean

## Release

`v0.1.0-data-foundation`

---

# M2 — Baseline Detection Model

## Goal

Establish a reproducible object-detection baseline.

## Training Infrastructure

* [ ] Add model configuration structure
* [ ] Add experiment configuration structure
* [ ] Configure deterministic random seed
* [ ] Add training CLI
* [ ] Add experiment output structure
* [ ] Record environment metadata

## Baseline

Model:

`YOLO11n`

Initial configuration:

* image size: 640
* fixed dataset split
* fixed random seed
* fixed augmentation configuration

## Experiments

* [ ] Train initial baseline
* [ ] Save training configuration
* [ ] Evaluate validation split
* [ ] Evaluate test split
* [ ] Record mAP@0.5
* [ ] Record mAP@0.5:0.95
* [ ] Record Precision
* [ ] Record Recall
* [ ] Record F1
* [ ] Record AP per class
* [ ] Generate confusion matrix
* [ ] Record model size
* [ ] Record inference latency

## Definition of Done

* baseline can be reproduced from repository configuration
* metrics are documented
* predictions are available for error analysis
* no test-set leakage is introduced

## Release

`v0.2.0-baseline`

---

# M3 — Error Analysis and Targeted Improvements

## Goal

Understand model failures before attempting optimization.

## Error Analysis

* [ ] Collect false positives
* [ ] Collect false negatives
* [ ] Analyze class confusion
* [ ] Analyze localization failures
* [ ] Analyze small-object failures
* [ ] Analyze edge-position failures
* [ ] Analyze low-light failures
* [ ] Analyze low-contrast failures
* [ ] Analyze thin-crack failures

## Improvement Experiments

Select at most a small number of justified changes.

Potential experiments:

* [ ] data augmentation
* [ ] class balancing
* [ ] image-size adjustment
* [ ] training configuration adjustment

Every experiment requires:

* hypothesis
* baseline
* treatment
* comparable metrics
* conclusion

## Definition of Done

* major model failure modes are known
* at least one targeted improvement is evaluated
* improvements are supported by controlled experiments
* regressions are documented

## Release

`v0.3.0-error-analysis`

---

# M4 — High-Resolution Tiled Inference

## Goal

Improve small-defect detection without blindly resizing high-resolution images.

## Implementation

* [ ] Implement tile generation
* [ ] Implement configurable tile size
* [ ] Implement configurable overlap
* [ ] Handle border tiles
* [ ] Run detector on individual tiles
* [ ] Convert local coordinates to global coordinates
* [ ] Clip predictions to image bounds
* [ ] Merge overlapping predictions
* [ ] Implement or integrate NMS
* [ ] Handle empty tile predictions
* [ ] Add coordinate-conversion tests

## Experiments

Compare:

* [ ] resize to 640
* [ ] resize to 1280
* [ ] tiled inference
* [ ] tiled inference with overlap

Metrics:

* mAP
* Recall
* small-object AP
* crack AP
* inference latency
* memory usage where practical

## Research Question

Does tiled high-resolution inference improve small-defect detection enough to justify its computational cost?

## Definition of Done

* tiled inference works on full-resolution images
* coordinate correctness is unit-tested
* accuracy-latency trade-off is benchmarked
* conclusions are documented

## Release

`v0.4.0-high-resolution`

---

# M5 — Cross-Dataset Generalization

## Goal

Measure how well the detector generalizes to different UAV infrastructure datasets.

## CUBIT-Det Integration

* [ ] Document CUBIT-Det source
* [ ] Document CUBIT-Det license
* [ ] Implement CUBIT dataset loader
* [ ] Validate CUBIT-Det
* [ ] Analyze dataset statistics

## Shared Taxonomy

Proposed mappings:

```text
MBDD2025             CUBIT-Det

crack       <------> crack

abscission  <------> spalling

leakage     <------> moisture
```

Shared taxonomy proposal:

```text
crack
surface_damage
moisture_damage
```

* [ ] document final mapping
* [ ] verify mapping assumptions
* [ ] ensure unmapped classes are handled explicitly

## Experiments

* [ ] Train MBDD → Test MBDD
* [ ] Train MBDD → Test CUBIT
* [ ] Train CUBIT → Test CUBIT
* [ ] Train CUBIT → Test MBDD
* [ ] Train combined → Test MBDD
* [ ] Train combined → Test CUBIT

## Analysis

* [ ] quantify domain shift
* [ ] compare per-class degradation
* [ ] investigate failure cases
* [ ] test one reasonable mitigation strategy

## Definition of Done

* cross-dataset evaluation is reproducible
* taxonomy is documented
* domain-shift effects are quantified
* no unsupported claims of generalization are made

## Release

`v0.5.0-cross-domain`

---

# M6 — Inference Packaging and ONNX

## Goal

Separate model research code from deployable inference code.

## Inference Package

* [ ] Implement preprocessing module
* [ ] Implement detector interface
* [ ] Implement postprocessing module
* [ ] Implement prediction schema
* [ ] Load model once per process
* [ ] Support single-image inference
* [ ] Support batch inference where practical

## ONNX

* [ ] Export trained model to ONNX
* [ ] Validate ONNX outputs
* [ ] Compare ONNX vs PyTorch predictions
* [ ] Benchmark CPU inference
* [ ] Benchmark GPU inference where available

## Benchmark

Record:

* latency
* FPS
* model size
* runtime
* hardware

## Definition of Done

* inference can run independently of training code
* exported model output is validated
* runtime benchmark is documented

## Release

`v0.6.0-inference`

---

# M7 — FastAPI Serving

## Goal

Expose InfraGuard through a clean REST API.

## Endpoints

* [ ] `GET /health`
* [ ] `GET /model/info`
* [ ] `POST /predict`
* [ ] `POST /predict/batch`

## Requirements

* [ ] request validation
* [ ] invalid-image handling
* [ ] structured error responses
* [ ] typed response schemas
* [ ] model loaded once
* [ ] inference latency returned where appropriate
* [ ] API documentation through OpenAPI/Swagger

## Tests

* [ ] health endpoint
* [ ] valid prediction
* [ ] invalid file
* [ ] unsupported image
* [ ] empty upload
* [ ] model loading

## Definition of Done

* API starts locally
* API tests pass
* a real UAV image can be uploaded and scored
* failures are handled gracefully

## Release

`v0.7.0-api`

---

# M8 — Docker and Harness CI/CD

## Goal

Make the project reproducible and automatically verifiable.

## Docker

* [ ] Add runtime Dockerfile
* [ ] Build image locally
* [ ] Run API inside container
* [ ] Validate health endpoint
* [ ] Ensure no dataset is included
* [ ] Ensure no credentials are included

## Harness CI

Pull Request pipeline:

* [ ] install dependencies
* [ ] Ruff
* [ ] pytest
* [ ] type checking if enabled
* [ ] inference smoke test
* [ ] Docker build
* [ ] security scan where configured

Important:

CI must not train the production model.

Use lightweight fixtures and smoke tests.

## Delivery

When appropriate:

* [ ] build tagged container
* [ ] push container artifact
* [ ] create release pipeline
* [ ] validate deployment

## Definition of Done

* PR checks are automated
* Docker build is reproducible
* main remains deployable
* CI does not require raw datasets

## Release

`v0.8.0-production-pipeline`

---

# M9 — Web Demo

## Goal

Provide a recruiter-friendly interactive demonstration.

Possible UI:

* Streamlit
* Gradio

Features:

* [ ] upload image
* [ ] display detections
* [ ] display detected defect count
* [ ] display class labels
* [ ] display confidence
* [ ] display inference latency
* [ ] handle invalid uploads
* [ ] connect to inference/API layer

Avoid building unnecessary frontend complexity.

## Definition of Done

A recruiter can upload an image and understand the model output without reading source code.

## Release

`v0.9.0-demo`

---

# M10 — Portfolio Release

## Goal

Transform the technical project into a professional AI Engineer portfolio artifact.

## README

* [ ] project summary
* [ ] architecture diagram
* [ ] dataset overview
* [ ] baseline results
* [ ] experiment results
* [ ] high-resolution inference results
* [ ] cross-dataset results
* [ ] error analysis
* [ ] runtime benchmarks
* [ ] API usage
* [ ] Docker instructions
* [ ] reproduction instructions
* [ ] limitations
* [ ] future work

## Visual Assets

* [ ] architecture diagram
* [ ] dataset distribution
* [ ] bounding-box distribution
* [ ] object-center heatmap
* [ ] confusion matrix
* [ ] per-class AP
* [ ] tiled-inference comparison
* [ ] cross-dataset comparison
* [ ] runtime benchmark
* [ ] failure cases
* [ ] demo GIF/video

## Documentation

* [ ] `DATASETS.md`
* [ ] `DECISIONS.md`
* [ ] `EXPERIMENTS.md`
* [ ] model card
* [ ] release notes

## CV

Prepare quantified bullet points using only actual results.

Do not fabricate metrics.

## Definition of Done

A technical recruiter should be able to understand within several minutes:

1. what problem was solved,
2. what you personally built,
3. what engineering decisions you made,
4. how the system performs,
5. where the model fails,
6. how to run the system.

## Release

`v1.0.0`

---

# Future Work — After v1.0

Potential extensions:

* CUBIT-InSeg instance segmentation
* damage-area quantification
* crack-length estimation
* TensorRT optimization
* improved domain adaptation
* model monitoring
* additional external datasets

These are not required for v1.0.

---

# Current Execution Rule

Always complete the earliest unfinished milestone that blocks later milestones.

Do not jump to deployment while the dataset pipeline is unreliable.

Do not optimize models before establishing a reproducible baseline.

Do not expand scope merely to add more technologies.
