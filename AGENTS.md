# InfraGuard AI — Development Guidelines

## 1. Project Overview

**InfraGuard AI** is an end-to-end Computer Vision (thị giác máy tính) project for detecting building surface defects from UAV imagery.

The project is designed to demonstrate practical AI Engineering (kỹ thuật xây dựng hệ thống AI), including:

* dataset engineering
* data validation
* exploratory data analysis
* model training
* model evaluation
* error analysis
* high-resolution inference
* cross-dataset generalization
* model optimization
* API serving
* containerized deployment
* automated testing
* reproducible experiments

The primary dataset is **MBDD2025**.

An external dataset such as **CUBIT-Det** may be used for cross-dataset evaluation (đánh giá chéo giữa các bộ dữ liệu).

---

# 2. Source of Truth

The GitHub repository is the canonical source of truth for the project.

Important technical information must be recorded in the repository rather than existing only in local notes or conversations.

Relevant sources include:

* source code
* configuration files
* GitHub Issues
* Pull Requests
* experiment records
* technical documentation
* release notes

Important project documentation is stored under:

```text
docs/
├── ROADMAP.md
├── DECISIONS.md
├── DATASETS.md
└── EXPERIMENTS.md
```

If documentation and implementation disagree, the inconsistency should be identified and resolved explicitly.

Before modifying the repository, inspect the current branch, working-tree status, repository structure, and the files relevant to the active Issue.

---

# 3. Development Workflow

All meaningful changes should follow:

```text
GitHub Issue
     ↓
Feature Branch
     ↓
Implementation
     ↓
Tests
     ↓
Pull Request
     ↓
Review
     ↓
Merge
```

Do not push feature implementations directly to `main`.

Recommended branch naming:

```text
feat/<issue>-<description>
fix/<issue>-<description>
docs/<issue>-<description>
chore/<issue>-<description>
```

Examples:

```text
feat/5-mbdd-loader
feat/12-tiled-inference
fix/18-bbox-coordinate
docs/23-model-card
```

Each Pull Request should reference the Issue it implements.

---

# 4. Scope Discipline

Each Issue should solve one clearly defined problem.

Avoid mixing unrelated changes into the same Pull Request.

Do not introduce additional features that are outside the current Issue unless they are necessary for correctness.

When additional work is discovered:

1. document it,
2. propose a follow-up Issue,
3. keep the current Pull Request focused.

Avoid unnecessary refactoring during feature implementation.

---

# 5. Repository Structure

Reusable production code belongs under:

```text
src/infraguard/
```

Expected high-level structure:

```text
src/infraguard/
├── data/
├── training/
├── evaluation/
├── inference/
├── serving/
└── utils/
```

Command-line entry points belong under:

```text
scripts/
```

Exploratory notebooks belong under:

```text
notebooks/
```

Tests belong under:

```text
tests/
```

Experiment and dataset configuration belongs under:

```text
configs/
```

Reusable logic must not exist only inside notebooks.

---

# 6. Python Guidelines

The project currently targets:

```text
Python 3.11
```

General rules:

* use type hints for public functions
* prefer small and focused functions
* avoid unnecessary global state
* prefer `pathlib.Path` for filesystem operations
* do not hardcode absolute paths
* use clear exception handling
* use logging for reusable application code
* avoid unnecessary dependencies
* do not add heavyweight dependencies or Deep Learning frameworks unless the current Issue explicitly requires them
* do not create fake functions, classes, or implementations solely to fill placeholders
* keep modules focused on a single responsibility

Configuration values should not be hidden inside Python source when they can reasonably live in configuration files.

If the available environment does not provide Python 3.11, report the mismatch clearly and do not claim incompatible checks passed.

---

# 7. Configuration-Driven Development

Important parameters should be configurable.

Examples include:

```text
dataset path
class names
model name
image size
batch size
epochs
random seed
confidence threshold
IoU threshold
tile size
tile overlap
```

Configuration files should live under:

```text
configs/
```

An experiment should be reproducible from its saved configuration whenever possible.

---

# 8. Dataset Safety

Raw datasets are treated as read-only.

Do not download any dataset unless the current Issue explicitly requires it.

Do not modify original downloaded images or annotations in place.

Recommended layout:

```text
data/
├── raw/
├── interim/
└── processed/
```

Transformations should create new derived outputs rather than modifying `data/raw/`.

Never silently:

* remove samples
* change annotations
* change class IDs
* rename classes
* modify train/validation/test splits
* regenerate dataset augmentations

Any intentional dataset transformation must be documented.

---

# 9. Git and Artifact Safety

Do not commit raw datasets or large model artifacts to normal Git history.

Files that should normally remain outside Git include:

```text
datasets
training checkpoints
model weights
training runs
large prediction artifacts
environment secrets
```

Common model artifact extensions include:

```text
.pt
.pth
.onnx
```

Model releases should use an appropriate artifact storage mechanism when needed.

---

# 10. Dataset Validation

Dataset validation should happen before serious model training.

Validation should consider:

```text
missing images
missing labels
empty annotation files
corrupted images
malformed annotation rows
invalid class IDs
invalid bounding boxes
zero-area bounding boxes
out-of-bounds coordinates
duplicate annotations
```

Validation tools should report problems instead of silently correcting raw data.

When useful, findings should be categorized as:

```text
ERROR
WARNING
INFO
```

---

# 11. Data Leakage

Data leakage (rò rỉ dữ liệu) is considered a critical Machine Learning (học máy) issue.

Always consider whether:

```text
duplicate images exist across splits
near-duplicate images exist across splits
augmented variants exist across splits
test data influences preprocessing
test results influence model selection
```

Do not alter dataset splits merely to improve reported metrics.

If leakage is suspected:

1. document the evidence,
2. preserve the original benchmark where appropriate,
3. create a separate leakage-safe evaluation if necessary.

---

# 12. Reproducibility

Experiments should be reproducible whenever practical.

Record important information such as:

```text
dataset version
dataset split
random seed
model version
image size
training configuration
augmentation configuration
evaluation thresholds
library versions
hardware information
```

Comparable experiments should not silently change important variables.

---

# 13. Model Development

The initial baseline model is:

```text
YOLO11n
```

unless a documented technical decision supersedes it.

Establish a baseline before optimization.

Avoid introducing:

```text
larger models
complex augmentation
custom losses
tiling
balancing strategies
```

before a trustworthy baseline has been measured.

Model changes should be motivated by observed evidence.

---

# 14. Evaluation

Object detection experiments should report appropriate metrics such as:

```text
Precision
Recall
F1-score
mAP@0.5
mAP@0.5:0.95
per-class AP
confusion matrix
inference latency
```

Where relevant, also measure:

```text
FPS
model size
parameter count
memory usage
small-object performance
```

Metric definitions must remain consistent across comparable experiments.

Do not report training-set performance as model evaluation.

---

# 15. Experiment Design

Important experiments should be hypothesis-driven.

A useful experiment record should contain:

```text
Hypothesis
Baseline
Treatment
Dataset
Configuration
Metrics
Results
Conclusion
```

Example:

```text
Hypothesis:
Tiled inference improves recall for small cracks.

Baseline:
YOLO11n using standard 640×640 resize inference.

Treatment:
640×640 overlapping tiles.

Metrics:
mAP@0.5:0.95
small-object AP
crack recall
latency
```

Avoid random hyperparameter experimentation without a clear question.

Experiment history should be recorded in:

```text
docs/EXPERIMENTS.md
```

or an equivalent experiment directory.

---

# 16. Error Analysis

Model performance should not be evaluated only through aggregate metrics.

Important models should be inspected for:

```text
false positives
false negatives
localization errors
class confusion
small defects
thin cracks
low-light images
low-contrast defects
edge-position defects
unusual backgrounds
```

Failure cases should be documented rather than hidden.

Observed model failures should guide later improvements.

---

# 17. High-Resolution Inference

Tiled inference (suy luận bằng cách chia ảnh thành các ô) is a planned core feature.

Any tiled-inference implementation must correctly handle:

```text
tile generation
overlap
border tiles
local-to-global coordinate conversion
image-bound clipping
duplicate detections
NMS
empty predictions
```

Coordinate transformation must have dedicated unit tests.

A visually correct result on one image is not sufficient verification.

---

# 18. Cross-Dataset Evaluation

InfraGuard AI is expected to evaluate generalization (khả năng tổng quát hóa) across different datasets.

MBDD2025 and CUBIT-Det may have different taxonomies.

Class mappings must therefore be:

* explicit
* documented
* justified
* reproducible

Do not silently reinterpret labels to make datasets compatible.

Cross-dataset performance should be reported separately from internal test performance.

---

# 19. Testing

New functionality should include appropriate tests.

Relevant development checks include:

```bash
pytest
ruff check .
```

If formatting checks are configured:

```bash
ruff format --check .
```

Do not claim that tests passed unless they were actually executed.

If testing cannot be completed because of environment limitations, document the limitation clearly.

---

# 20. CI Test Data

Continuous Integration (tích hợp liên tục) must not depend on downloading the full training dataset.

Use lightweight fixtures under:

```text
tests/fixtures/
```

Synthetic images and annotations may be used for validation tests.

Test fixtures should cover behavior rather than reproduce the full dataset.

---

# 21. Security

Never commit:

```text
API keys
access tokens
passwords
private credentials
cloud credentials
.env files
```

Environment-specific secrets should use environment variables or the relevant secret-management system.

A public example may be provided through:

```text
.env.example
```

using placeholder values only.

---

# 22. API Design

The planned serving layer uses FastAPI.

HTTP handling and model inference should remain separated.

The serving layer should eventually support endpoints such as:

```text
GET /health
GET /model/info
POST /predict
POST /predict/batch
```

API implementations should:

```text
validate inputs
return structured responses
handle invalid files gracefully
avoid loading the model on every request
expose predictable errors
```

---

# 23. Deployment

The project is expected to support Docker-based deployment.

The final runtime container should:

```text
build reproducibly
contain only required runtime dependencies
exclude raw datasets
exclude secrets
provide a predictable entry point
```

Deployment optimization should not take priority over correctness during early development.

---

# 24. Documentation

Documentation should evolve together with the code.

Important files include:

```text
README.md
docs/ROADMAP.md
docs/DECISIONS.md
docs/DATASETS.md
docs/EXPERIMENTS.md
```

Update the relevant documentation whenever a change affects:

```text
dataset interpretation
model behavior
evaluation
architecture
public API behavior
experiment methodology
```

---

# 25. Technical Decisions

Important architecture and Machine Learning (học máy) decisions must be recorded in:

```text
docs/DECISIONS.md
```

Do not silently contradict an Accepted Architecture Decision Record (bản ghi quyết định kiến trúc).

If a previous decision must change, record the new evidence and superseding decision.

---

# 26. Pull Request Requirements

Every meaningful Pull Request should clearly describe:

```text
what changed
why it changed
which Issue it addresses
how it was tested
whether ML/data behavior changed
known limitations
```

Changes affecting any of the following require explicit disclosure:

```text
dataset splits
labels
taxonomy
preprocessing
metrics
random seeds
training behavior
evaluation behavior
```

---

# 27. Definition of Done

A task is not complete merely because code has been written.

A feature is considered complete when:

```text
requirements are implemented
acceptance criteria are satisfied
relevant tests pass
lint checks pass
documentation is current
no unintended data or model artifacts are committed
ML/data implications are understood
the Pull Request is reviewable
```

For experiment tasks, completion also requires:

```text
results
comparison with baseline
conclusion
```

---

# 28. Project Priorities

When trade-offs occur, prioritize:

```text
1. Correctness
2. Data integrity
3. Reproducibility
4. Testability
5. Clarity
6. Maintainability
7. Performance
8. Optimization
```

Do not trade correctness or reproducibility for small performance gains.

---

# 29. v1.0 Scope

InfraGuard AI v1.0 focuses on:

```text
UAV building surface defect detection
```

The following are currently outside the core v1.0 scope:

```text
satellite damage assessment
xBD integration
LLM features
RAG systems
AI agents inside the product
Kubernetes
microservices
mobile applications
Digital Twin systems
custom neural network architectures
```

Future extensions should be considered only after the core system is complete.

---

# 30. Engineering Principle

The objective of InfraGuard AI is not merely to train a model with a high score.

The objective is to build a reliable, explainable, reproducible, and deployable AI system.

Every technical decision should contribute to that goal.
