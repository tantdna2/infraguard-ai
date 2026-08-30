# InfraGuard AI — Technical Decision Log

This document records important technical and Machine Learning (học máy) decisions made during the development of InfraGuard AI.

The goal is to preserve context behind important choices so that humans and AI agents can understand:

* what was decided,
* why it was decided,
* what alternatives were considered,
* what trade-offs were accepted,
* and whether the decision is still active.

GitHub `main` is the canonical source of truth for accepted decisions.

---

# Decision Status

Each decision must use one of the following statuses:

* **Proposed** — under discussion, not yet adopted
* **Accepted** — currently active
* **Superseded** — replaced by a newer decision
* **Rejected** — considered but intentionally not adopted
* **Deprecated** — still present but should no longer be used

Do not silently modify an Accepted decision.

If an Accepted decision must change:

1. create a new ADR,
2. explain why the old decision is no longer appropriate,
3. mark the old ADR as `Superseded`,
4. link the new ADR.

---

# ADR Template

Use this template when adding a new decision.

```text
## ADR-XXX — Decision title

**Date:** YYYY-MM-DD
**Status:** Proposed / Accepted / Superseded / Rejected / Deprecated

### Context

What problem or uncertainty required a decision?

### Decision

What was decided?

### Rationale

Why was this choice made?

### Alternatives Considered

What other options were evaluated?

### Trade-offs

What disadvantages or limitations are accepted?

### Consequences

What does this decision require from future development?

### Revisit When

Under what conditions should this decision be reconsidered?

### Related

- Issue:
- Pull Request:
- Experiment:
- ADR:
```

---

# ADR-001 — Use Python 3.11

**Date:** 2026-08-30
**Status:** Accepted

## Context

InfraGuard AI will use Computer Vision (thị giác máy tính), Deep Learning (học sâu), data processing (xử lý dữ liệu), model serving (phục vụ mô hình), testing (kiểm thử), and deployment tooling (công cụ triển khai).

The Python version should provide:

* broad library compatibility,
* stable GPU and PyTorch ecosystem support,
* modern typing features,
* predictable local and CI environments.

## Decision

Use:

```text
Python 3.11
```

as the default project Python version.

## Rationale

Python 3.11 provides a mature balance between modern Python features and compatibility with common Machine Learning (học máy) libraries.

It reduces the risk of dependency incompatibilities compared with adopting the newest Python release too early.

## Alternatives Considered

### Python 3.10

Advantages:

* very mature ecosystem support.

Disadvantages:

* older than necessary for a new project.

### Python 3.12+

Advantages:

* newer language/runtime improvements.

Disadvantages:

* some Computer Vision (thị giác máy tính), CUDA, ONNX, and deployment packages may lag in compatibility.

## Trade-offs

The project may not use features available only in later Python versions.

## Consequences

Local development, CI, Docker, and documentation should target Python 3.11 unless this ADR is superseded.

## Revisit When

Reconsider after major dependencies officially and consistently support a newer Python version.

---

# ADR-002 — MBDD2025 Is the Primary Dataset

**Date:** 2026-08-30
**Status:** Accepted

## Context

InfraGuard AI needs a primary dataset suitable for UAV-based building surface defect detection.

The project should support a realistic AI Engineering (kỹ thuật xây dựng hệ thống AI) workflow including:

* dataset validation,
* exploratory data analysis,
* object detection,
* model evaluation,
* error analysis,
* and high-resolution inference.

## Decision

Use **MBDD2025** as the primary dataset for the first complete version of InfraGuard AI.

The initial target classes are:

```text
crack
leakage
abscission
corrosion
bulge
```

## Rationale

MBDD2025 provides:

* UAV imagery,
* multiple building surface defect categories,
* object detection annotations,
* sufficient scale for meaningful training,
* a published benchmark,
* known dataset limitations that can be investigated.

It is well suited to the project's goal of demonstrating both Computer Vision (thị giác máy tính) and AI Engineering (kỹ thuật hệ thống AI).

## Alternatives Considered

### CUBIT-Det as the primary dataset

Advantages:

* high-resolution imagery,
* useful infrastructure defects.

Disadvantages:

* fewer target classes,
* less appropriate as the single core dataset for the intended first version.

### xBD

Advantages:

* large satellite disaster-damage dataset.

Disadvantages:

* solves a fundamentally different problem:
  disaster-level building damage from satellite imagery rather than close-range surface defect detection.

## Trade-offs

MBDD2025 has potential limitations including:

* class imbalance,
* augmented samples in minority classes,
* spatial bias,
* domain-specific imagery.

These limitations must be explicitly investigated rather than ignored.

## Consequences

MBDD2025-specific data handling must be implemented first.

No other dataset should delay completion of the MBDD2025 baseline.

## Revisit When

Reconsider only if dataset quality issues make reliable training or evaluation impossible.

---

# ADR-003 — Use CUBIT-Det for External Validation

**Date:** 2026-08-30
**Status:** Accepted

## Context

A model performing well only on MBDD2025 does not demonstrate strong generalization (khả năng tổng quát hóa).

InfraGuard AI should evaluate domain shift (sự thay đổi miền dữ liệu) using a different real infrastructure dataset.

## Decision

Use **CUBIT-Det** as the primary external validation dataset.

## Rationale

CUBIT-Det provides infrastructure defect imagery from a different data distribution.

It enables experiments such as:

```text
Train MBDD2025 → Test MBDD2025
Train MBDD2025 → Test CUBIT-Det
Train CUBIT-Det → Test MBDD2025
Train combined → Test both
```

This creates a stronger portfolio project than reporting only internal test metrics.

## Alternatives Considered

### CODEBRIM

Useful for external testing but less directly aligned with the initial shared object-detection taxonomy.

### dacl1k

Useful for real-world damage analysis but introduces additional taxonomy and dataset-integration complexity.

### Multiple external datasets immediately

Rejected for v1.0 because it would expand scope before the core pipeline is stable.

## Trade-offs

MBDD2025 and CUBIT-Det do not use identical class taxonomies.

A documented taxonomy mapping will be required.

## Consequences

CUBIT-Det integration should occur only after the MBDD2025 baseline and error-analysis pipeline are stable.

## Revisit When

Add more external datasets after v1.0 if additional generalization evidence is valuable.

---

# ADR-004 — YOLO11n Is the Initial Baseline

**Date:** 2026-08-30
**Status:** Accepted

## Context

The project needs a simple, reproducible object detection baseline (mô hình mốc).

The goal of the first experiment is not to achieve maximum possible accuracy.

The goal is to establish a trustworthy reference point for later experiments.

## Decision

Use:

```text
YOLO11n
```

as the initial object detection baseline (mô hình phát hiện đối tượng mốc).

## Rationale

YOLO11n is:

* lightweight,
* easy to train,
* suitable for rapid experimentation,
* appropriate for latency benchmarking,
* easy to export for later deployment.

Using a small baseline makes later trade-offs between accuracy and performance easier to interpret.

## Alternatives Considered

### YOLO11s

Potentially stronger accuracy but less minimal as a baseline.

### RT-DETR

Useful for model comparison but introduces a different architecture before the data pipeline is validated.

### Faster R-CNN

Well-established detector but slower and less aligned with the deployment-oriented goals of this project.

### Custom architecture

Rejected because inventing a new architecture is outside the v1.0 scope.

## Trade-offs

YOLO11n may not provide the highest possible mAP.

This is acceptable because baseline simplicity and reproducibility are more important initially.

## Consequences

Model improvements must be measured relative to this baseline unless a newer baseline ADR supersedes this decision.

## Revisit When

After baseline completion and error analysis, YOLO11s or RT-DETR may be introduced as comparison models.

---

# ADR-005 — GitHub Main Is the Single Source of Truth

**Date:** 2026-08-30
**Status:** Accepted

## Context

InfraGuard AI will be developed using multiple AI tools:

* ChatGPT,
* Codex,
* Antigravity,
* Harness,
* GitHub.

Different tools may have different conversation contexts or temporary memory.

Without a central project state, decisions can become inconsistent.

## Decision

The `main` branch of the GitHub repository is the **single source of truth (nguồn sự thật duy nhất)**.

The official project state must be reconstructable from:

```text
source code
Git history
GitHub Issues
Pull Requests
docs/
configs/
experiment records
release notes
```

## Rationale

This allows ChatGPT to inspect the repository after each implementation cycle and decide the next step based on actual project state.

It also prevents important decisions from existing only inside chat history.

## Alternatives Considered

### Chat history as project memory

Rejected because chat context is not a reliable long-term engineering record.

### Local notes

Rejected as the primary source because other agents cannot reliably access them.

### One AI agent maintaining all state

Rejected because it creates dependency on one tool's context.

## Trade-offs

Developers must spend additional effort keeping repository documentation current.

## Consequences

Important decisions, experiments, and project-state changes must be committed to GitHub.

ChatGPT should inspect GitHub before making major next-step recommendations whenever repository state may have changed.

## Revisit When

This decision should remain stable throughout the project.

---

# ADR-006 — Pull Requests Are Required Before Main

**Date:** 2026-08-30
**Status:** Accepted

## Context

Codex and Antigravity will generate and modify production code.

Allowing AI-generated changes to be pushed directly to `main` increases the risk of:

* incorrect implementations,
* unnoticed regressions,
* ML/data mistakes,
* undocumented behavior changes.

## Decision

No implementation agent should push directly to `main`.

Required flow:

```text
Issue
→ branch
→ implementation
→ tests
→ Pull Request
→ CI
→ ChatGPT review
→ human merge decision
```

## Rationale

Pull Requests provide a review boundary where:

* code changes can be inspected,
* tests can run,
* ChatGPT can review the diff,
* data/ML risks can be assessed,
* the human owner retains final control.

## Alternatives Considered

### Direct push to main

Rejected due to insufficient review and rollback discipline.

### AI-controlled automatic merge

Rejected for the initial project phase.

May be reconsidered only after the project becomes mature and CI coverage is strong.

## Trade-offs

Development requires additional Git operations and PR overhead.

## Consequences

All feature work should have a linked Issue and feature branch.

## Revisit When

Automatic merge may be considered after v1.0 if:

* CI is reliable,
* review policy is mature,
* automated checks cover critical behavior.

---

# ADR-007 — ChatGPT Acts as Technical Reviewer and Next-Step Planner

**Date:** 2026-08-30
**Status:** Accepted

## Context

The project uses multiple coding agents.

A consistent authority is needed to review project progress and decide what work should happen next.

## Decision

ChatGPT acts as:

* AI technical reviewer,
* architecture reviewer,
* ML/data reviewer,
* roadmap planner.

After a coding agent pushes a Pull Request, ChatGPT should inspect the GitHub state before recommending the next task.

## Rationale

This separates:

```text
implementation
```

from:

```text
technical direction
```

Codex and Antigravity should focus on implementing scoped tasks.

ChatGPT evaluates how those tasks fit the broader project.

## Alternatives Considered

### Codex decides the next task

Rejected because the implementation agent should not silently expand its own scope.

### Antigravity decides the roadmap

Rejected for the same reason.

### Static roadmap only

Insufficient because experiment outcomes may require adapting future work.

## Trade-offs

The workflow requires an explicit review cycle after meaningful PRs.

## Consequences

Agents should not independently add major roadmap items without a GitHub Issue or explicit direction.

## Revisit When

This workflow may evolve if a more reliable automated project-management system is introduced.

---

# ADR-008 — Human Owner Retains Final Merge Authority

**Date:** 2026-08-30
**Status:** Accepted

## Context

ChatGPT may review Pull Requests, while Codex and Antigravity may implement changes.

However, the project is intended to demonstrate the owner's engineering knowledge and decision-making.

## Decision

The human project owner retains final authority to merge Pull Requests into `main`.

ChatGPT may recommend:

```text
APPROVE
REQUEST CHANGES
```

but approval does not automatically imply merge.

## Rationale

The owner must remain accountable for understanding the code and technical decisions.

This is especially important for an AI Engineering portfolio project.

## Alternatives Considered

### Automatic merge after ChatGPT approval

Potentially useful later, but inappropriate during early development.

## Trade-offs

Manual merge requires one additional human step.

## Consequences

The project owner should inspect major changes and understand why they are being merged.

## Revisit When

Potential automation may be considered after project maturity.

---

# ADR-009 — Codex Is the Default Implementation Agent

**Date:** 2026-08-30
**Status:** Accepted

## Context

Codex and Antigravity can both modify code.

Using both agents without clear role separation risks duplicated effort and conflicting implementations.

## Decision

Codex is the default implementation agent for repository-level software tasks.

Examples:

* dataset loaders,
* validators,
* reusable Python modules,
* tests,
* training scripts,
* evaluation tools,
* inference modules,
* FastAPI,
* ONNX integration,
* Docker.

## Rationale

This creates clear task ownership and reduces unnecessary agent overlap.

## Alternatives Considered

### Both agents implement every task

Rejected because it wastes effort and creates conflicting code paths.

### Antigravity as primary implementation agent

Possible for specific interactive tasks, but not the default.

## Trade-offs

Some tasks may be better handled by Antigravity and require explicit reassignment.

## Consequences

Unless the current Issue specifies otherwise, implementation prompts should be targeted to Codex.

## Revisit When

Reconsider if actual project experience shows another division of responsibilities is more effective.

---

# ADR-010 — Antigravity Is the Default Interactive Verification Agent

**Date:** 2026-08-30
**Status:** Accepted

## Context

Automated tests cannot catch all integration and visual issues.

The project will eventually include:

* image pipelines,
* inference outputs,
* API endpoints,
* web demonstrations.

## Decision

Antigravity is the default agent for interactive verification and integration debugging.

Typical tasks include:

* running the project locally,
* testing CLI behavior,
* inspecting actual predictions,
* verifying API responses,
* browser testing,
* testing the web demo,
* investigating environment failures.

## Rationale

This creates independent verification after Codex implementation.

## Alternatives Considered

### Codex writes and validates everything

Rejected as the default because independent review provides stronger quality control.

## Trade-offs

Requires context handoff between agents.

## Consequences

For significant features, Antigravity should verify real behavior before final approval when practical.

## Revisit When

Role allocation may change if tooling capabilities evolve.

---

# ADR-011 — Harness Provides Deterministic CI/CD Checks

**Date:** 2026-08-30
**Status:** Accepted

## Context

AI review is useful but should not replace repeatable automated verification.

The project requires deterministic checks for engineering correctness.

## Decision

Harness will be used for CI/CD (tích hợp và triển khai liên tục) and automated engineering checks.

Initial Pull Request checks should include:

```text
dependency installation
Ruff
pytest
type checking when configured
lightweight smoke tests
```

Later stages may add:

```text
Docker build
security scanning
inference smoke test
deployment pipeline
```

## Rationale

Harness provides repeatable checks independent of AI judgment.

## Alternatives Considered

### AI-only review

Rejected because AI review is not deterministic.

### Train the full model in CI

Rejected because it is expensive and unnecessary.

## Trade-offs

CI configuration adds infrastructure work.

## Consequences

Full MBDD2025 must not be required for CI.

Tests must use lightweight fixtures.

## Revisit When

CI stages should expand incrementally as project complexity grows.

---

# ADR-012 — Production Logic Must Not Live Only in Notebooks

**Date:** 2026-08-30
**Status:** Accepted

## Context

Many Machine Learning (học máy) portfolio projects place all logic inside Jupyter notebooks.

This makes code difficult to:

* test,
* reuse,
* deploy,
* maintain.

## Decision

Reusable production logic must live under:

```text
src/infraguard/
```

Notebooks may use package functions but should not be the sole implementation of reusable functionality.

## Rationale

The project targets an AI Engineer role, where maintainable software structure matters alongside model quality.

## Alternatives Considered

### Notebook-first repository

Rejected as the main architecture.

Notebooks remain appropriate for exploratory analysis.

## Trade-offs

Some exploratory code must later be extracted into reusable modules.

## Consequences

EDA notebooks should import reusable code from `src/infraguard`.

## Revisit When

No planned revisit.

---

# ADR-013 — Use Configuration-Driven Experiments

**Date:** 2026-08-30
**Status:** Accepted

## Context

Hardcoded training and evaluation parameters make experiments difficult to reproduce.

## Decision

Experiment parameters should be stored under:

```text
configs/
```

Configuration should cover relevant values such as:

```text
dataset
model
image size
epochs
batch size
seed
augmentation
confidence threshold
IoU threshold
tile size
tile overlap
```

## Rationale

Configuration-driven experimentation improves reproducibility and makes comparisons easier.

## Alternatives Considered

### Hardcoded Python constants

Rejected because they hide experiment state.

### Command-line-only experiment history

Rejected because terminal commands are easy to lose.

## Trade-offs

Configuration management adds some complexity.

## Consequences

Important experiment runs should preserve their configuration alongside their reported results.

## Revisit When

A dedicated experiment-management framework may later supersede the configuration format.

---

# ADR-014 — Dataset Raw Files Are Read-Only

**Date:** 2026-08-30
**Status:** Accepted

## Context

Data-processing scripts or agents may accidentally modify original labels or images.

This would undermine reproducibility.

## Decision

Directories containing original downloaded datasets are treated as read-only.

Example:

```text
data/raw/
```

All transformations should produce derived outputs elsewhere.

Example:

```text
data/processed/
data/interim/
```

## Rationale

Preserving raw data makes all transformations traceable.

## Alternatives Considered

### Edit source annotations in-place

Rejected.

## Trade-offs

Derived datasets require additional disk space.

## Consequences

Validators should report problems rather than silently repairing original annotations.

## Revisit When

No planned revisit.

---

# ADR-015 — Do Not Commit Raw Datasets or Large Model Weights

**Date:** 2026-08-30
**Status:** Accepted

## Context

MBDD2025 and future model artifacts are too large and may have redistribution restrictions.

## Decision

Do not commit:

```text
raw dataset images
large dataset archives
.pt
.pth
.onnx
training checkpoints
large experiment artifacts
```

to normal Git history.

## Rationale

This keeps repository history clean and avoids licensing or repository-size problems.

## Alternatives Considered

### Git LFS

May be useful later for selected artifacts but is unnecessary for raw datasets.

### Commit everything directly

Rejected.

## Trade-offs

Users need separate setup steps to obtain datasets and model artifacts.

## Consequences

`data/README.md` must explain dataset acquisition.

Release model artifacts should use an appropriate artifact-storage mechanism.

## Revisit When

Git LFS or another artifact manager may be considered for selected files later.

---

# ADR-016 — Data Validation Must Precede Model Training

**Date:** 2026-08-30
**Status:** Accepted

## Context

Starting training before understanding dataset integrity risks producing misleading metrics and hidden failures.

## Decision

No serious baseline training should begin until the MBDD2025 data foundation is complete.

Required first:

* dataset registration,
* dataset loader,
* annotation validation,
* class statistics,
* bounding-box analysis,
* duplicate investigation,
* leakage investigation.

## Rationale

Data quality is a prerequisite for trustworthy Machine Learning (học máy) results.

## Alternatives Considered

### Train immediately and inspect data later

Rejected because errors may contaminate early conclusions.

## Trade-offs

The first model training is delayed.

## Consequences

Week 1 focuses primarily on data readiness rather than model optimization.

## Revisit When

No planned revisit.

---

# ADR-017 — Data Leakage Is a Blocking Issue

**Date:** 2026-08-30
**Status:** Accepted

## Context

MBDD2025 includes augmented samples, especially around the minority `bulge` class.

Near-duplicate samples appearing across train, validation, or test splits could inflate evaluation metrics.

## Decision

Potential data leakage (rò rỉ dữ liệu) is considered a blocking ML issue.

The project must explicitly investigate:

* exact duplicates,
* near duplicates,
* augmented variants,
* duplicate samples across splits.

## Rationale

An impressive metric is meaningless if evaluation independence is compromised.

This analysis also provides valuable AI Engineering portfolio evidence.

## Alternatives Considered

### Trust the published split without auditing

Rejected.

### Immediately replace the official split

Rejected.

The official split should first be audited and documented before alternative evaluation strategies are introduced.

## Trade-offs

Duplicate analysis requires additional engineering work.

## Consequences

If leakage is detected:

1. preserve the original benchmark result where appropriate,
2. clearly label it,
3. create an additional leakage-safe evaluation,
4. never hide the finding.

## Revisit When

After the Week 1 duplicate and leakage audit.

---

# ADR-018 — Baseline Before Optimization

**Date:** 2026-08-30
**Status:** Accepted

## Context

It is tempting to introduce many improvements immediately:

* larger models,
* complex augmentations,
* tiling,
* balancing,
* custom loss functions.

Without a baseline, their real impact cannot be measured reliably.

## Decision

Establish a simple YOLO11n baseline before introducing model improvements.

## Rationale

Every improvement needs a trustworthy reference point.

## Alternatives Considered

### Start directly with an optimized pipeline

Rejected because it prevents meaningful ablation and comparison.

## Trade-offs

The first result may not be impressive.

## Consequences

Improvement experiments must report comparable metrics against the baseline.

## Revisit When

After baseline evaluation and error analysis.

---

# ADR-019 — Error Analysis Before Major Model Changes

**Date:** 2026-08-30
**Status:** Accepted

## Context

Blindly trying larger models or random hyperparameters may improve metrics without providing understanding.

## Decision

After the baseline, perform error analysis before selecting major improvements.

Investigate:

* false positives,
* false negatives,
* small objects,
* class confusion,
* edge cases,
* low-light images,
* low-contrast defects,
* localization failures.

## Rationale

Model changes should target observed failure modes.

## Alternatives Considered

### Hyperparameter search immediately after baseline

Rejected as the primary strategy.

## Trade-offs

Requires manual inspection and analysis time.

## Consequences

Each major improvement should be associated with a documented hypothesis.

## Revisit When

No planned revisit.

---

# ADR-020 — Tiled Inference Is a Core v1.0 Feature

**Date:** 2026-08-30
**Status:** Accepted

## Context

Building defects such as cracks may occupy very small image regions.

Resizing high-resolution imagery directly to a small model input may destroy useful visual detail.

## Decision

Implement and benchmark tiled inference (suy luận bằng cách chia ảnh thành các ô) as a core v1.0 project feature.

## Rationale

This provides:

* a technically meaningful Computer Vision problem,
* a strong portfolio feature,
* an accuracy-vs-latency engineering trade-off,
* direct relevance to high-resolution UAV imagery.

## Alternatives Considered

### Only increase model input size

Useful as a comparison but does not fully address very high-resolution images.

### Use only standard resize inference

Insufficient for the intended portfolio depth.

## Trade-offs

Tiled inference increases:

* latency,
* implementation complexity,
* duplicate prediction handling.

## Consequences

The implementation requires dedicated tests for:

* tile boundaries,
* coordinate transformation,
* overlap,
* duplicate detections,
* NMS.

## Revisit When

After tiled inference benchmarking.

---

# ADR-021 — Cross-Dataset Evaluation Is Required for v1.0

**Date:** 2026-08-30
**Status:** Accepted

## Context

Internal test performance does not prove real-world generalization.

## Decision

InfraGuard AI v1.0 must include at least one external cross-dataset evaluation using CUBIT-Det.

## Rationale

This demonstrates awareness of domain shift (sự thay đổi miền dữ liệu), a real issue in deployed AI systems.

## Alternatives Considered

### MBDD-only evaluation

Rejected as the final project scope.

It remains sufficient for early milestones.

## Trade-offs

Taxonomy mapping and additional data engineering are required.

## Consequences

Shared class mapping must be explicitly documented and justified.

## Revisit When

After completing cross-dataset experiments.

---

# ADR-022 — FastAPI Is the Default Model Serving Layer

**Date:** 2026-08-30
**Status:** Accepted

## Context

The project needs to demonstrate that a trained model can be exposed as a usable software service.

## Decision

Use FastAPI as the default REST API (giao diện lập trình REST) framework.

Expected endpoints include:

```text
GET /health
GET /model/info
POST /predict
POST /predict/batch
```

## Rationale

FastAPI provides:

* typed request/response schemas,
* automatic OpenAPI documentation,
* straightforward testing,
* strong relevance to AI Engineer roles.

## Alternatives Considered

### Flask

Simpler but provides less built-in typing and schema support.

### Django

Unnecessarily heavy for model-serving scope.

## Trade-offs

Adds a serving dependency.

## Consequences

HTTP concerns and model inference logic must remain separated.

## Revisit When

Only if deployment requirements materially change.

---

# ADR-023 — ONNX Is the Initial Deployment Export Format

**Date:** 2026-08-30
**Status:** Accepted

## Context

The project should demonstrate model portability and inference optimization beyond training inside PyTorch.

## Decision

Export the selected production model to ONNX (định dạng trao đổi mô hình).

Benchmark at minimum:

```text
PyTorch inference
vs
ONNX Runtime inference
```

## Rationale

ONNX is widely used for deployment interoperability and gives the project an engineering dimension beyond training.

## Alternatives Considered

### TensorRT first

Potentially faster on NVIDIA hardware but more environment-specific.

### PyTorch-only serving

Simpler but provides less deployment depth.

## Trade-offs

Export validation is required because runtime outputs may differ slightly.

## Consequences

ONNX output correctness must be checked against the original model before benchmarking.

## Revisit When

TensorRT may be added after ONNX integration is stable.

---

# ADR-024 — Docker Is Required for v1.0

**Date:** 2026-08-30
**Status:** Accepted

## Context

Local Python environments vary.

A recruiter or reviewer should be able to run the inference API predictably.

## Decision

Containerize the final serving application with Docker.

## Rationale

Docker demonstrates:

* reproducible runtime setup,
* dependency isolation,
* deployment readiness.

## Alternatives Considered

### Local-only installation

Insufficient for the final AI Engineering portfolio target.

### Kubernetes

Rejected as unnecessary complexity for v1.0.

## Trade-offs

Requires additional build configuration.

## Consequences

The final Docker image must not include:

* raw datasets,
* secrets,
* unnecessary development files.

## Revisit When

No planned revisit for v1.0.

---

# ADR-025 — v1.0 Will Not Include Satellite Damage Assessment

**Date:** 2026-08-30
**Status:** Accepted

## Context

The broader infrastructure-damage domain includes both UAV surface inspection and satellite disaster assessment.

Combining both would dramatically increase project scope.

## Decision

InfraGuard AI v1.0 focuses on:

```text
UAV building surface defect detection
```

Satellite imagery and xBD are explicitly out of scope.

## Rationale

A focused, complete project is more valuable than an unfinished system covering too many modalities.

## Alternatives Considered

### Combine MBDD2025 and xBD

Rejected because they represent fundamentally different:

* image scales,
* sensor viewpoints,
* labels,
* tasks.

## Trade-offs

The project will not demonstrate satellite imagery experience in v1.0.

## Consequences

Satellite damage assessment may become a separate future project.

## Revisit When

After InfraGuard AI v1.0 is complete.

---

# ADR-026 — v1.0 Will Not Add LLM/RAG Features Without Product Need

**Date:** 2026-08-30
**Status:** Accepted

## Context

AI portfolio projects are often expanded with LLM (mô hình ngôn ngữ lớn), RAG (sinh nội dung có truy xuất dữ liệu), or AI Agent (tác tử AI) features even when those technologies do not solve the core problem.

## Decision

Do not introduce LLM, RAG, or AI-agent features into InfraGuard AI v1.0 unless a concrete product requirement appears.

## Rationale

The project should demonstrate depth in Computer Vision (thị giác máy tính) and AI Engineering rather than stacking unrelated technologies.

## Alternatives Considered

### Generate inspection reports using an LLM

Potential future extension but not necessary for proving the core AI Engineer skills.

## Trade-offs

The project will contain fewer trendy technologies.

## Consequences

Engineering time remains focused on:

* data quality,
* model performance,
* inference,
* deployment,
* reliability.

## Revisit When

After the Computer Vision pipeline is complete and stable.

---

# ADR-027 — Metrics Must Never Be Fabricated or Cherry-Picked

**Date:** 2026-08-30
**Status:** Accepted

## Context

The final project will be used in a CV and portfolio.

Metrics may create pressure to present only favorable results.

## Decision

All reported project performance must come from reproducible experiments.

Do not fabricate, estimate, or selectively report metrics in a misleading way.

## Rationale

Engineering credibility matters more than artificially impressive numbers.

## Alternatives Considered

None.

## Trade-offs

Some project results may be lower than expected.

## Consequences

CV bullet points must use actual measured values.

Failure cases and known limitations should remain visible in project documentation.

## Revisit When

Never.

---

# ADR-028 — AI-Generated Code Must Be Understandable by the Project Owner

**Date:** 2026-08-30
**Status:** Accepted

## Context

Codex and Antigravity can accelerate implementation, but this project will be used during technical interviews.

The owner may be asked to explain implementation details.

## Decision

No AI-generated implementation should be merged if the project owner cannot reasonably explain:

* what the code does,
* why it was designed that way,
* important trade-offs,
* major failure modes.

## Rationale

The purpose of the project is to demonstrate the owner's engineering ability, not merely the ability to delegate work to AI.

## Alternatives Considered

### Merge code based only on passing tests

Rejected.

Tests do not prove understanding.

## Trade-offs

Some implementation work will require additional reading before merging.

## Consequences

Major PRs should be reviewed not only for correctness but also for explainability.

## Revisit When

Never for this portfolio project.

---

# ADR-029 — Use pyproject.toml for Python Packaging and Dependencies

**Date:** 2026-08-30
**Status:** Accepted

## Context

The repository needs one standard location for package metadata, dependency groups, package discovery, and development-tool configuration.

## Decision

Use `pyproject.toml` with setuptools as the build backend.

The project uses the `src` layout, with package discovery rooted at:

```text
src/infraguard/
```

Runtime dependencies belong in the main project dependency list. Development and notebook tools belong in separate optional dependency groups.

## Rationale

This follows current Python packaging standards and supports a reproducible editable development installation without introducing an additional package manager.

## Alternatives Considered

### Separate requirements files

Possible, but would split dependency declarations across multiple files during the initial bootstrap.

### Poetry, PDM, or Hatch

Not required for the current project scope.

## Trade-offs

Setuptools configuration remains part of the project metadata and must be maintained with the package structure.

## Consequences

Development environments should be installable with:

```text
pip install -e ".[dev,notebook]"
```

## Revisit When

Reconsider only if packaging or reproducibility requirements outgrow the current setup.

---

# ADR-030 — Defer Deep Learning Frameworks During Repository Bootstrap

**Date:** 2026-08-30
**Status:** Accepted

## Context

The repository-bootstrap milestone establishes project structure, data tooling dependencies, documentation, and quality checks. Model training is not part of this milestone.

## Decision

Do not add Deep Learning framework dependencies such as PyTorch, TorchVision, or Ultralytics during bootstrap.

Add them only in a later Issue that implements the documented model-training stage and can justify the required versions.

## Rationale

Deferring large, platform-sensitive dependencies keeps the initial environment focused and avoids choosing training packages before their requirements are known.

## Alternatives Considered

### Install the planned model stack immediately

Rejected because ADR-004 selects a future baseline but does not require model implementation during repository bootstrap.

## Trade-offs

The bootstrap environment cannot train or run the planned detector.

## Consequences

Day 1 validation covers packaging, repository structure, and development tooling only. Model dependencies must be introduced through a scoped future Issue.

## Revisit When

Revisit when work begins on the baseline training milestone.

---

# Current Accepted Decision Summary

| ADR     | Decision                               | Status   |
| ------- | -------------------------------------- | -------- |
| ADR-001 | Python 3.11                            | Accepted |
| ADR-002 | MBDD2025 primary dataset               | Accepted |
| ADR-003 | CUBIT-Det external validation          | Accepted |
| ADR-004 | YOLO11n baseline                       | Accepted |
| ADR-005 | GitHub main is source of truth         | Accepted |
| ADR-006 | PR required before main                | Accepted |
| ADR-007 | ChatGPT reviews and plans next work    | Accepted |
| ADR-008 | Human retains merge authority          | Accepted |
| ADR-009 | Codex default implementation agent     | Accepted |
| ADR-010 | Antigravity default verification agent | Accepted |
| ADR-011 | Harness for deterministic CI/CD        | Accepted |
| ADR-012 | Production logic outside notebooks     | Accepted |
| ADR-013 | Configuration-driven experiments       | Accepted |
| ADR-014 | Raw datasets are read-only             | Accepted |
| ADR-015 | No raw data or large weights in Git    | Accepted |
| ADR-016 | Validate data before model training    | Accepted |
| ADR-017 | Data leakage is blocking               | Accepted |
| ADR-018 | Baseline before optimization           | Accepted |
| ADR-019 | Error analysis before major changes    | Accepted |
| ADR-020 | Tiled inference required               | Accepted |
| ADR-021 | Cross-dataset evaluation required      | Accepted |
| ADR-022 | FastAPI serving                        | Accepted |
| ADR-023 | ONNX deployment format                 | Accepted |
| ADR-024 | Docker required                        | Accepted |
| ADR-025 | Satellite imagery out of v1.0          | Accepted |
| ADR-026 | No unnecessary LLM/RAG features        | Accepted |
| ADR-027 | Metrics must be real and reproducible  | Accepted |
| ADR-028 | AI-generated code must be explainable  | Accepted |
| ADR-029 | pyproject.toml packaging                | Accepted |
| ADR-030 | Defer Deep Learning frameworks         | Accepted |

---

# Maintenance Rule

Whenever a future GitHub Issue introduces a change that contradicts one of these ADRs:

1. do not silently implement the contradiction,
2. identify the affected ADR,
3. propose a new ADR,
4. explain the evidence for changing the decision,
5. update this document after approval.

The purpose of this file is not to prevent the project from evolving.

Its purpose is to make project evolution intentional, documented, and reproducible.
