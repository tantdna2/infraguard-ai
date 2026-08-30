## Summary

<!--
Briefly explain what this Pull Request changes.

Example:
Implements the MBDD2025 dataset validator, including annotation checks,
corrupted-image detection, CLI support, and unit tests.
-->

---

## Related Issue

Closes #

<!--
Every implementation PR should reference a GitHub Issue.
-->

---

## Why

<!--
Why is this change necessary?

Explain the technical or project reason rather than only describing the code.
-->

---

## Changes

<!--
List the main changes.

Example:
- Added MBDD2025 YOLO annotation parser.
- Added validation for normalized bounding-box coordinates.
- Added corrupted-image checks.
- Added CLI validation command.
- Added unit tests.
-->

*

---

## Files / Components Affected

<!--
Examples:
src/infraguard/data/
tests/
configs/
docs/
-->

*

---

## How to Test

Commands executed:

```bash
# Add exact commands here.
```

Expected result:

```text
# Example:
35 tests passed
Ruff passed
```

---

## Test Coverage

New or updated tests:

* [ ] normal / expected behavior
* [ ] invalid input
* [ ] edge cases
* [ ] failure behavior

Describe relevant cases:

*

---

# ML / Data Impact

## Dataset Changes

Does this PR modify or reinterpret dataset data?

* [ ] No
* [ ] Yes

If yes, explain:

*

---

## Dataset Split Changes

Does this PR modify:

* training split

* validation split

* test split

* [ ] No

* [ ] Yes

If yes, explain exactly why:

*

---

## Label / Taxonomy Changes

Does this PR modify:

* class IDs

* class names

* annotation interpretation

* taxonomy mapping

* [ ] No

* [ ] Yes

If yes, explain:

*

---

## Preprocessing Changes

Does this PR change preprocessing such as:

* resizing

* normalization

* image conversion

* augmentation

* cropping

* tiling

* [ ] No

* [ ] Yes

If yes, explain:

*

---

## Metric / Evaluation Changes

Does this PR change:

* metric definitions

* evaluation thresholds

* IoU thresholds

* confidence thresholds

* evaluation dataset

* test procedure

* [ ] No

* [ ] Yes

If yes, explain:

*

---

## Reproducibility Impact

Does this PR change:

* random seed

* training parameters

* experiment configuration

* dataset sampling

* dependency versions

* [ ] No

* [ ] Yes

If yes, explain:

*

---

# Data Leakage Review

Could this change create or hide data leakage (rò rỉ dữ liệu)?

Consider:

* duplicates across splits
* augmented samples across splits
* preprocessing fitted on test data
* test-set-based model selection
* cached outputs from another split

Assessment:

* [ ] No known leakage risk
* [ ] Leakage risk reviewed and documented
* [ ] Potential leakage risk remains

Notes:

*

---

# Performance Impact

Does this PR affect runtime performance?

* [ ] No meaningful impact
* [ ] Faster
* [ ] Slower
* [ ] Unknown / not benchmarked

If benchmarked, provide results:

| Metric     | Before | After |
| ---------- | -----: | ----: |
| Latency    |        |       |
| FPS        |        |       |
| Memory     |        |       |
| Model size |        |       |

---

# Experiment Evidence

Is this PR based on an experiment?

* [ ] No
* [ ] Yes

If yes:

**Experiment ID / document:**

**Hypothesis:**

**Baseline:**

**Treatment:**

**Result:**

**Conclusion:**

---

# Visual Verification

For changes affecting images, predictions, plots, UI, or bounding boxes:

* [ ] Not applicable
* [ ] Visually verified

Evidence:

<!--
Add screenshots, plots, or a brief description where appropriate.
-->

---

# Documentation

Documentation updated:

* [ ] Not required
* [ ] README
* [ ] ROADMAP
* [ ] DATASETS
* [ ] DECISIONS
* [ ] EXPERIMENTS
* [ ] API documentation
* [ ] other

Notes:

*

---

# Known Limitations

<!--
Be explicit. Do not hide unresolved limitations.

Example:
The validator currently checks exact duplicate annotations but does not
perform perceptual duplicate-image detection.
-->

*

---

# Agent Verification

Implementation performed by:

* [ ] Human
* [ ] Codex
* [ ] Antigravity
* [ ] Other

Independent verification performed by:

* [ ] Human
* [ ] Codex
* [ ] Antigravity
* [ ] Automated tests
* [ ] Not yet independently verified

Verification notes:

*

---

# Final Checklist

## Engineering

* [ ] Scope matches the linked Issue
* [ ] No unrelated refactor included
* [ ] Public functions use appropriate type hints
* [ ] Paths are not hardcoded
* [ ] New dependencies are justified
* [ ] Errors are handled appropriately

## Testing

* [ ] Relevant tests were added or updated
* [ ] `pytest` passes
* [ ] `ruff check .` passes
* [ ] formatting checks pass if configured

## ML / Data

* [ ] No dataset files were unintentionally modified
* [ ] No model weights were committed
* [ ] Dataset split changes are documented
* [ ] Metric changes are documented
* [ ] Data leakage risk was considered
* [ ] Reproducibility impact was considered

## Security

* [ ] No API keys or secrets committed
* [ ] No `.env` committed
* [ ] No private credentials included

## Documentation

* [ ] Relevant documentation is current

---

# Completion Report

## What was completed?

*

## What was verified?

*

## What remains uncertain?

*

## Recommended next step

*
