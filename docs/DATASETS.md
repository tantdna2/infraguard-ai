# InfraGuard AI — Datasets

## MBDD2025

### Registration Status

- Local archive available.
- Local archive identity matches the official Zenodo release.
- Official metadata verified.

### Local Dataset

Local storage convention:

```text
data/raw/mbdd2025/MBDD2025.zip
```

The archive is local only and is ignored by Git through the repository's
`data/*` rule. It is raw data and must be treated as read-only. Inspection and
validation tools must not modify the archive or its contents in place.

### Integrity

- Local archive checksum (SHA256):
  `db37469e0ee59be132d0e3773affec89a1c49fad3a873a9d47e7221bcfc3f95e`
- Local archive checksum (MD5): `b2dfdce060ef687c327b1f8203b52636`
- Official Zenodo checksum (MD5): `b2dfdce060ef687c327b1f8203b52636`

The local MD5 matches the checksum published by Zenodo. The SHA256 value is a
local archive checksum; no official SHA256 checksum has been verified.

### Official Source

- Dataset record: <https://zenodo.org/records/15622584>
- Dataset DOI: <https://doi.org/10.5281/zenodo.15622584>
- Official archive: <https://zenodo.org/records/15622584/files/MBDD2025.zip>

### Publication

- **Paper:** "A dataset of building surface defects collected by UAVs for
  machine learning-based detection"
- **Authors:** Qikai Zha, Yiming Yao, Yufan Zheng, Wanqi Ma, and Wenkang Zhang
- **Year:** 2025
- **Venue:** *Scientific Data* (Nature Portfolio), Volume 12, Article 2031
- **DOI:** <https://doi.org/10.1038/s41597-025-06318-5>

### Version

The official Zenodo release is explicitly versioned as **v1.0** and was
published on 2025-06-09.

### License and Usage Terms

The dataset is released under the **Creative Commons Attribution 4.0
International (CC BY 4.0)** license. The license permits sharing and adaptation,
including commercial use, provided its attribution and other terms are
followed.

This dataset license is separate from the repository's MIT License. The MIT
License for InfraGuard AI source code does not apply to MBDD2025.

### Dataset Structure

**VERIFIED:** The locally observed structure matches the structure documented
by the official source.

```text
MBDD2025/
├── Annotations/    # 14,471 PASCAL VOC XML files
├── JPEGImages/     # 14,471 JPEG images
├── Labels/         # 14,471 YOLO TXT files
└── README.md       # Dataset metadata, license, and citation
```

The official release does not contain fixed train, validation, or test
directories or file lists.

### Annotation Format

**Officially verified:** MBDD2025 provides two annotation representations for
2D object detection:

- PASCAL VOC XML in `Annotations/`, using unnormalized pixel coordinates
  `xmin`, `ymin`, `xmax`, and `ymax`.
- YOLO TXT in `Labels/`, using
  `class_id x_center y_center width height`, with coordinates normalized to
  `[0, 1]`.

### Project Loader Decision

- InfraGuard AI currently uses YOLO TXT as the primary MBDD2025 loader
  representation.
- PASCAL VOC XML remains available as an independent annotation representation
  for future systematic cross-validation. Day 4 used it only for limited,
  targeted inspection of selected findings.

### Class Taxonomy

**Officially verified:** The following ID-to-name mapping is canonical for the
official MBDD2025 v1.0 release.

| Class ID | Class name |
| -------- | ---------- |
| 0        | crack      |
| 1        | leakage    |
| 2        | abscission |
| 3        | corrosion  |
| 4        | bulge      |

### Day 4 Dataset Validation

The Day 4 validator implements YOLO syntax and semantic validation, corrupt and
unreadable image checks, image-label matching, and deterministic issue ordering.
It can emit a deterministic, machine-readable JSON report with a schema version
and portable dataset-relative paths.

Reconstructed XYXY coordinates use an OOB tolerance of `1e-9`. This tolerance
only suppresses floating-point reconstruction noise at normalized image
boundaries. It is not clipping, does not alter annotation or `BoundingBox`
values, and does not permit material out-of-bounds annotations.

The final read-only validation of the local MBDD2025 v1.0 release produced:

| Validation code     | Severity | Count |
| ------------------- | -------- | ----: |
| `OUT_OF_BOUNDS_BOX` | ERROR    |   116 |
| `EMPTY_LABEL`       | INFO     |     8 |

Limited, targeted inspection of corresponding PASCAL VOC annotations supports
the conclusion that the 116 material OOB results are source-annotation quality
findings. This is targeted evidence only; a full YOLO-versus-VOC
cross-validation has not been completed.

### Known Unknowns

- The official release does not provide fixed train, validation, or test file
  lists.
- The publication describes a 70%/20%/10% train/validation/test ratio, but the
  exact sample membership, random seed, and reproducible split procedure have
  not been verified.
- No official SHA256 checksum has been verified; the SHA256 value above is
  local evidence only.

### Evidence Sources

- Zenodo dataset record: <https://zenodo.org/records/15622584>
- Zenodo dataset DOI: <https://doi.org/10.5281/zenodo.15622584>
- Official archive download:
  <https://zenodo.org/records/15622584/files/MBDD2025.zip>
- Peer-reviewed publication:
  <https://doi.org/10.1038/s41597-025-06318-5>

Dataset configuration completion remains part of the MBDD2025 Data Foundation
milestone.

---

## CUBIT-Det

Status: Planned / Unverified

### Usage

Candidate external dataset for a future cross-dataset generalization evaluation.

### Dataset Metadata and Taxonomy

Source, version, license, annotation format, class taxonomy, and any cross-dataset mapping remain unverified. No mapping is canonical during repository bootstrap.
