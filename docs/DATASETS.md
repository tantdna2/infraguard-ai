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

### Day 5 Dataset Statistics

The statistics below were computed read-only from the official local MBDD2025
v1.0 release using the repository's public statistics API and CLI. They are
measured values, not values inferred from the paper.

#### Dataset and Annotation Counts

| Metric                    |  Count |
| ------------------------- | -----: |
| JPEG images               | 14,471 |
| YOLO label files          | 14,471 |
| Total annotation rows     | 57,613 |
| Usable annotations        | 57,613 |
| Excluded annotations      |      0 |
| Material OOB annotations  |    116 |
| Empty label files         |      8 |

The 116 material OOB annotations are a usable subset of the 57,613 source
annotations for statistics purposes; they are not additional instances.

| Class ID | Class      | Image count | Instance count |
| -------: | ---------- | ----------: | -------------: |
|        0 | crack      |       6,253 |         17,044 |
|        1 | leakage    |       2,464 |          6,642 |
|        2 | abscission |       5,166 |         22,702 |
|        3 | corrosion  |       2,102 |          9,207 |
|        4 | bulge      |       1,337 |          2,018 |

The class instance counts total 57,613, matching the usable annotation count.

#### Objects per Image

| Count  | Min | Max |   Mean | Median | P25 | P75 |
| -----: | --: | --: | -----: | -----: | --: | --: |
| 14,471 |   0 |  84 | 3.9813 |      2 |   1 |   5 |

#### Bounding Boxes

Bounding-box statistics use the normalized YOLO source values without
reconstructing or clipping width, height, area, or centers.

| Metric       |  Count |         Min |       Max |     Mean |   Median |      P25 |      P75 |
| ------------ | -----: | ----------: | --------: | -------: | -------: | -------: | -------: |
| Width        | 57,613 |    0.001086 |  1.000000 | 0.159931 | 0.090766 | 0.045359 | 0.194760 |
| Height       | 57,613 |    0.001944 |  1.000000 | 0.180540 | 0.109222 | 0.054944 | 0.218958 |
| Area         | 57,613 | 4.20801e-06 |  1.000000 | 0.032810 | 0.011452 | 0.003801 | 0.033355 |
| Aspect ratio | 57,613 |    0.014525 | 27.502565 | 1.618539 | 0.872121 | 0.414529 | 1.864153 |
| Center X     | 57,613 |    0.002922 |  0.998145 | 0.507413 | 0.508605 | 0.318563 | 0.694844 |
| Center Y     | 57,613 |    0.004868 |  0.995451 | 0.474777 | 0.467569 | 0.312417 | 0.625910 |

#### Image Statistics

| Metric     |  Count |       Min |        Max |        Mean |      Median |         P25 |         P75 |
| ---------- | -----: | --------: | ---------: | ----------: | ----------: | ----------: | ----------: |
| Width      | 14,471 |       720 |       1280 | 1267.627669 |        1280 |        1280 |        1280 |
| Height     | 14,471 |       544 |       1280 |  728.430654 |         720 |         720 |         720 |
| Brightness | 14,471 | 18.574378 | 185.177644 |  109.879749 |  109.137610 |   99.604147 |  120.668219 |
| Contrast   | 14,471 |  9.934902 |  91.460690 |   57.856010 |   59.485999 |   50.230766 |   67.342122 |

The exact decoded-image resolution distribution is:

| Resolution | Image count |
| ---------- | ----------: |
| 1280 x 720 |      14,102 |
| 720 x 1280 |         254 |
| 960 x 544  |         115 |

Displayed continuous statistics are rounded for readability; deterministic
JSON output preserves the computed float values.

#### Methodology

- YOLO TXT is the primary annotation representation. Each usable source row is
  one annotation instance. Exact duplicate source rows are not deduplicated.
- A class image count includes an image at most once for that class, regardless
  of how many instances of the class the image contains.
- Empty labels represent zero-object images. Malformed or semantically unusable
  rows are excluded from annotation-derived aggregates, with one explicit root
  cause recorded for each exclusion.
- Finite source rows with valid normalized components and positive sizes remain
  usable when their reconstructed XYXY boxes are materially out of bounds.
  Their original YOLO values are retained without clamping, clipping, rewriting,
  or modifying the raw annotation files.
- Numeric summaries contain `count`, `min`, `max`, `mean`, `median`, `p25`, and
  `p75`. Percentiles use linear interpolation at the zero-based index
  `(n - 1) * q`.
- Image width and height come from decoded JPEGs. Exact resolution counts group
  decoded `(width, height)` pairs. A discovered JPEG that cannot be decoded
  fails the statistics run explicitly rather than being skipped.
- Each image is converted to Pillow mode `L`. Its brightness is the arithmetic
  mean of its grayscale pixel intensities on the `0` (black) to `255` (white)
  scale.
- An image's contrast is the population standard deviation of those mode `L`
  intensities around that image's brightness, in intensity units on the same
  `0`-to-`255` scale.
- Each image contributes one brightness observation and one contrast
  observation to the dataset summaries. Images are not weighted by pixel count.

The statistics policy does not mean that the dataset passes validation. The
Day 4 validator reports the same 116 material boxes as
`ERROR OUT_OF_BOUNDS_BOX` and the 8 empty labels as `INFO EMPTY_LABEL`.
Retaining the OOB source rows in aggregates is an intentional statistics policy
that preserves source data; it does not reclassify, repair, or hide the
validator findings. In the verified run, all 57,613 rows were usable for the
statistics policy and no rows were excluded. The exclusion counters for
malformed rows, invalid classes, invalid coordinates, negative sizes, and
zero-area boxes were all zero.

#### Reproduction

Print a concise summary using the canonical dataset configuration:

```bash
python scripts/dataset_statistics.py \
  --dataset-root data/raw/mbdd2025/MBDD2025/
```

Write the full deterministic, strict JSON report to a path outside both
`data/raw/` and the supplied dataset root:

```bash
python scripts/dataset_statistics.py \
  --dataset-root data/raw/mbdd2025/MBDD2025/ \
  --output <PATH_OUTSIDE_DATA_RAW>
```

Repeated read-only verification produced identical typed results across two
independent core runs and byte-identical JSON across two CLI runs. Strict JSON
serialization succeeded, and the CLI JSON matched the public API result.

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
