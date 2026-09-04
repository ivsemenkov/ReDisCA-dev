"""Download and parse official ERP CORE ``ICA_Components_N170.xlsx``."""

from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from common.hashing import sha256_file
from common.paths import ERPCORE_DIR, ensure_data_layout

ICA_XLSX_NAME = "ICA_Components_N170.xlsx"
ICA_XLSX_URL = "https://osf.io/download/f9r7c/"
ICA_XLSX_SHA256 = "23373a2b7aae80e7b01abfdc523fb1d04fbc6f41fc48c090f7e840534224cf85"
ICA_XLSX_SIZE_BYTES = 9904
USER_AGENT = "ReDisCA-paper-reproduction/1.0 (+https://github.com/ivsemenkov/ReDisCA-dev)"
_SSML = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def default_xlsx_path() -> Path:
    return ERPCORE_DIR / ICA_XLSX_NAME


def download_ica_components_xlsx(dest: Path | None = None) -> Path:
    """Fetch the OSF spreadsheet if missing or hash-mismatched."""
    ensure_data_layout()
    dest = dest or default_xlsx_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == ICA_XLSX_SHA256:
        return dest
    request = urllib.request.Request(ICA_XLSX_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    dest.write_bytes(payload)
    digest = sha256_file(dest)
    if digest != ICA_XLSX_SHA256:
        raise ValueError(
            f"{dest} SHA-256 {digest} != pinned {ICA_XLSX_SHA256}"
        )
    return dest


def _column_index(ref: str) -> tuple[int, int]:
    match = re.match(r"([A-Z]+)(\d+)$", ref)
    if not match:
        raise ValueError(f"Bad cell ref {ref!r}")
    column = 0
    for char in match.group(1):
        column = column * 26 + (ord(char) - 64)
    return column, int(match.group(2))


def parse_ica_components_xlsx(path: Path) -> dict[str, list[int]]:
    """Map subject-id string -> 1-based ICA component indices."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: dict[int, dict[int, int]] = {}
    for cell in root.findall(".//m:c", _SSML):
        ref = cell.get("r")
        value_el = cell.find("m:v", _SSML)
        if ref is None or value_el is None or value_el.text is None:
            continue
        if cell.get("t") == "s":
            continue
        column, row = _column_index(ref)
        rows.setdefault(row, {})[column] = int(float(value_el.text))
    mapping: dict[str, list[int]] = {}
    for row, columns in rows.items():
        if row == 1:
            continue
        subject = columns.get(1)
        if subject is None:
            continue
        components = [columns[key] for key in sorted(columns) if key > 1]
        mapping[str(int(subject))] = components
    return mapping


def xlsx_summary(path: Path) -> dict[str, Any]:
    mapping = parse_ica_components_xlsx(path)
    counts = [len(values) for values in mapping.values()]
    histogram: dict[str, int] = {}
    for count in counts:
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "n_subjects": len(mapping),
        "subject_1_components_1based": list(mapping.get("1", [])),
        "count_histogram_n_components": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        "n_subjects_with_exactly_3": sum(1 for values in mapping.values() if len(values) == 3),
        "modal_n_components": int(max(histogram.items(), key=lambda kv: kv[1])[0]) if histogram else None,
        "mean_n_components": float(sum(counts) / len(counts)) if counts else None,
        "all_subjects": {key: mapping[key] for key in sorted(mapping, key=lambda item: int(item))},
        "script4_label": "ocular artifacts (not cardiac)",
        "source": {
            "url": ICA_XLSX_URL,
            "pinned_sha256": ICA_XLSX_SHA256,
            "datasets_md": "paper/reference/provenance/datasets.md",
        },
    }
