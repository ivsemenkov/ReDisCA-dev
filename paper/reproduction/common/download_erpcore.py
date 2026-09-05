"""Download ERP CORE N170 assets (subject 1 by default) into the cache."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .hashing import sha256_file, write_json
from .paths import ERPCORE_DIR, ensure_data_layout

ALL_DATA_FOLDER = "5f2479095f705a010e619b0a"
USER_AGENT = "ReDisCA-stage-a-reproduction/1.0 (+https://github.com/ivsemenkov/ReDisCA-dev)"

SELECTED_SUBJECT_FILES = (
    "1_N170_erp_ar.erp",
    "1_N170_erp_ar_lpfilt.erp",
    "1_AR_Percentages_N170.csv",
    "1_N170_Eventlist_Bins.txt",
)

# Official hashes recorded on the old paper branch from OSF pfde9.
EXPECTED_SHA256 = {
    "1_N170_erp_ar.erp": "53e74e931e6f0adaf1e5be4d606d028fcc3e04ee8b066569c2ed2d033d9bbc72",
    "1_N170_erp_ar_lpfilt.erp": "228b52ad69b9dc9b88f6b4c0b1d32dc778450e0d9aa32850b9ad9a8a61a8b9fe",
    "ICA_Components_N170.xlsx": "23373a2b7aae80e7b01abfdc523fb1d04fbc6f41fc48c090f7e840534224cf85",
}


def _urlopen(url: str, timeout: int = 600):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def _list_folder(folder_id: str) -> list[dict]:
    url = f"https://api.osf.io/v2/nodes/pfde9/files/osfstorage/{folder_id}/?page[size]=100"
    items: list[dict] = []
    while url:
        with _urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items.extend(payload.get("data", []))
        url = payload.get("links", {}).get("next")
    return items


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _urlopen(url) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)


def _find_named(items: list[dict], name: str, kind: str) -> dict:
    for item in items:
        if item["attributes"]["name"] == name and item["attributes"]["kind"] == kind:
            return item
    raise FileNotFoundError(f"OSF {kind} {name!r} not found")


def download_processed_subject(subject_id: str = "1") -> dict[str, str]:
    """Download official processed ERP CORE files for one subject."""
    ensure_data_layout()
    dest_root = ERPCORE_DIR / "all_data_and_scripts" / subject_id
    top = _list_folder(ALL_DATA_FOLDER)
    subject = _find_named(top, subject_id, "folder")
    files = _list_folder(subject["id"])
    paths: dict[str, str] = {}
    manifest = []
    for item in files:
        attrs = item["attributes"]
        if attrs["kind"] != "file":
            continue
        if attrs["name"] not in SELECTED_SUBJECT_FILES and not attrs["name"].endswith(".erp"):
            continue
        dest = dest_root / attrs["name"]
        url = item.get("links", {}).get("download")
        expected = EXPECTED_SHA256.get(attrs["name"])
        if dest.exists() and (expected is None or sha256_file(dest) == expected):
            digest = sha256_file(dest)
        else:
            if not url:
                raise FileNotFoundError(f"No download URL for {attrs['name']}")
            _download(url, dest)
            digest = sha256_file(dest)
            if expected and digest != expected:
                raise ValueError(f"SHA-256 mismatch for {attrs['name']}: got {digest}")
        paths[attrs["name"]] = str(dest)
        manifest.append(
            {
                "name": attrs["name"],
                "size_bytes": dest.stat().st_size,
                "sha256": digest,
                "osf_id": item["id"],
            }
        )
    write_json(dest_root / "downloaded.json", {"subject": subject_id, "files": manifest})
    return paths


def download_ica_xlsx() -> Path:
    """Download the official ICA component list (subject 1 removes 2 and 7)."""
    ensure_data_layout()
    dest = ERPCORE_DIR / "ICA_Components_N170.xlsx"
    expected = EXPECTED_SHA256["ICA_Components_N170.xlsx"]
    if dest.exists() and sha256_file(dest) == expected:
        return dest
    top = _list_folder(ALL_DATA_FOLDER)
    # The workbook lives next to the processing scripts, not in subject folders.
    for item in top:
        if item["attributes"]["name"] == "ICA_Components_N170.xlsx":
            url = item.get("links", {}).get("download")
            if not url:
                raise FileNotFoundError("No download URL for ICA_Components_N170.xlsx")
            _download(url, dest)
            digest = sha256_file(dest)
            if digest != expected:
                raise ValueError(f"SHA-256 mismatch for ICA xlsx: got {digest}")
            return dest
    # Search one level of named processing folders.
    for item in top:
        if item["attributes"]["kind"] != "folder":
            continue
        try:
            children = _list_folder(item["id"])
        except Exception:
            continue
        for child in children:
            if child["attributes"]["name"] == "ICA_Components_N170.xlsx":
                url = child.get("links", {}).get("download")
                _download(url, dest)
                digest = sha256_file(dest)
                if digest != expected:
                    raise ValueError(f"SHA-256 mismatch for ICA xlsx: got {digest}")
                return dest
    raise FileNotFoundError("ICA_Components_N170.xlsx not found on OSF node pfde9")
