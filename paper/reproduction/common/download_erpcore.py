"""Download ERP CORE N170 assets (subject 1 by default) into the cache.

Prefers the official processed subject folder from OSF node pfde9
("N170 All Data and Scripts") so ICA/preprocessing can be taken from the
ERP CORE pipeline rather than re-invented. Raw BIDS is optional.

This is a download helper only. Scientific choices (which ERP CORE stage
to treat as the paper input) belong to the N170 track.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hashing import sha256_file, write_json
    from paths import ERPCORE_DIR, ensure_data_layout
else:
    from .hashing import sha256_file, write_json
    from .paths import ERPCORE_DIR, ensure_data_layout

OSF_PFDE9 = "https://api.osf.io/v2/nodes/pfde9/files/osfstorage/"
ALL_DATA_FOLDER = "5f2479095f705a010e619b0a"
BIDS_FOLDER = "60060f8ae80d370812a5b15d"
SUB001_BIDS = "60060f9686541a084814ce9e"
ERP_CORE_SCRIPTS = "https://github.com/lucklab/ERP_CORE"


def _list_folder(folder_id: str) -> list[dict]:
    url = f"https://api.osf.io/v2/nodes/pfde9/files/osfstorage/{folder_id}/?page[size]=100"
    items: list[dict] = []
    while url:
        with _urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items.extend(payload.get("data", []))
        url = payload.get("links", {}).get("next")
    return items


USER_AGENT = "ReDisCA-paper-reproduction/1.0 (+https://github.com/ivsemenkov/ReDisCA-dev)"

SELECTED_SUBJECT_FILES = (
    "1_N170_erp_ar.erp",
    "1_N170_erp_ar_lpfilt.erp",
    "1_AR_Percentages_N170.csv",
    "1_N170_Eventlist_Bins.txt",
)


def _urlopen(url: str, timeout: int = 600):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


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


def _find_named_folder(items: list[dict], name: str) -> dict:
    for item in items:
        if item["attributes"]["name"] == name and item["attributes"]["kind"] == "folder":
            return item
    raise FileNotFoundError(f"OSF folder {name!r} not found")


def download_processed_subject(subject_id: str = "1") -> dict[str, str]:
    """Download ERP CORE processed files for one subject (paper uses index '1')."""
    ensure_data_layout()
    dest_root = ERPCORE_DIR / "all_data_and_scripts" / subject_id
    top = _list_folder(ALL_DATA_FOLDER)
    subject = _find_named_folder(top, subject_id)
    folder_id = subject["id"]
    files = _list_folder(folder_id)
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
        if dest.exists() and dest.stat().st_size == attrs.get("size"):
            digest = sha256_file(dest)
        else:
            if not url:
                raise FileNotFoundError(f"No download URL for {attrs['name']}")
            _download(url, dest)
            digest = sha256_file(dest)
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


def download_bids_subject001_listing() -> dict:
    """Record BIDS sub-001 file listing; download EEG files if small enough."""
    ensure_data_layout()
    files = _list_folder(SUB001_BIDS)
    listing = []
    for item in files:
        listing.append(
            {
                "name": item["attributes"]["name"],
                "kind": item["attributes"]["kind"],
                "size": item["attributes"].get("size"),
                "download": item.get("links", {}).get("download"),
                "id": item["id"],
            }
        )
    dest = ERPCORE_DIR / "bids" / "sub-001" / "listing.json"
    write_json(dest, listing)
    return {"listing": str(dest), "n_entries": len(listing)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ERP CORE N170 files.")
    parser.add_argument("subset", choices=["processed-subject-1", "bids-sub-001-listing"])
    args = parser.parse_args()
    if args.subset == "processed-subject-1":
        print(json.dumps(download_processed_subject("1"), indent=2))
    else:
        print(json.dumps(download_bids_subject001_listing(), indent=2))
