"""Download OSF assets into the gitignored reproduction cache."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from .hashing import sha256_file, write_json
from .paths import MEG_DIR, OSF_DIR, SOURCE_MODEL_DIR, ensure_data_layout

OSF_8RK67_FILES: dict[str, dict[str, Any]] = {
    "MEG_AD_run1.mat": {
        "download": "https://osf.io/download/h9zpq/",
        "sha256": "0eca2756c9190ce637a3e14abd24e7cf975d758d3ccea03107963e8b5841a4f6",
        "size_bytes": 1243214548,
        "dest_dir": "meg",
        "required_for": ["meg_sensor"],
    },
    "ibfctfprespm8_AD_run1_raw_tsss_mc.mat": {
        "download": "https://osf.io/download/673e184585d2961fe2886e03/",
        "sha256": "87890337c385e81c718c421d7be35e54423ca9ceb985e047b276b02018334950",
        "size_bytes": 62701,
        "dest_dir": "meg",
        "required_for": ["meg_sensor"],
        "note": "SPM trial labels; companion .dat is not loaded by the AIRI main script",
    },
    "topo_face_vs_tool_correct_filt15.mat": {
        "download": "https://osf.io/download/gxbe5/",
        "sha256": "b18be3e159164846c0e9d82e3d7dd62e1f01e53d00b511b10f11bd1f8b3b7328",
        "size_bytes": 105738,
        "dest_dir": "meg",
        "required_for": ["source_localization"],
        "note": "Author-saved A1; not a substitute for a local ReDisCA run",
    },
    "headmodel_surf_os_meg.mat": {
        "download": "https://osf.io/download/2afzg/",
        "sha256": "a365912cae29c3ddda7be90b4bb3830f4ce081e7d4de1206d0c1406985ec439c",
        "size_bytes": 35957600,
        "dest_dir": "source_models",
        "required_for": ["source_localization", "simulations"],
    },
    "results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat": {
        "download": "https://osf.io/download/673e19715cbaa22c0a75e832/",
        "sha256": "794043eb34f588a14186b297721d78e71ac9a08187938f611bdf0a0e0a92a1d3",
        "size_bytes": 13203960,
        "dest_dir": "source_models",
        "required_for": ["source_localization"],
    },
    "tess_cortex_pial_low.mat": {
        "download": "https://osf.io/download/673e1974b0f7255a4475e61b/",
        "sha256": "40502997c4c21d89a4c7ea207ab77c1c458a005d74e7cb78e6e0e2beb578cad1",
        "size_bytes": 481530,
        "dest_dir": "source_models",
        "required_for": ["source_localization", "simulations"],
    },
}

OSF_8RK67_API = "https://api.osf.io/v2/nodes/8rk67/files/osfstorage/?filter[kind]=file"


def _dest_dir(name: str) -> Path:
    spec = OSF_8RK67_FILES[name]
    key = spec["dest_dir"]
    if key == "meg":
        return MEG_DIR
    if key == "source_models":
        return SOURCE_MODEL_DIR
    return OSF_DIR / key


def _osf_catalog() -> dict[str, dict[str, Any]]:
    request = urllib.request.Request(
        OSF_8RK67_API,
        headers={"User-Agent": "ReDisCA-stage-a-reproduction/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    catalog = {}
    for item in payload.get("data", []):
        attrs = item["attributes"]
        links = item.get("links", {})
        catalog[attrs["name"]] = {
            "download": links.get("download"),
            "sha256": attrs.get("extra", {}).get("hashes", {}).get("sha256"),
            "size_bytes": attrs.get("size"),
        }
    return catalog


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "ReDisCA-stage-a-reproduction/1.0"}
    )
    with urllib.request.urlopen(request, timeout=600) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)


def ensure_osf_file(name: str, *, catalog: dict[str, dict[str, Any]] | None = None) -> Path:
    spec = OSF_8RK67_FILES[name]
    dest = _dest_dir(name) / name
    if dest.exists() and sha256_file(dest) == spec["sha256"]:
        return dest
    catalog = catalog or _osf_catalog()
    remote = catalog.get(name, {})
    url = remote.get("download") or spec.get("download")
    if not url:
        raise FileNotFoundError(
            f"No download URL for {name}. Query {OSF_8RK67_API} and update the manifest."
        )
    download_file(url, dest)
    digest = sha256_file(dest)
    if spec["sha256"] and digest != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {name}: got {digest}")
    return dest


def download_subset(required_for: str) -> dict[str, str]:
    ensure_data_layout()
    catalog = _osf_catalog()
    paths = {}
    for name, spec in OSF_8RK67_FILES.items():
        if required_for in spec["required_for"]:
            path = ensure_osf_file(name, catalog=catalog)
            paths[name] = str(path)
    dest = MEG_DIR if required_for == "meg_sensor" else SOURCE_MODEL_DIR
    write_json(dest / "downloaded.json", paths)
    return paths


def download_meg_sensor_assets() -> dict[str, str]:
    return download_subset("meg_sensor")


def download_source_model_assets() -> dict[str, str]:
    return download_subset("source_localization")


def download_simulation_forward_assets() -> dict[str, str]:
    return download_subset("simulations")
