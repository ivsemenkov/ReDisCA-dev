"""Load ERPLAB ``.erp`` files produced by the official ERP CORE pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def load_erplab_erp(path: Path) -> dict[str, Any]:
    """Return condition-average EEG from an ERPLAB ``.erp`` file.

    ``data`` has shape ``(n_bins, n_channels, n_times)`` so it can be passed
    to ``ReDisCA.fit`` after channel selection. This loader does not decide
    which channels or bins the paper used.
    """
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    erp = payload["ERP"]
    bindata = np.asarray(erp.bindata, dtype=np.float64)
    # ERPLAB: (n_channels, n_times, n_bins)
    data = np.moveaxis(bindata, 2, 0)
    chanlocs = list(erp.chanlocs)
    labels = [str(ch.labels) for ch in chanlocs]
    xyz = []
    for ch in chanlocs:
        coords = []
        for name in ("X", "Y", "Z"):
            value = getattr(ch, name, None)
            arr = np.asarray(value).reshape(-1)
            coords.append(float("nan") if arr.size == 0 else float(arr[0]))
        xyz.append(coords)
    xyz = np.asarray(xyz, dtype=np.float64)
    ntrials = erp.ntrials
    accepted = np.asarray(getattr(ntrials, "accepted", []), dtype=np.int64)
    return {
        "path": str(path),
        "erpname": str(erp.erpname),
        "srate": float(erp.srate),
        "times_ms": np.asarray(erp.times, dtype=np.float64),
        "data": data,
        "channel_labels": labels,
        "channel_xyz": xyz,
        "bin_descriptions": [str(x) for x in np.atleast_1d(erp.bindescr)],
        "n_accepted": accepted,
        "xmin": float(erp.xmin),
        "xmax": float(erp.xmax),
        "isfilt": getattr(erp, "isfilt", None),
    }
