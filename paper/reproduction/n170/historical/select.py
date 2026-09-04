"""Qualitative Track B candidate selection.

This is not a scalar loss. Joint candidates are source-supported
(pair_mode, matrix_mode, car window) settings together with the paper
face window. The procedure is lexicographic and documented.
"""

from __future__ import annotations

from typing import Any

from .variants import FACE_CENTER_MS, PRINTED_CAR, PRINTED_FACE, variant_id_for


def _abs_delta(value: float | None, target: float) -> float:
    if value is None or value != value:
        return float("inf")
    return abs(float(value) - float(target))


def joint_settings_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["variant_id"]: row for row in rows}
    settings: list[dict[str, Any]] = []
    car_rows = [row for row in rows if row["contrast"] == "car"]
    for car in car_rows:
        face_id = variant_id_for(
            "face",
            FACE_CENTER_MS,
            car["pair_mode"],
            car["matrix_mode"],
        )
        face = by_id[face_id]
        face_p1 = face["primary_p_head"][0]
        car_p1 = car["primary_p_head"][0]
        car_p2 = car["primary_p_head"][1] if len(car["primary_p_head"]) > 1 else None
        settings.append(
            {
                "id": (
                    f"{car['pair_mode']}__{car['matrix_mode']}"
                    f"__car{int(car['window_center_ms'])}"
                ),
                "pair_mode": car["pair_mode"],
                "matrix_mode": car["matrix_mode"],
                "face_variant_id": face["variant_id"],
                "car_variant_id": car["variant_id"],
                "face_window": {
                    "center_ms": face["window_center_ms"],
                    "duration_ms": face["window_duration_ms"],
                },
                "car_window": {
                    "center_ms": car["window_center_ms"],
                    "duration_ms": car["window_duration_ms"],
                },
                "face_lambda1": face["eigenvalues_head"][0],
                "face_p1": face_p1,
                "face_corr_wTRw": face["corr_wTRw_comp0"],
                "face_corr_trace_sq": face["corr_trace_sq_comp0"],
                "face_peak_ms": face.get("faces_peak_ms"),
                "car_lambda1": car["eigenvalues_head"][0],
                "car_lambda2": car["eigenvalues_head"][1],
                "car_p1": car_p1,
                "car_p2": car_p2,
                "car_corr_wTRw": car["corr_wTRw_comp0"],
                "car_peak_ms": car.get("cars_peak_ms"),
                "abs_delta_face_lambda1": _abs_delta(
                    face["eigenvalues_head"][0], PRINTED_FACE["lambda1"]
                ),
                "abs_delta_car_lambda1": _abs_delta(
                    car["eigenvalues_head"][0], PRINTED_CAR["lambda1"]
                ),
                "abs_delta_car_lambda2": _abs_delta(
                    car["eigenvalues_head"][1], PRINTED_CAR["lambda2"]
                ),
                "abs_delta_car_p2": _abs_delta(car_p2, PRINTED_CAR["p2"]),
                "face_p1_is_zero": face_p1 == 0.0,
                "car_p1_is_zero": car_p1 == 0.0,
            }
        )
    return settings


def pick_two_joint_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return two joint settings by a lexicographic qualitative rule.

    Order of preference (no weights, no combined score):

    1. Prefer PRIMARY face p1 == 0 and PRIMARY car p1 == 0.
    2. Then smaller |car λ1 − 0.91639|.
    3. Then smaller |car λ2 − 0.77036|.
    4. Then smaller |car p2 − 0.009|.
    5. Then smaller |face λ1 − 0.87209|.

    Face window corr vs 0.82 is recorded and discussed; it is not used to
    discard a candidate because every source-supported window fit on this
    28-channel subject-1 average is expected to sit near 1, not 0.82.
    """
    settings = joint_settings_from_rows(rows)

    def sort_key(item: dict[str, Any]) -> tuple:
        both_p1_zero = item["face_p1_is_zero"] and item["car_p1_is_zero"]
        return (
            0 if both_p1_zero else 1,
            0 if item["car_p1_is_zero"] else 1,
            0 if item["face_p1_is_zero"] else 1,
            item["abs_delta_car_lambda1"],
            item["abs_delta_car_lambda2"],
            item["abs_delta_car_p2"],
            item["abs_delta_face_lambda1"],
            item["id"],
        )

    ranked = sorted(settings, key=sort_key)
    picked: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str, float]] = set()
    for item in ranked:
        signature = (
            item["pair_mode"],
            item["matrix_mode"],
            float(item["car_window"]["center_ms"]),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        picked.append(item)
        if len(picked) == 2:
            break
    for rank, item in enumerate(picked, start=1):
        item["selection_rank"] = rank
        item["selection_rule"] = (
            "lexicographic: both p1==0, then |car λ1-0.91639|, "
            "|car λ2-0.77036|, |car p2-0.009|, |face λ1-0.87209|; "
            "not a scalar loss; face corr vs 0.82 not used as a discard rule"
        )
    return picked
