"""Nominal vs literal AIRI plotting-axis mapping."""

from __future__ import annotations

import numpy as np

from paper.reproduction.meg.time_axes import (
    AIRI_LITERAL_N_SAMPLES,
    airi_literal_plotting_axis_ms,
    dual_times_from_sample,
    nominal_axis_ms,
    paper_anchors_on_both_axes,
    remap_interval,
    sample_index_from_nominal_ms,
)


def test_airi_literal_axis_is_exactly_matlab_linspace():
    axis = airi_literal_plotting_axis_ms(1501)
    expected = np.linspace(-536.0, 964.0, 1501)
    assert axis.shape == (1501,)
    assert np.array_equal(axis, expected)
    assert axis[0] == -536.0
    assert axis[-1] == 964.0
    assert np.isclose(axis[1] - axis[0], 1.0)


def test_known_sample_maps_to_both_axes():
    # sample 605: nominal −500+605 = 105 ms; AIRI −536+605 = 69 ms
    dual = dual_times_from_sample(605)
    assert dual["nominal_ms"] == 105.0
    assert dual["airi_literal_ms"] == 69.0
    assert sample_index_from_nominal_ms(105.0) == 605
    # sample 610 → 110 / 74
    dual = dual_times_from_sample(610)
    assert dual["nominal_ms"] == 110.0
    assert dual["airi_literal_ms"] == 74.0


def test_remap_does_not_move_stored_times():
    stored = {"t_start_ms": 105.0, "t_end_ms": 200.0, "peak_ms": 105.0}
    remapped = remap_interval(stored)
    assert remapped["t_start_ms"] == 105.0
    assert remapped["t_start_ms_nominal"] == 105.0
    assert remapped["t_start_ms_airi_literal"] == 69.0
    assert remapped["t_start_ms_sample_index"] == 605
    assert remapped["peak_ms_is_waveform_peak"] is False


def test_nominal_full_epoch_is_minus_500_plus_index():
    axis = nominal_axis_ms(AIRI_LITERAL_N_SAMPLES)
    assert axis[0] == -500.0
    assert axis[-1] == 1000.0
    assert np.allclose(axis, -500.0 + np.arange(1501))


def test_paper_anchors_are_reported_on_both_interpretations():
    anchors = paper_anchors_on_both_axes()
    face65 = anchors["face_c1_first_onset"]
    assert face65["printed_ms"] == 65.0
    # If the printed 65 ms used the AIRI plotting axis, the same sample
    # is 101 ms on the nominal axis.
    assert face65["if_printed_is_airi_literal"]["airi_literal_ms"] == 65.0
    assert face65["if_printed_is_airi_literal"]["nominal_ms"] == 101.0
