"""Simulation unit tests that do not require the AD forward model."""

from __future__ import annotations

import numpy as np

from paper.reproduction.common.method import AIRI_SPOC_KWARGS, fit_redisca
from paper.reproduction.simulations.config import (
    CAUSAL_CROSS_CANDIDATES,
    EQ16_CANDIDATES,
    REVIEW2_SIM_CANDIDATES,
    REVIEW_ADDED_SIM_CANDIDATES,
    SimulationConfig,
    config_for_candidate,
)
from paper.reproduction.simulations.forward_model import ForwardModel
from paper.reproduction.simulations.generate import simulate_multi_source
from paper.reproduction.simulations.generate import (
    add_symmetric_rdm_noise,
    cortical_one_over_f_sensor_noise,
    mix_signal_and_noise,
    squared_euclidean_rdm,
    topography_with_forward_error,
)
from paper.reproduction.simulations.metrics_roc import cosine_abs_scan, roc_from_mc


def test_squared_euclidean_rdm_properties():
    series = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    rdm = squared_euclidean_rdm(series)
    assert rdm.shape == (3, 3)
    assert np.allclose(np.diag(rdm), 0.0)
    assert np.allclose(rdm, rdm.T)
    assert rdm[0, 1] == 2.0


def test_rdm_noise_is_symmetric_nonneg():
    rng = np.random.default_rng(0)
    d0 = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    noisy = add_symmetric_rdm_noise(d0, rng, relative_std=0.05)
    assert np.allclose(noisy, noisy.T)
    assert np.allclose(np.diag(noisy), 0.0)
    assert np.all(noisy >= 0.0)


def test_roc_and_cosine_helpers():
    scores = np.array([[1.0, 0.2, 0.0], [0.9, 0.1, 0.0]])
    inside = np.array([[True, False, False], [True, False, False]])
    roc = roc_from_mc(scores, inside, np.array([0.5, 0.0]))
    assert roc["auc"] >= 0.0
    gain = np.eye(3)
    scan = cosine_abs_scan(np.array([1.0, 0.0, 0.0]), gain)
    assert scan[0] == 1.0


def test_redisca_factory_on_simulated_averages():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 8, 40))
    y = np.ones((5, 5))
    np.fill_diagonal(y, 0.0)
    y[0, 1] = y[1, 0] = 0.1
    model = fit_redisca(X, y)
    assert model.n_conditions_ == 5
    assert model.filters_.shape[1] == 8


def test_literal_delta_matches_printed_covariance_scale():
    rng = np.random.default_rng(0)
    g = np.ones(16)
    _topo, delta = topography_with_forward_error(
        g, rng, sigma_rel=0.15, delta_mode="literal_covariance"
    )
    # ||δ|| is typically ~0.15*sqrt(N)*||g||, not 0.15||g||.
    expected_rms_norm = 0.15 * np.linalg.norm(g) * np.sqrt(g.size)
    assert np.linalg.norm(delta) > 0.5 * expected_rms_norm


def test_norm_15pct_delta_has_requested_length():
    rng = np.random.default_rng(1)
    g = np.linspace(-1.0, 1.0, 20)
    _topo, delta = topography_with_forward_error(
        g, rng, sigma_rel=0.15, delta_mode="norm_15pct"
    )
    assert np.isclose(np.linalg.norm(delta), 0.15 * np.linalg.norm(g))


def test_global_gamma_is_one_scalar():
    rng = np.random.default_rng(2)
    signal = rng.standard_normal((3, 4, 10))
    noise = rng.standard_normal((5, 3, 4, 10))
    _trials, gammas = mix_signal_and_noise(signal, noise, 0.2, gamma_mode="global")
    assert gammas.shape == (5,)
    assert np.allclose(gammas, gammas[0])


def test_per_trial_gamma_varies():
    rng = np.random.default_rng(3)
    signal = rng.standard_normal((3, 4, 10))
    noise = rng.standard_normal((6, 3, 4, 10))
    noise[0] *= 3.0
    _trials, gammas = mix_signal_and_noise(signal, noise, 0.2, gamma_mode="per_trial")
    assert gammas.shape == (6,)
    assert np.std(gammas) > 0.0


def test_review_candidates_are_labeled_and_do_not_change_p1_defaults():
    p1 = config_for_candidate("SIM-P1")
    assert p1.i_c == 40
    assert p1.delta_mode == "literal_covariance"
    assert p1.snr_gamma_mode == "per_trial"
    assert p1.noise_loci_mode == "per_epoch"
    assert p1.fig5_generate_c == 6
    p4 = config_for_candidate("SIM-P4")
    assert p4.i_c == 100
    r1 = config_for_candidate("SIM-R1")
    assert r1.delta_mode == "norm_15pct"
    assert r1.snr_gamma_mode == "global"
    assert r1.noise_loci_mode == "fixed"
    assert r1.i_c == 100
    assert "SIM-P4" in REVIEW_ADDED_SIM_CANDIDATES
    assert "SIM-P1" not in REVIEW_ADDED_SIM_CANDIDATES
    assert "SIM-C1" in REVIEW_ADDED_SIM_CANDIDATES
    assert "EQ16-CAUSAL" in REVIEW2_SIM_CANDIDATES


def test_literal_delta_rng_matches_original_formula():
    g = np.linspace(-2.0, 1.0, 12)
    rng_old = np.random.default_rng(99)
    rng_new = np.random.default_rng(99)
    sigma = 0.15 * float(np.linalg.norm(g))
    delta_old = rng_old.standard_normal(g.shape) * sigma
    _topo, delta_new = topography_with_forward_error(
        g, rng_new, sigma_rel=0.15, delta_mode="literal_covariance"
    )
    assert np.allclose(delta_old, delta_new)


def test_fixed_noise_loci_keep_the_same_active_channels():
    gain = np.eye(20)
    noise = cortical_one_over_f_sensor_noise(
        gain,
        np.random.default_rng(0),
        n_epochs=6,
        n_times=16,
        n_sources=3,
        fs_hz=1000.0,
        exponent=1.0,
        loci_mode="fixed",
    )
    energy = np.sqrt(np.mean(noise * noise, axis=2))
    active = energy > 1e-8
    assert np.all(active == active[0])


def test_per_epoch_noise_loci_can_change_active_channels():
    gain = np.eye(30)
    noise = cortical_one_over_f_sensor_noise(
        gain,
        np.random.default_rng(0),
        n_epochs=12,
        n_times=16,
        n_sources=2,
        fs_hz=1000.0,
        exponent=1.0,
        loci_mode="per_epoch",
    )
    energy = np.sqrt(np.mean(noise * noise, axis=2))
    active = energy > 1e-8
    assert not np.all(active == active[0])


def test_uninformative_target_is_labeled_not_a_library_patch():
    from paper.reproduction.simulations.run import UninformativeTargetRDM, _fit_first_component

    X = np.random.default_rng(0).standard_normal((4, 6, 20))
    y = np.ones((4, 4))
    np.fill_diagonal(y, 0.0)
    try:
        _fit_first_component(X, y)
    except UninformativeTargetRDM:
        return
    raise AssertionError("constant off-diagonal RDM must be labeled uninformative")


def test_unknown_candidate_is_rejected():
    try:
        config_for_candidate("SIM-P9")
    except ValueError as exc:
        assert "SIM-P9" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_causal_cross_candidates_inherit_sim_p3_and_only_declared_knobs():
    p3 = config_for_candidate("SIM-P3")
    assert p3.butter_zero_phase is False
    expected = {
        "SIM-C1": {"i_c": 100},
        "SIM-C2": {"delta_mode": "norm_15pct"},
        "SIM-C3": {"snr_gamma_mode": "global"},
        "SIM-C4": {"noise_loci_mode": "fixed"},
        "SIM-CR1": {
            "i_c": 100,
            "delta_mode": "norm_15pct",
            "snr_gamma_mode": "global",
            "noise_loci_mode": "fixed",
        },
    }
    compare_keys = (
        "i_c",
        "delta_mode",
        "snr_gamma_mode",
        "noise_loci_mode",
        "butter_zero_phase",
        "eq16_single_matrix",
        "fig5_generate_c",
        "n_times",
        "n_noise_sources",
        "sigma_delta_rel",
    )
    for cid in CAUSAL_CROSS_CANDIDATES:
        cfg = config_for_candidate(cid)
        assert cfg.butter_zero_phase is False
        assert cfg.eq16_single_matrix is False
        for key in compare_keys:
            if key in expected[cid] or key == "butter_zero_phase":
                continue
            assert getattr(cfg, key) == getattr(p3, key), (cid, key)
        for key, value in expected[cid].items():
            assert getattr(cfg, key) == value
    assert AIRI_SPOC_KWARGS["solver"] == "whitening"
    assert AIRI_SPOC_KWARGS["directed_pairs"] is True
    assert AIRI_SPOC_KWARGS["aggregation"] == "mean"


def test_eq16_candidates_are_causal_multi_source_only():
    for cid in EQ16_CANDIDATES:
        cfg = config_for_candidate(cid)
        assert cfg.butter_zero_phase is False
        assert cfg.eq16_single_matrix is True
        assert cfg.fig5_generate_c == 6
    assert config_for_candidate("EQ16-CAUSAL").delta_mode == "literal_covariance"
    assert config_for_candidate("EQ16-CAUSAL-D").delta_mode == "norm_15pct"


def _toy_forward(n_channels: int = 8, n_vertices: int = 12) -> ForwardModel:
    rng = np.random.default_rng(0)
    vertices = rng.normal(size=(n_vertices, 3))
    gain = rng.normal(size=(n_channels, n_vertices))
    return ForwardModel(
        gain=gain,
        vertices=vertices,
        normals=np.tile(np.array([0.0, 0.0, 1.0]), (n_vertices, 1)),
        channel_index=np.arange(n_channels),
        n_channels=n_channels,
        n_vertices=n_vertices,
        tess_path="toy",
        headmodel_path="toy",
        tess_sha256=None,
        headmodel_sha256=None,
        surface_file="toy",
        meg_method="toy",
        notes=("toy forward for unit tests",),
    )


def test_eq16_single_matrix_has_no_hidden_ic_averaging():
    forward = _toy_forward()
    mixings = np.random.default_rng(1).standard_normal((4, 6, 6))
    cfg = SimulationConfig(
        n_times=32,
        i_c=40,
        n_noise_sources=3,
        butter_zero_phase=False,
        eq16_single_matrix=True,
        n_mc=1,
    )
    draw = simulate_multi_source(
        forward,
        np.random.default_rng(2),
        mixings,
        config=cfg,
        snr=0.4,
        n_conditions=6,
    )
    assert draw.trials.shape[0] == 1
    assert draw.averages.shape == (6, forward.n_channels, 32)
    assert np.allclose(draw.averages, draw.trials[0])
    assert draw.trials.shape[0] != cfg.i_c
    cfg_other_ic = SimulationConfig(
        n_times=32,
        i_c=100,
        n_noise_sources=3,
        butter_zero_phase=False,
        eq16_single_matrix=True,
        n_mc=1,
    )
    draw_other = simulate_multi_source(
        forward,
        np.random.default_rng(2),
        mixings,
        config=cfg_other_ic,
        snr=0.4,
        n_conditions=6,
    )
    assert np.allclose(draw.averages, draw_other.averages)
    assert np.allclose(draw.trials, draw_other.trials)


def test_trial_average_path_still_uses_ic():
    forward = _toy_forward()
    mixings = np.random.default_rng(1).standard_normal((4, 6, 6))
    cfg = SimulationConfig(
        n_times=16,
        i_c=5,
        n_noise_sources=2,
        butter_zero_phase=False,
        eq16_single_matrix=False,
        n_mc=1,
    )
    draw = simulate_multi_source(
        forward,
        np.random.default_rng(3),
        mixings,
        config=cfg,
        snr=0.4,
        n_conditions=6,
    )
    assert draw.trials.shape[0] == 5
    assert np.allclose(draw.averages, draw.trials.mean(axis=0))
