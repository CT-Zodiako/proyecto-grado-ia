"""Unit tests for app/dashboard/similar_programs.py.

Plain pytest, no Streamlit imports, synthetic fixtures — mirrors
tests/test_diagnostics.py's style. Tests the pure profile/clustering/
selection logic independently of the real 373-row dataset (except for one
empirical end-to-end test using a synthetic 73-program set).
"""
import numpy as np
import pytest

from app.dashboard.similar_programs import (
    MAX_SIMILAR_PROGRAMS,
    MIN_SAME_CLUSTER_PEERS,
    N_CLUSTERS,
    RANDOM_STATE,
    TREND_DOWN_THRESHOLD,
    TREND_UP_THRESHOLD,
    ProgramProfile,
    build_profile_matrix,
    build_similar_programs_index,
    compute_program_profile,
    fit_clusters,
    fit_scaler,
    select_similar_programs,
    standardize,
)


# ---------------------------------------------------------------------------
# compute_program_profile
# ---------------------------------------------------------------------------

def test_compute_program_profile_multi_year_exact_values():
    key = {"id_institucion": 1, "id_programa_acad": 1}
    profile = compute_program_profile(key, [150.0, 152.0, 148.0, 156.0])
    assert profile["avg"] == pytest.approx(151.5)
    assert profile["trend"] == pytest.approx(1.4, abs=0.01)
    # Population std (ddof=0) of [150,152,148,156] around mean 151.5:
    # deviations [-1.5, 0.5, -3.5, 4.5] -> mean squared 8.75 -> sqrt = 2.958.
    assert profile["volatility"] == pytest.approx(2.958, abs=0.01)
    assert profile["n_years"] == 4
    assert profile["key"] == key


def test_compute_program_profile_single_year_no_crash():
    profile = compute_program_profile({"id_institucion": 1, "id_programa_acad": 1}, [160.0])
    assert profile["avg"] == 160.0
    assert profile["trend"] == 0.0
    assert profile["volatility"] == 0.0
    assert profile["n_years"] == 1


def test_compute_program_profile_empty_raises():
    with pytest.raises(ValueError):
        compute_program_profile({"id_institucion": 1, "id_programa_acad": 1}, [])


# ---------------------------------------------------------------------------
# build_profile_matrix / fit_scaler / standardize
# ---------------------------------------------------------------------------

def _synthetic_profiles(n=15):
    profiles = []
    rng = np.random.default_rng(42)
    for i in range(n):
        profiles.append(
            ProgramProfile(
                key={"id_institucion": i, "id_programa_acad": i},
                avg=float(150 + rng.normal(0, 10)),
                trend=float(rng.normal(0, 2)),
                volatility=float(abs(rng.normal(3, 1))),
                n_years=5,
            )
        )
    return profiles


def test_standardize_mean_zero_std_one():
    profiles = _synthetic_profiles()
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    standardized = standardize(matrix, scaler)
    assert np.allclose(standardized.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(standardized.std(axis=0), 1, atol=1e-8)


def test_standardize_reuses_fitted_scaler_no_hidden_refit():
    profiles = _synthetic_profiles()
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    first = standardize(matrix, scaler)
    second = standardize(matrix, scaler)
    assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# fit_clusters
# ---------------------------------------------------------------------------

def test_fit_clusters_deterministic():
    profiles = _synthetic_profiles(n=30)
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    standardized = standardize(matrix, scaler)
    labels1 = fit_clusters(standardized)
    labels2 = fit_clusters(standardized)
    assert np.array_equal(labels1, labels2)
    assert len(labels1) == 30


# ---------------------------------------------------------------------------
# select_similar_programs — 3 branches (spec FR-4)
# ---------------------------------------------------------------------------

def _profiles_and_matrix(n):
    profiles = _synthetic_profiles(n)
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    standardized = standardize(matrix, scaler)
    return profiles, standardized


def test_select_similar_programs_sufficient_same_cluster_peers():
    profiles, standardized = _profiles_and_matrix(10)
    # Program 0 shares cluster 0 with programs 1-6 (6 peers, >=3); rest in cluster 1.
    cluster_labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    result = select_similar_programs(0, profiles, standardized, cluster_labels)
    assert len(result) == 5  # capped at MAX_SIMILAR_PROGRAMS
    for peer in result:
        assert peer["key"] != profiles[0]["key"]
        peer_index = next(i for i, p in enumerate(profiles) if p["key"] == peer["key"])
        assert cluster_labels[peer_index] == 0


def test_select_similar_programs_exactly_three_no_fallback():
    profiles, standardized = _profiles_and_matrix(10)
    # Program 0 has exactly 3 same-cluster peers (1,2,3); rest in other clusters.
    cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    result = select_similar_programs(0, profiles, standardized, cluster_labels)
    assert len(result) == 3
    result_keys = {tuple(p["key"].values()) for p in result}
    expected_keys = {tuple(profiles[i]["key"].values()) for i in (1, 2, 3)}
    assert result_keys == expected_keys


@pytest.mark.parametrize("n_same_cluster", [0, 1, 2])
def test_select_similar_programs_sparse_cluster_falls_back_to_five(n_same_cluster):
    profiles, standardized = _profiles_and_matrix(10)
    # Program 0 alone (or with 1-2 peers) in cluster 0; rest spread across other clusters.
    labels = [0] + [0] * n_same_cluster + list(range(1, 10 - n_same_cluster))
    cluster_labels = np.array(labels[:10])
    result = select_similar_programs(0, profiles, standardized, cluster_labels)
    assert len(result) == MAX_SIMILAR_PROGRAMS
    for peer in result:
        assert peer["key"] != profiles[0]["key"]


def test_select_similar_programs_never_includes_self():
    profiles, standardized = _profiles_and_matrix(20)
    cluster_labels = np.zeros(20, dtype=int)  # everyone in one cluster
    for idx in range(20):
        result = select_similar_programs(idx, profiles, standardized, cluster_labels)
        assert profiles[idx]["key"] not in [p["key"] for p in result]


# ---------------------------------------------------------------------------
# trend_label thresholds (spec FR-5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "trend,expected_label",
    [
        (0.51, "en alza"),
        (-0.51, "en baja"),
        (0.5, "estable"),
        (-0.5, "estable"),
        (0.0, "estable"),
    ],
)
def test_trend_label_thresholds(trend, expected_label):
    # Profile 0 carries the trend under test; profile 1 is the "selected"
    # program. All 4 share one cluster so profile 0 is always returned as a
    # peer (fallback branch guarantees inclusion regardless of distance).
    profiles = [
        ProgramProfile(key={"id_institucion": 0, "id_programa_acad": 0}, avg=150.0, trend=trend, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 1, "id_programa_acad": 1}, avg=150.0, trend=0.0, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 2, "id_programa_acad": 2}, avg=150.0, trend=0.0, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 3, "id_programa_acad": 3}, avg=150.0, trend=0.0, volatility=1.0, n_years=5),
    ]
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    standardized = standardize(matrix, scaler)
    cluster_labels = np.zeros(4, dtype=int)
    result = select_similar_programs(1, profiles, standardized, cluster_labels)
    dominant_peer = next(p for p in result if p["key"] == profiles[0]["key"])
    assert dominant_peer["trend_label"] == expected_label


def test_trend_up_down_threshold_constants():
    assert TREND_UP_THRESHOLD == 0.5
    assert TREND_DOWN_THRESHOLD == -0.5


# ---------------------------------------------------------------------------
# Empirical end-to-end test with a synthetic 73-program set
# ---------------------------------------------------------------------------

def test_build_similar_programs_index_end_to_end_73_programs():
    rng = np.random.default_rng(7)
    profiles = []
    for i in range(73):
        n_years = rng.integers(1, 7)
        series = list(150 + rng.normal(0, 15, size=n_years))
        profiles.append(compute_program_profile({"id_institucion": i, "id_programa_acad": i}, series))

    result_profiles, standardized, scaler, cluster_labels = build_similar_programs_index(profiles)
    assert len(result_profiles) == 73
    assert cluster_labels.shape == (73,)

    for idx in range(73):
        similares = select_similar_programs(idx, result_profiles, standardized, cluster_labels)
        assert MIN_SAME_CLUSTER_PEERS <= len(similares) <= MAX_SIMILAR_PROGRAMS or len(similares) == MAX_SIMILAR_PROGRAMS
        assert 3 <= len(similares) <= 5
        assert result_profiles[idx]["key"] not in [p["key"] for p in similares]


def test_n_clusters_and_random_state_constants():
    assert N_CLUSTERS == 12
    assert RANDOM_STATE == 42


# ---------------------------------------------------------------------------
# TRIANGULATE: ties and cross-cluster fallback
# ---------------------------------------------------------------------------

def test_select_similar_programs_handles_ties_and_excludes_self():
    # Programs 0 and 2 have IDENTICAL standardized profiles (a tie in
    # distance). Program 1 is selected; ensure no crash, self is excluded,
    # and the tie doesn't cause a duplicate or missing entry.
    profiles = [
        ProgramProfile(key={"id_institucion": i, "id_programa_acad": i}, avg=150.0, trend=0.0, volatility=1.0, n_years=5)
        for i in range(6)
    ]
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    standardized = standardize(matrix, scaler)
    cluster_labels = np.zeros(6, dtype=int)  # all identical -> all one cluster

    result = select_similar_programs(1, profiles, standardized, cluster_labels)
    assert len(result) == 5
    result_keys = [p["key"] for p in result]
    assert profiles[1]["key"] not in result_keys
    assert len(result_keys) == len(set(tuple(k.values()) for k in result_keys))


def test_select_similar_programs_fallback_ignores_cluster_boundary():
    # Program 0 is selected, alone in cluster 0 (fallback branch). Its
    # nearest neighbor by distance (program 1) is in a DIFFERENT cluster —
    # the fallback must still include it, proving cluster boundaries are
    # ignored in this branch.
    profiles = [
        ProgramProfile(key={"id_institucion": 0, "id_programa_acad": 0}, avg=150.0, trend=0.0, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 1, "id_programa_acad": 1}, avg=150.1, trend=0.0, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 2, "id_programa_acad": 2}, avg=170.0, trend=0.0, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 3, "id_programa_acad": 3}, avg=175.0, trend=0.0, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 4, "id_programa_acad": 4}, avg=180.0, trend=0.0, volatility=1.0, n_years=5),
        ProgramProfile(key={"id_institucion": 5, "id_programa_acad": 5}, avg=185.0, trend=0.0, volatility=1.0, n_years=5),
    ]
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    standardized = standardize(matrix, scaler)
    # Program 0 alone in cluster 0; program 1 (its nearest neighbor) is in cluster 1.
    cluster_labels = np.array([0, 1, 1, 1, 1, 1])

    result = select_similar_programs(0, profiles, standardized, cluster_labels)
    assert len(result) == 5
    result_keys = [p["key"] for p in result]
    assert profiles[1]["key"] in result_keys  # nearest neighbor included despite different cluster
