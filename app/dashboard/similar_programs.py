"""Pure clustering helpers for the "5. Programas similares" block of the
"Diagnóstico" dashboard page.

No Streamlit imports, no network calls. This module operates over the FULL
population of Medicina programs (unlike diagnostics.py, which explains one
selected program's own prediction) — clustering programs by historical
performance profile (average, trend, volatility), never by geography.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Fixed clustering configuration (locked decision) — module-level constants,
# not user-configurable, for determinism and discoverability.
N_CLUSTERS: int = 12
RANDOM_STATE: int = 42
MIN_SAME_CLUSTER_PEERS: int = 3
MAX_SIMILAR_PROGRAMS: int = 5

# Trend-direction thresholds for display labels.
TREND_UP_THRESHOLD: float = 0.5
TREND_DOWN_THRESHOLD: float = -0.5


class ProgramKey(TypedDict):
    """Identity for one program — mirrors the (ID_INSTITUCION,
    ID_PROGRAMA_ACAD) grouping key already used in streamlit_app.py's
    Diagnóstico page."""

    id_institucion: object
    id_programa_acad: object


class ProgramProfile(TypedDict):
    key: ProgramKey
    avg: float
    trend: float
    volatility: float
    n_years: int


class SimilarProgramResult(TypedDict):
    key: ProgramKey
    avg: float
    trend: float
    trend_label: str
    distance: float


def compute_program_profile(
    key: ProgramKey,
    yearly_scores: list[float],
) -> ProgramProfile:
    """Compute (avg, trend, volatility) for one program from its own
    promedio_global_anual series, ordered by AÑO ascending (caller's
    responsibility to pass it pre-sorted).

    - avg = mean(yearly_scores)
    - trend: n>=2 -> OLS slope via numpy.polyfit(x, yearly_scores, 1)[0]
             with x = 0..n-1; n==1 -> 0.0
    - volatility: n>=2 -> population std (ddof=0); n==1 -> 0.0

    Raises ValueError if yearly_scores is empty.
    """
    n = len(yearly_scores)
    if n == 0:
        raise ValueError("yearly_scores must not be empty")

    avg = float(np.mean(yearly_scores))

    if n >= 2:
        x = np.arange(n)
        trend = float(np.polyfit(x, yearly_scores, 1)[0])
        volatility = float(np.std(yearly_scores, ddof=0))
    else:
        trend = 0.0
        volatility = 0.0

    return ProgramProfile(key=key, avg=avg, trend=trend, volatility=volatility, n_years=n)


def build_profile_matrix(profiles: list[ProgramProfile]) -> np.ndarray:
    """Stack `profiles` into an (n_programs, 3) ndarray, columns ordered
    [avg, trend, volatility], preserving `profiles` list order."""
    return np.array([[p["avg"], p["trend"], p["volatility"]] for p in profiles])


def fit_scaler(matrix: np.ndarray) -> StandardScaler:
    """Fit and return a StandardScaler on `matrix` (mean 0, std 1 per
    column). Fit-once contract: callers must reuse this same fitted scaler
    for both clustering input and distance-fallback calculations."""
    scaler = StandardScaler()
    scaler.fit(matrix)
    return scaler


def standardize(matrix: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply an already-fitted `scaler` to `matrix`. Never fits here."""
    return scaler.transform(matrix)


def fit_clusters(standardized_matrix: np.ndarray) -> np.ndarray:
    """Run KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE) on
    `standardized_matrix` and return the fitted `.labels_` array."""
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(standardized_matrix)
    return kmeans.labels_


def _trend_label(trend: float) -> str:
    if trend > TREND_UP_THRESHOLD:
        return "en alza"
    if trend < TREND_DOWN_THRESHOLD:
        return "en baja"
    return "estable"


def select_similar_programs(
    selected_index: int,
    profiles: list[ProgramProfile],
    standardized_matrix: np.ndarray,
    cluster_labels: np.ndarray,
) -> list[SimilarProgramResult]:
    """Implements the 3-branch selection logic for the program at
    `profiles[selected_index]`:

    1. same_cluster_indices = all i != selected_index where
       cluster_labels[i] == cluster_labels[selected_index]
    2. if len(same_cluster_indices) >= MIN_SAME_CLUSTER_PEERS: rank by
       ascending Euclidean distance, return top min(MAX_SIMILAR_PROGRAMS,
       len(same_cluster_indices)).
    3. else: fall back to the 5 nearest programs across the ENTIRE set
       (ignoring cluster boundaries), excluding selected_index.

    selected_index is always excluded via explicit index comparison, never
    via distance == 0.
    """
    selected_vector = standardized_matrix[selected_index]
    selected_cluster = cluster_labels[selected_index]

    same_cluster_indices = [
        i
        for i in range(len(profiles))
        if i != selected_index and cluster_labels[i] == selected_cluster
    ]

    if len(same_cluster_indices) >= MIN_SAME_CLUSTER_PEERS:
        candidate_indices = same_cluster_indices
        limit = min(MAX_SIMILAR_PROGRAMS, len(same_cluster_indices))
    else:
        candidate_indices = [i for i in range(len(profiles)) if i != selected_index]
        limit = MAX_SIMILAR_PROGRAMS

    ranked = sorted(
        candidate_indices,
        key=lambda i: float(np.linalg.norm(standardized_matrix[i] - selected_vector)),
    )
    top_indices = ranked[:limit]

    results: list[SimilarProgramResult] = []
    for i in top_indices:
        distance = float(np.linalg.norm(standardized_matrix[i] - selected_vector))
        results.append(
            SimilarProgramResult(
                key=profiles[i]["key"],
                avg=profiles[i]["avg"],
                trend=profiles[i]["trend"],
                trend_label=_trend_label(profiles[i]["trend"]),
                distance=distance,
            )
        )
    return results


def build_similar_programs_index(
    profiles: list[ProgramProfile],
) -> tuple[list[ProgramProfile], np.ndarray, StandardScaler, np.ndarray]:
    """Orchestrator: build_profile_matrix -> fit_scaler -> standardize ->
    fit_clusters. Returns (profiles, standardized_matrix, scaler,
    cluster_labels) as one bundle."""
    matrix = build_profile_matrix(profiles)
    scaler = fit_scaler(matrix)
    standardized = standardize(matrix, scaler)
    cluster_labels = fit_clusters(standardized)
    return profiles, standardized, scaler, cluster_labels
