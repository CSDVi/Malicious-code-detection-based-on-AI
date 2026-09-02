"""Compatibility helpers for supported scikit-learn calibration APIs."""

from __future__ import annotations

from typing import Any


def calibrate_prefit(
    estimator: Any,
    x_validation: Any,
    y_validation: Any,
    *,
    method: str = "sigmoid",
) -> Any:
    """Calibrate an already-fitted estimator on a disjoint validation split.

    scikit-learn 1.8 removed ``cv='prefit'`` in favour of
    ``FrozenEstimator``.  Existing model-building environments may still use
    the older API, so choose the supported form at runtime.
    """

    from sklearn.calibration import CalibratedClassifierCV

    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # pragma: no cover - exercised by older sklearn envs
        calibrated = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    else:
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(estimator),
            method=method,
        )
    calibrated.fit(x_validation, y_validation)
    return calibrated
