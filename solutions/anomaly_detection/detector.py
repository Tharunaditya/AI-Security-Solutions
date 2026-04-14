"""
Anomaly Detector
================
Detects anomalous data points in numeric feature vectors using
scikit-learn's Isolation Forest algorithm.

An Isolation Forest isolates anomalies by randomly partitioning data;
anomalous points require fewer partitions to isolate and receive lower
(more negative) anomaly scores.

Usage
-----
    from solutions.anomaly_detection import AnomalyDetector

    detector = AnomalyDetector(contamination=0.05)
    detector.fit(X_train)

    results = detector.predict(X_test)
    # [{'is_anomaly': False, 'anomaly_score': 0.12}, ...]

    # Or score a single sample
    result = detector.predict_one(sample)
    # {'is_anomaly': True, 'anomaly_score': -0.34}
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Isolation-Forest-based anomaly detector.

    Parameters
    ----------
    contamination : float or 'auto'
        Expected proportion of anomalies in the training data.
        Use ``'auto'`` to let the algorithm decide (default ``0.05``).
    n_estimators : int
        Number of trees in the forest (default 100).
    random_state : int
        Seed for reproducibility (default 42).
    """

    def __init__(
        self,
        contamination: Union[float, str] = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self._model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: Union[List[List[float]], np.ndarray]) -> "AnomalyDetector":
        """Fit the Isolation Forest on training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data – assumed to consist predominantly of *normal*
            (non-anomalous) samples.

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_scaled = self._scaler.fit_transform(X_arr)
        self._model.fit(X_scaled)
        self._fitted = True
        self._n_features = X_arr.shape[1]
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_one(
        self, sample: Union[List[float], float, np.ndarray]
    ) -> Dict[str, object]:
        """Score a single data point.

        Parameters
        ----------
        sample : array-like, shape (n_features,)

        Returns
        -------
        dict
            ``{'is_anomaly': bool, 'anomaly_score': float}``

            *anomaly_score* is the raw decision-function value.  More
            negative values indicate higher anomaly likelihood.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted yet.
        """
        self._check_fitted()
        arr = np.asarray(sample, dtype=float).reshape(1, -1)
        arr = self._scaler.transform(arr)

        score = float(self._model.decision_function(arr)[0])
        label = int(self._model.predict(arr)[0])   # -1=anomaly, 1=normal
        return {
            "is_anomaly": label == -1,
            "anomaly_score": round(score, 6),
        }

    def predict(
        self, X: Union[List[List[float]], np.ndarray]
    ) -> List[Dict[str, object]]:
        """Score multiple data points.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        list of dict
            One result dict per sample (same format as :meth:`predict_one`).
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_scaled = self._scaler.transform(X_arr)
        scores = self._model.decision_function(X_scaled)
        labels = self._model.predict(X_scaled)   # -1=anomaly, 1=normal

        return [
            {"is_anomaly": int(label) == -1, "anomaly_score": round(float(score), 6)}
            for label, score in zip(labels, scores)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

    @property
    def is_fitted(self) -> bool:
        """Return ``True`` if the detector has been fitted."""
        return self._fitted
