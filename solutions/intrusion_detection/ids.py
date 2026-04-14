"""
Intrusion Detection System (IDS)
=================================
Classifies network connections as 'normal' or 'attack' using a
Random Forest classifier trained on network-traffic feature vectors.

Expected Feature Vector (19 numeric features)
----------------------------------------------
0.  duration          – connection duration (seconds)
1.  protocol_type     – encoded protocol (0=tcp, 1=udp, 2=icmp)
2.  service           – encoded service category (0–9)
3.  flag              – encoded connection flag (0–10)
4.  src_bytes         – bytes from source to destination
5.  dst_bytes         – bytes from destination to source
6.  land              – 1 if connection is from/to same host/port
7.  wrong_fragment    – number of wrong fragments
8.  urgent            – number of urgent packets
9.  hot               – number of "hot" indicators
10. num_failed_logins – number of failed login attempts
11. logged_in         – 1 if logged in
12. num_compromised   – number of compromised conditions
13. root_shell        – 1 if root shell obtained
14. su_attempted      – 1 if su root attempted
15. num_root          – number of root accesses
16. num_file_creations– number of file-creation operations
17. count             – connections to same host in past 2 seconds
18. srv_count         – connections to same service in past 2 seconds

Usage
-----
    from solutions.intrusion_detection import IDS

    ids = IDS()
    ids.train(X_train, y_train)   # y: 1=attack, 0=normal
    result = ids.predict(feature_vector)
    # {'label': 'normal', 'confidence': 0.97, 'attack_probability': 0.03}
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Number of features the IDS expects
NUM_FEATURES = 19


class IDS:
    """Random-Forest network intrusion detector.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default 100).
    random_state : int
        Seed for reproducibility (default 42).
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        self._scaler = StandardScaler()
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: Union[List[List[float]], np.ndarray],
        y: Union[List[int], np.ndarray],
    ) -> "IDS":
        """Train the classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, NUM_FEATURES)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Binary labels – ``1`` for attack, ``0`` for normal.

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        if X_arr.ndim != 2 or X_arr.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"X must have shape (n_samples, {NUM_FEATURES}), "
                f"got {X_arr.shape}"
            )

        X_scaled = self._scaler.fit_transform(X_arr)
        self._clf.fit(X_scaled, y_arr)
        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, features: Union[List[float], np.ndarray]
    ) -> Dict[str, object]:
        """Classify a single connection.

        Parameters
        ----------
        features : array-like, shape (NUM_FEATURES,) or (1, NUM_FEATURES)

        Returns
        -------
        dict
            ``{'label': str, 'confidence': float, 'attack_probability': float}``

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        ValueError
            If the feature vector has the wrong length.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        arr = np.asarray(features, dtype=float).reshape(1, -1)

        if arr.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected {NUM_FEATURES} features, got {arr.shape[1]}."
            )

        arr_scaled = self._scaler.transform(arr)
        proba = self._clf.predict_proba(arr_scaled)[0]
        predicted_class = int(self._clf.predict(arr_scaled)[0])

        # Probability order follows clf.classes_
        classes = list(self._clf.classes_)
        attack_prob = float(proba[classes.index(1)]) if 1 in classes else 0.0
        confidence = float(proba[predicted_class])

        return {
            "label": "attack" if predicted_class == 1 else "normal",
            "confidence": round(confidence, 4),
            "attack_probability": round(attack_prob, 4),
        }

    def predict_batch(
        self, X: Union[List[List[float]], np.ndarray]
    ) -> List[Dict[str, object]]:
        """Classify multiple connections.

        Parameters
        ----------
        X : array-like, shape (n_samples, NUM_FEATURES)

        Returns
        -------
        list of dict
        """
        return [self.predict(row) for row in np.asarray(X, dtype=float)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Return ``True`` if the model has been trained."""
        return self._trained

    @property
    def feature_importances(self) -> np.ndarray:
        """Return the feature-importance array from the underlying forest.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._clf.feature_importances_
