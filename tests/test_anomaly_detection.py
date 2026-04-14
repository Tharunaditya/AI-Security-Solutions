"""Tests for the Anomaly Detector."""

import numpy as np
import pytest
from solutions.anomaly_detection import AnomalyDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_data(n=200, n_features=5, seed=0):
    return np.random.default_rng(seed).normal(loc=0, scale=1, size=(n, n_features))


def _outlier_data(n=10, n_features=5, seed=1):
    return np.random.default_rng(seed).normal(loc=10, scale=0.5, size=(n, n_features))


@pytest.fixture
def fitted_detector():
    detector = AnomalyDetector(contamination=0.05, n_estimators=50, random_state=42)
    detector.fit(_normal_data())
    return detector


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestAnomalyDetectorLifecycle:
    def test_unfitted_predict_raises(self):
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="fit"):
            detector.predict([[0.0] * 5])

    def test_unfitted_predict_one_raises(self):
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="fit"):
            detector.predict_one([0.0] * 5)

    def test_is_fitted_false_before_fitting(self):
        assert AnomalyDetector().is_fitted is False

    def test_is_fitted_true_after_fitting(self, fitted_detector):
        assert fitted_detector.is_fitted is True

    def test_fit_returns_self(self):
        detector = AnomalyDetector(n_estimators=10)
        result = detector.fit(_normal_data(n=50))
        assert result is detector

    def test_fit_1d_input(self):
        """1-D input should be treated as a single-feature dataset."""
        detector = AnomalyDetector(n_estimators=10)
        detector.fit([1.0, 2.0, 1.5, 2.1, 1.8])
        result = detector.predict_one(1.9)
        assert "is_anomaly" in result


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestAnomalyDetectorPrediction:
    def test_predict_returns_list(self, fitted_detector):
        results = fitted_detector.predict(_normal_data(n=5))
        assert isinstance(results, list)
        assert len(results) == 5

    def test_predict_one_returns_dict(self, fitted_detector):
        result = fitted_detector.predict_one([0.0] * 5)
        assert isinstance(result, dict)

    def test_result_keys(self, fitted_detector):
        result = fitted_detector.predict_one([0.0] * 5)
        assert set(result.keys()) == {"is_anomaly", "anomaly_score"}

    def test_is_anomaly_is_bool(self, fitted_detector):
        result = fitted_detector.predict_one([0.0] * 5)
        assert isinstance(result["is_anomaly"], bool)

    def test_anomaly_score_is_float(self, fitted_detector):
        result = fitted_detector.predict_one([0.0] * 5)
        assert isinstance(result["anomaly_score"], float)

    def test_normal_sample_not_anomaly(self, fitted_detector):
        """Samples at the centre of the normal distribution should be normal."""
        normal_sample = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = fitted_detector.predict_one(normal_sample)
        assert result["is_anomaly"] is False

    def test_extreme_outlier_is_anomaly(self, fitted_detector):
        """Extreme outliers should be flagged as anomalous."""
        outlier = [100.0, 100.0, 100.0, 100.0, 100.0]
        result = fitted_detector.predict_one(outlier)
        assert result["is_anomaly"] is True

    def test_outlier_has_lower_score(self, fitted_detector):
        """Anomaly scores for outliers should be lower (more negative)."""
        normal_result = fitted_detector.predict_one([0.0] * 5)
        outlier_result = fitted_detector.predict_one([100.0] * 5)
        assert outlier_result["anomaly_score"] < normal_result["anomaly_score"]

    def test_batch_contains_known_anomalies(self, fitted_detector):
        """Outliers injected into a batch should be detected as anomalous."""
        normal = _normal_data(n=10)
        outliers = _outlier_data(n=3)
        X_test = np.vstack([normal, outliers])
        results = fitted_detector.predict(X_test)
        anomaly_flags = [r["is_anomaly"] for r in results]
        # At least one outlier should be detected
        assert any(anomaly_flags[10:])

    def test_predict_accepts_list_of_lists(self, fitted_detector):
        results = fitted_detector.predict([[0.1, 0.2, 0.1, 0.0, 0.3]] * 3)
        assert len(results) == 3
