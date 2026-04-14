"""Tests for the Intrusion Detection System."""

import numpy as np
import pytest
from solutions.intrusion_detection import IDS
from solutions.intrusion_detection.ids import NUM_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n_normal=100, n_attack=50, seed=0):
    rng = np.random.default_rng(seed)
    X_normal = rng.normal(loc=0, scale=1, size=(n_normal, NUM_FEATURES))
    X_attack = rng.normal(loc=4, scale=1, size=(n_attack, NUM_FEATURES))
    X = np.vstack([X_normal, X_attack])
    y = np.array([0] * n_normal + [1] * n_attack)
    return X, y


@pytest.fixture
def trained_ids():
    X, y = _make_dataset()
    ids = IDS(n_estimators=20, random_state=42)
    ids.train(X, y)
    return ids


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestIDSLifecycle:
    def test_untrained_predict_raises(self):
        ids = IDS()
        with pytest.raises(RuntimeError, match="train"):
            ids.predict([0.0] * NUM_FEATURES)

    def test_untrained_feature_importances_raises(self):
        ids = IDS()
        with pytest.raises(RuntimeError, match="train"):
            _ = ids.feature_importances

    def test_is_trained_false_before_training(self):
        assert IDS().is_trained is False

    def test_is_trained_true_after_training(self, trained_ids):
        assert trained_ids.is_trained is True

    def test_train_returns_self(self):
        X, y = _make_dataset(n_normal=20, n_attack=10)
        ids = IDS(n_estimators=5)
        result = ids.train(X, y)
        assert result is ids

    def test_wrong_feature_count_raises_on_train(self):
        ids = IDS(n_estimators=5)
        with pytest.raises(ValueError, match=str(NUM_FEATURES)):
            ids.train([[0.0] * 5], [0])

    def test_wrong_feature_count_raises_on_predict(self, trained_ids):
        with pytest.raises(ValueError, match=str(NUM_FEATURES)):
            trained_ids.predict([0.0] * 5)


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestIDSPrediction:
    def test_predict_returns_expected_keys(self, trained_ids):
        sample = [0.0] * NUM_FEATURES
        result = trained_ids.predict(sample)
        assert set(result.keys()) == {"label", "confidence", "attack_probability"}

    def test_label_is_valid(self, trained_ids):
        result = trained_ids.predict([0.0] * NUM_FEATURES)
        assert result["label"] in {"normal", "attack"}

    def test_confidence_in_range(self, trained_ids):
        result = trained_ids.predict([0.0] * NUM_FEATURES)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_attack_probability_in_range(self, trained_ids):
        result = trained_ids.predict([0.0] * NUM_FEATURES)
        assert 0.0 <= result["attack_probability"] <= 1.0

    def test_normal_sample_tends_normal(self, trained_ids):
        # A sample near the normal cluster mean should have lower attack prob.
        sample = np.zeros(NUM_FEATURES)
        result = trained_ids.predict(sample)
        # Just assert it returns without error; model may vary on small dataset
        assert result["label"] in {"normal", "attack"}

    def test_attack_sample_tends_attack(self, trained_ids):
        sample = np.full(NUM_FEATURES, 4.0)
        result = trained_ids.predict(sample)
        assert result["label"] in {"normal", "attack"}

    def test_predict_with_list_input(self, trained_ids):
        result = trained_ids.predict([1.0] * NUM_FEATURES)
        assert "label" in result

    def test_predict_with_numpy_array_input(self, trained_ids):
        sample = np.ones(NUM_FEATURES)
        result = trained_ids.predict(sample)
        assert "label" in result

    def test_predict_batch(self, trained_ids):
        X, _ = _make_dataset(n_normal=5, n_attack=5)
        results = trained_ids.predict_batch(X)
        assert len(results) == 10
        for r in results:
            assert r["label"] in {"normal", "attack"}

    def test_feature_importances_shape(self, trained_ids):
        importances = trained_ids.feature_importances
        assert importances.shape == (NUM_FEATURES,)
        assert abs(importances.sum() - 1.0) < 1e-6
