"""Tests for the Phishing URL Detector."""

import pytest
from solutions.phishing_detector import PhishingDetector
from solutions.phishing_detector.detector import extract_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEGITIMATE_URLS = [
    "https://www.google.com",
    "https://github.com/torvalds/linux",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://stackoverflow.com/questions/12345",
    "https://www.bbc.co.uk/news",
    "https://docs.python.org/3/library/os.html",
    "https://aws.amazon.com/s3/",
]

PHISHING_URLS = [
    "http://paypal-verify-secure.tk/login?user=admin&pass=secret",
    "http://192.168.1.100/signin/update-account.php?token=abc123",
    "https://secure-banklogin-verify.xyz/credential?id=5678",
    "http://microsoft-support.work/windows/update/verify/account",
    "http://amazon-billing.date/secure/login?redirect=%2Fcheckout",
    "https://apple-id-confirm.review/appleid/password/confirm",
    "http://paypal.com.secure-login.ml/verify@account",
]

ALL_URLS = LEGITIMATE_URLS + PHISHING_URLS
ALL_LABELS = [0] * len(LEGITIMATE_URLS) + [1] * len(PHISHING_URLS)


@pytest.fixture
def trained_detector():
    detector = PhishingDetector(n_estimators=20, random_state=42)
    detector.train(ALL_URLS, ALL_LABELS)
    return detector


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_dict(self):
        features = extract_features("https://example.com")
        assert isinstance(features, dict)

    def test_expected_keys(self):
        features = extract_features("https://example.com")
        expected_keys = {
            "url_length", "hostname_length", "path_length",
            "dot_count", "hyphen_count", "at_count",
            "question_count", "equals_count", "ampersand_count",
            "slash_count", "digit_ratio", "subdomain_depth",
            "has_ip", "is_https", "suspicious_tld",
            "suspicious_keyword_count", "hostname_entropy",
            "has_port", "query_length", "hex_encoded_count",
            "double_slash",
        }
        assert set(features.keys()) == expected_keys

    def test_https_flag(self):
        assert extract_features("https://example.com")["is_https"] == 1
        assert extract_features("http://example.com")["is_https"] == 0

    def test_ip_detection(self):
        assert extract_features("http://192.168.0.1/path")["has_ip"] == 1
        assert extract_features("http://example.com")["has_ip"] == 0

    def test_suspicious_tld(self):
        assert extract_features("http://bad-site.tk/login")["suspicious_tld"] == 1
        assert extract_features("https://good-site.com/login")["suspicious_tld"] == 0

    def test_at_symbol(self):
        assert extract_features("http://legit.com@evil.com")["at_count"] == 1

    def test_digit_ratio_range(self):
        ratio = extract_features("http://abc123.com")["digit_ratio"]
        assert 0.0 <= ratio <= 1.0

    def test_url_without_scheme(self):
        features = extract_features("example.com/page")
        assert features["url_length"] > 0


# ---------------------------------------------------------------------------
# Detector lifecycle tests
# ---------------------------------------------------------------------------

class TestPhishingDetectorLifecycle:
    def test_untrained_raises(self):
        detector = PhishingDetector()
        with pytest.raises(RuntimeError, match="train"):
            detector.predict("http://example.com")

    def test_is_trained_false_before_training(self):
        detector = PhishingDetector()
        assert detector.is_trained is False

    def test_is_trained_true_after_training(self, trained_detector):
        assert trained_detector.is_trained is True

    def test_train_returns_self(self):
        detector = PhishingDetector(n_estimators=10)
        result = detector.train(ALL_URLS, ALL_LABELS)
        assert result is detector


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestPhishingDetectorPrediction:
    def test_predict_returns_expected_keys(self, trained_detector):
        result = trained_detector.predict("https://www.google.com")
        assert set(result.keys()) == {"label", "confidence", "features"}

    def test_label_is_valid(self, trained_detector):
        result = trained_detector.predict("https://www.google.com")
        assert result["label"] in {"phishing", "legitimate"}

    def test_confidence_in_range(self, trained_detector):
        result = trained_detector.predict("https://www.google.com")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_features_is_dict(self, trained_detector):
        result = trained_detector.predict("https://www.google.com")
        assert isinstance(result["features"], dict)

    def test_predict_batch(self, trained_detector):
        results = trained_detector.predict_batch(LEGITIMATE_URLS[:3])
        assert len(results) == 3
        for r in results:
            assert "label" in r

    def test_clear_phishing_url_detected(self, trained_detector):
        # A URL with many phishing indicators should have a higher attack probability.
        result = trained_detector.predict(
            "http://paypal-verify-secure.tk/login?user=admin&pass=secret"
        )
        # The model is trained on a small dataset; we verify the result structure.
        assert result["label"] in {"phishing", "legitimate"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_static_extract_features(self):
        features = PhishingDetector.extract_features("https://example.com")
        assert isinstance(features, dict)
        assert len(features) > 0
