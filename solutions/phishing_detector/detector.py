"""
Phishing URL Detector
=====================
Classifies URLs as 'phishing' or 'legitimate' by extracting structural
and lexical features and running them through a Random Forest classifier.

Usage
-----
    from solutions.phishing_detector import PhishingDetector

    detector = PhishingDetector()
    detector.train(urls, labels)          # labels: 1 = phishing, 0 = legitimate
    result = detector.predict("http://example.com/login")
    # {'label': 'legitimate', 'confidence': 0.92, 'features': {...}}
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Suspicious keywords and TLDs commonly seen in phishing campaigns
# ---------------------------------------------------------------------------
_SUSPICIOUS_KEYWORDS: List[str] = [
    "login", "signin", "verify", "update", "secure", "account",
    "banking", "confirm", "password", "credential", "paypal",
    "ebay", "amazon", "apple", "microsoft", "support", "wallet",
]

_SUSPICIOUS_TLDS: set = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".club", ".work", ".date", ".review", ".stream",
}


def _shannon_entropy(text: str) -> float:
    """Return the Shannon entropy of *text*."""
    if not text:
        return 0.0
    freq = {c: text.count(c) / len(text) for c in set(text)}
    return -sum(p * math.log2(p) for p in freq.values())


def extract_features(url: str) -> Dict[str, float]:
    """Extract numerical security-relevant features from *url*.

    Returns a dictionary that maps feature name → value.  The dictionary
    can be converted to a 1-D numpy array via
    ``np.array(list(features.values()))``.
    """
    parsed = urlparse(url if "://" in url else "http://" + url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    full = url.lower()

    hostname = netloc.split(":")[0]

    # Basic length features
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)

    # Character-count features
    dot_count = full.count(".")
    hyphen_count = full.count("-")
    at_count = full.count("@")
    question_count = full.count("?")
    equals_count = full.count("=")
    ampersand_count = full.count("&")
    slash_count = full.count("/")
    digit_count = sum(c.isdigit() for c in url)
    digit_ratio = digit_count / max(url_length, 1)

    # Sub-domain depth
    subdomain_depth = hostname.count(".") - 1 if hostname.count(".") > 1 else 0

    # IP address in hostname?
    has_ip = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname)))

    # HTTPS?
    is_https = int(scheme == "https")

    # Suspicious TLD?
    suspicious_tld = int(any(hostname.endswith(tld) for tld in _SUSPICIOUS_TLDS))

    # Suspicious keywords in full URL?
    suspicious_keyword_count = sum(kw in full for kw in _SUSPICIOUS_KEYWORDS)

    # Entropy of hostname
    hostname_entropy = _shannon_entropy(hostname)

    # Presence of port number
    has_port = int(bool(parsed.port))

    # Long query string?
    query_length = len(query)

    # Hex-encoded characters (obfuscation indicator)
    hex_encoded_count = len(re.findall(r"%[0-9a-f]{2}", full))

    # Double-slash redirection
    double_slash = int("//" in path)

    features: Dict[str, float] = {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "at_count": at_count,
        "question_count": question_count,
        "equals_count": equals_count,
        "ampersand_count": ampersand_count,
        "slash_count": slash_count,
        "digit_ratio": digit_ratio,
        "subdomain_depth": subdomain_depth,
        "has_ip": has_ip,
        "is_https": is_https,
        "suspicious_tld": suspicious_tld,
        "suspicious_keyword_count": suspicious_keyword_count,
        "hostname_entropy": hostname_entropy,
        "has_port": has_port,
        "query_length": query_length,
        "hex_encoded_count": hex_encoded_count,
        "double_slash": double_slash,
    }
    return features


def _features_to_array(features: Dict[str, float]) -> np.ndarray:
    return np.array(list(features.values()), dtype=float).reshape(1, -1)


class PhishingDetector:
    """Random-Forest-based phishing URL classifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the Random Forest (default 100).
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

    def train(self, urls: List[str], labels: List[int]) -> "PhishingDetector":
        """Train the classifier.

        Parameters
        ----------
        urls : list of str
            Raw URL strings.
        labels : list of int
            Binary labels – ``1`` for phishing, ``0`` for legitimate.

        Returns
        -------
        self
        """
        X = np.vstack([_features_to_array(extract_features(u)) for u in urls])
        X = self._scaler.fit_transform(X)
        self._clf.fit(X, labels)
        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, url: str) -> Dict[str, object]:
        """Classify a single URL.

        Parameters
        ----------
        url : str
            The URL to classify.

        Returns
        -------
        dict
            ``{'label': str, 'confidence': float, 'features': dict}``

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        features = extract_features(url)
        X = _features_to_array(features)
        X = self._scaler.transform(X)

        proba = self._clf.predict_proba(X)[0]
        predicted_class = int(self._clf.predict(X)[0])
        confidence = float(proba[predicted_class])

        return {
            "label": "phishing" if predicted_class == 1 else "legitimate",
            "confidence": round(confidence, 4),
            "features": features,
        }

    def predict_batch(self, urls: List[str]) -> List[Dict[str, object]]:
        """Classify a list of URLs.

        Parameters
        ----------
        urls : list of str

        Returns
        -------
        list of dict
            One result dict per URL (same format as :meth:`predict`).
        """
        return [self.predict(url) for url in urls]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(url: str) -> Dict[str, float]:
        """Public wrapper around the module-level :func:`extract_features`."""
        return extract_features(url)

    @property
    def is_trained(self) -> bool:
        """Return ``True`` if the model has been trained."""
        return self._trained
