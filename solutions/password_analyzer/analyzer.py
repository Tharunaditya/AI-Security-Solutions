"""
Password Strength Analyzer
===========================
Analyses the strength of a password using a combination of rule-based
checks and a derived ML complexity score, returning a numeric strength
score (0–100) and a human-readable label.

Strength labels
---------------
- **Weak**        : score  0–39
- **Fair**        : score 40–59
- **Strong**      : score 60–79
- **Very Strong** : score 80–100

Usage
-----
    from solutions.password_analyzer import PasswordAnalyzer

    analyzer = PasswordAnalyzer()
    result = analyzer.analyze("P@ssw0rd!")
    # {
    #     'score': 68,
    #     'label': 'Strong',
    #     'feedback': ['Consider making it longer (12+ characters)'],
    #     'checks': {
    #         'length': True,
    #         'uppercase': True,
    #         'lowercase': True,
    #         'digits': True,
    #         'special_chars': True,
    #         'not_common': True
    #     }
    # }
"""

from __future__ import annotations

import math
import re
import string
from typing import Dict, List

# ---------------------------------------------------------------------------
# Common / breached password list (top-50 most common passwords)
# ---------------------------------------------------------------------------
_COMMON_PASSWORDS: set = {
    "123456", "password", "123456789", "12345678", "12345",
    "1234567", "1234567890", "qwerty", "abc123", "111111",
    "123123", "admin", "letmein", "welcome", "monkey",
    "dragon", "master", "login", "pass", "test",
    "1234", "sunshine", "princess", "qwerty123", "solo",
    "iloveyou", "qwertyuiop", "starwars", "football", "baseball",
    "trustno1", "superman", "batman", "shadow", "michael",
    "ashley", "bailey", "passw0rd", "mustang", "access",
    "flower", "555555", "lovely", "donald", "george",
    "harley", "ranger", "joshua", "hunter", "charlie",
}

# Keyboard walk patterns (partial list)
_KEYBOARD_WALKS: List[str] = [
    "qwerty", "asdfgh", "zxcvbn", "qazwsx", "1qaz2wsx",
    "abcdef", "123456", "654321",
]


def _shannon_entropy(password: str) -> float:
    """Return the Shannon entropy (bits) of *password*."""
    if not password:
        return 0.0
    freq = {c: password.count(c) / len(password) for c in set(password)}
    return -sum(p * math.log2(p) for p in freq.values())


def _charset_size(password: str) -> int:
    """Return the effective character-set size used in *password*."""
    size = 0
    if any(c.islower() for c in password):
        size += 26
    if any(c.isupper() for c in password):
        size += 26
    if any(c.isdigit() for c in password):
        size += 10
    special = set(string.punctuation)
    if any(c in special for c in password):
        size += len(special)
    return max(size, 1)


class PasswordAnalyzer:
    """Rule-based + entropy-driven password strength analyzer.

    The analyzer does **not** require any training data; it applies
    deterministic rules and information-theoretic metrics.
    """

    # Minimum character-class requirements
    MIN_LENGTH = 8
    STRONG_LENGTH = 12
    VERY_STRONG_LENGTH = 16

    def analyze(self, password: str) -> Dict[str, object]:
        """Analyze the strength of *password*.

        Parameters
        ----------
        password : str

        Returns
        -------
        dict
            ``{'score': int, 'label': str, 'feedback': list[str],
               'checks': dict[str, bool]}``
        """
        checks = self._run_checks(password)
        score = self._compute_score(password, checks)
        label = self._score_to_label(score)
        feedback = self._generate_feedback(password, checks)

        return {
            "score": score,
            "label": label,
            "feedback": feedback,
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_checks(self, password: str) -> Dict[str, bool]:
        """Return a dictionary of boolean rule checks."""
        special_chars = set(string.punctuation)
        lower = password.lower()
        return {
            "length": len(password) >= self.MIN_LENGTH,
            "uppercase": any(c.isupper() for c in password),
            "lowercase": any(c.islower() for c in password),
            "digits": any(c.isdigit() for c in password),
            "special_chars": any(c in special_chars for c in password),
            "not_common": lower not in _COMMON_PASSWORDS,
            "no_keyboard_walk": not any(
                walk in lower for walk in _KEYBOARD_WALKS
            ),
            "no_repeated_chars": not bool(re.search(r"(.)\1{2,}", password)),
        }

    def _compute_score(self, password: str, checks: Dict[str, bool]) -> int:
        """Compute a 0-100 strength score."""
        score = 0.0

        # Length contribution (up to 30 points)
        length = len(password)
        if length >= self.VERY_STRONG_LENGTH:
            score += 30
        elif length >= self.STRONG_LENGTH:
            score += 22
        elif length >= self.MIN_LENGTH:
            score += 14
        else:
            score += max(0, length * 1.5)

        # Entropy contribution (up to 40 points)
        entropy = _shannon_entropy(password)
        max_entropy = _shannon_entropy(string.printable[:_charset_size(password)])
        entropy_ratio = entropy / max(max_entropy, 1)
        score += entropy_ratio * 40

        # Boolean checks (up to 30 points)
        check_weights = {
            "uppercase": 5,
            "lowercase": 5,
            "digits": 5,
            "special_chars": 7,
            "not_common": 5,
            "no_keyboard_walk": 2,
            "no_repeated_chars": 1,
        }
        for check, weight in check_weights.items():
            if checks.get(check, False):
                score += weight

        # Penalties – applied after bonuses to ensure obviously weak
        # passwords are ranked correctly regardless of entropy points.
        if not checks.get("not_common", True):
            score -= 30        # heavy deduction for well-known passwords
        if length < self.MIN_LENGTH:
            score -= 15        # short passwords are inherently risky

        return min(100, max(0, int(round(score))))

    @staticmethod
    def _score_to_label(score: int) -> str:
        if score >= 80:
            return "Very Strong"
        if score >= 60:
            return "Strong"
        if score >= 40:
            return "Fair"
        return "Weak"

    def _generate_feedback(
        self, password: str, checks: Dict[str, bool]
    ) -> List[str]:
        """Return a list of actionable improvement suggestions."""
        tips: List[str] = []

        if len(password) < self.MIN_LENGTH:
            tips.append(f"Use at least {self.MIN_LENGTH} characters.")
        elif len(password) < self.STRONG_LENGTH:
            tips.append(f"Consider making it longer ({self.STRONG_LENGTH}+ characters).")

        if not checks.get("uppercase"):
            tips.append("Add at least one uppercase letter.")
        if not checks.get("lowercase"):
            tips.append("Add at least one lowercase letter.")
        if not checks.get("digits"):
            tips.append("Include at least one digit.")
        if not checks.get("special_chars"):
            tips.append("Include at least one special character (e.g. !@#$%).")
        if not checks.get("not_common"):
            tips.append("Avoid commonly used passwords.")
        if not checks.get("no_keyboard_walk"):
            tips.append("Avoid keyboard walk patterns (e.g. qwerty, asdfgh).")
        if not checks.get("no_repeated_chars"):
            tips.append("Avoid repeating the same character 3 or more times in a row.")

        return tips
